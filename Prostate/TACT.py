import os
import random
import torch
import torch.nn as nn
import numpy as np
import argparse, sys, datetime
from utils.config import Logger, Seg_loss
from torch.autograd import Variable
from utils.metrics import calculate_metrics
from networks.ResUnet import ResUnet
from torch.utils.data import DataLoader
from dataloaders.ProstateSlice_dataloader import ProstateSlice_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from utils.utils import save_code
from utils.utils import sigmoid_entropy
from copy import deepcopy
import torchvision.transforms as transforms
from dataloaders import my_transforms as my_transforms
import PIL
from PIL import Image

import torch.nn.functional as F

from utils.memory_fft import Memory
import math
import torch.jit
from utils.pixel_contras_loss_our import PixelContrastLoss 
import ast
torch.set_num_threads(1)


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
    x_prob = x.sigmoid()
    x_ema_prob = x_ema.sigmoid()
    loss = (- x_prob * torch.log(x_ema_prob + 1e-8)).sum(1)
    return loss

class TACT:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, config.exp_name, config.Source_Dataset)
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        save_code(config.work_root,os.path.join(log_root, time_now, 'code'))
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        self.img_save_path = os.path.join(log_root, time_now, 'mask')
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        self.dataset_root = config.dataset_root
        
        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')



        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = ProstateSlice_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size
        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # Loss
        self.lossmap = config.lossmap
        
        self.seg_cost = Seg_loss(self.lossmap, config.lossweight)

        # GPU
        self.device = config.device

        # Prompt
        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.sparse_thr = config.sparse_thr
        self.build_model()

        self.Target_Dataset = config.Target_Dataset

        self.fft_l1 = nn.L1Loss()

        self.neighbor = 1
        self.fft_size = int(self.image_size* 0.01) if int(self.image_size* 0.01) > 1 else 1
        self.memory_bank = Memory(size=40, dimension=1*self.fft_size*self.fft_size)

        self.contrast_criterion = PixelContrastLoss(
                                    temperature=0.5,
                                    max_samples=1024,
                                    max_views=5,
                                    memory=True,
                                    memory_size=500,
                                    pixel_classes=2,
                                    dim=32)

        self.gamma_con = config.gamma_con
        self.gamma_ent = config.gamma_ent

        self.alpha_seg = config.alpha_seg
        self.alpha_con = config.alpha_con

        self.thr_t_mask = config.thr_t_mask
        self.thr_pcl_en = config.thr_pcl_en
        self.thr_pcl_pre = config.thr_pcl_pre

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, in_ch=self.in_ch).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)
        self.t_model = deepcopy(self.model)
        self.target_model = deepcopy(self.model)

        for param in self.t_model.parameters():
            param.detach_()
        
        threshold = self.sparse_thr

        disable_ub = False

        if not disable_ub:
            import SaRA.adamw2
            optimizer_cls = SaRA.adamw2.AdamW
            self.SaRA_optimizer = optimizer_cls(
                self.model.parameters(),
                betas=(0.9, 0.999),
                # weight_decay=0.0,
                lr=self.lr, # 1e-4
                eps=1e-08,
                threshold=threshold,
                progressive_iter=-1,
                lambda_rank=0,#0.005
            )
        else:
            import SaRA.adamw2
            optimizer_cls = SaRA.adamw2.AdamW
            self.SaRA_optimizer = optimizer_cls(
                self.model.parameters(),
                betas=(0.9,  0.999),
                # weight_decay=1e-2,
                lr=0.05,
                eps=1e-08,
                threshold=threshold,
                progressive_iter=100,
                lambda_rank=0,#0.005
            )
    
    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def FFT_transform(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        amp_src_shifted = amp_src.clone()

        # Zero out the center 16x16 region in the frequency domain
        center_x, center_y = imgH // 2, imgW // 2
        amp_src[:,:,center_x-1:center_x+1, center_y-1:center_y+1] = 0  # Set 16x16 region to zero
        
        amp_src_zero = amp_src.clone()

        # Reconstruct the image using the modified amplitude and the original phase
        amp_src_ = torch.fft.ifftshift(amp_src)  # Inverse FFT shift to the original position
        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        
        return src_in_trg

    def FFT_transform_swap(self, x, swap_data, freq_size):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        center_h, center_w = imgH // 2, imgW // 2
        half_size = freq_size // 2
        amp_src[:, :, center_h - half_size: center_h + half_size + 1, center_w - half_size: center_w + half_size + 1] = swap_data
        
        amp_src_ = torch.fft.ifftshift(amp_src)  # Inverse FFT shift to the original position
        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        
        return src_in_trg

    def get_low_freq(self, x, freq_size):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, _ = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        center_h, center_w = imgH // 2, imgW // 2
        half_size = freq_size // 2

        low_freq_amp = amp_src[:, :, center_h - half_size: center_h + half_size + 1, center_w - half_size: center_w + half_size + 1]

        return low_freq_amp

    def logit2masked_self(self, pred_logit, pseudo_label):
        entropy = torch.sigmoid(pred_logit)
        entropy = sigmoid_entropy(entropy)
        entropy[torch.isnan(entropy)] = 0
        entropy = ((entropy - entropy.min()) * (1 / (entropy.max() - entropy.min())))
        
        batch_size, num_classes, height, width = entropy.size()
    
        masks = torch.zeros_like(entropy)
        
        for b in range(batch_size):
            for c in range(num_classes):
                threshold = 1-torch.quantile(entropy[b, c].flatten(), 0.1)
                valid_positions = (entropy[b, c] > threshold)
                indices = valid_positions.nonzero(as_tuple=False)
                num_high = indices.size(0)
                if num_high == 0:
                    continue
                mask_size = int(valid_positions.sum().sqrt()) if int(valid_positions.sum().sqrt()) > 1 else 1
                Num = int(num_high*0.1) if int(num_high*0.1) > 1 else 1
                if num_high > Num:
                    selected_indices = indices[torch.randperm(num_high)[:Num]]
                else:
                    selected_indices = indices

                for idx in selected_indices:
                    y_pos, x_pos = idx[0].item(), idx[1].item()
                    
                    y_min = max(0, y_pos - mask_size // 2)
                    y_max = min(height, y_pos + mask_size // 2 + 1)
                    x_min = max(0, x_pos - mask_size // 2)
                    x_max = min(width, x_pos + mask_size // 2 + 1)
                    
                    masks[b, c, y_min:y_max, x_min:x_max] = 1
        
        return masks
    
    def logit2mask(self, pred_logit, threshold=0.5, on_hot=True):
        pred = torch.sigmoid(pred_logit.clone())
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        if not on_hot:
            pseudo_labels = torch.zeros([pred.size()[0], pred.size()[2], pred.size()[3]])
            if torch.cuda.is_available():
                pseudo_labels = pseudo_labels.cuda()
            pseudo_labels[pred[:, 0] == 1] = 1
            # pseudo_labels[pred[:, 1] == 1] = 2

            return pseudo_labels.unsqueeze(dim=1)

        return pred

    def logit2mask_en(self, pred_logit, threshold=0.5, en_threshold=0.75, on_hot=True):
        pseudo_label_s = torch.sigmoid(pred_logit.clone())
        pseudo_label_s[pseudo_label_s >= threshold] = 1
        pseudo_label_s[pseudo_label_s < threshold] = 0
        
        prediction_entropy = pred_logit.clone()
        prediction_entropy = self.get_entropy(prediction_entropy)
        high_entropy = prediction_entropy.clone()
        high_entropy[high_entropy >= en_threshold] = 1
        high_entropy[high_entropy < en_threshold] = 0

        # pseudo_label = high_entropy.int()
        pred = pseudo_label_s.int() | high_entropy.int()
        pred = pred.float()
        
        if not on_hot:
            pseudo_labels = torch.zeros([pred.size()[0], pred.size()[2], pred.size()[3]])
            if torch.cuda.is_available():
                pseudo_labels = pseudo_labels.cuda()
            pseudo_labels[pred[:, 0] == 1] = 1
            # pseudo_labels[pred[:, 1] == 1] = 2

            return pseudo_labels.unsqueeze(dim=1)

        return pred

    def get_entropy(self, pred_logit):
        entropy = torch.sigmoid(pred_logit)
        entropy = sigmoid_entropy(entropy)
        entropy[torch.isnan(entropy)] = 0
        entropy = ((entropy - entropy.min()) * (1 / (entropy.max() - entropy.min())))
        return entropy
    
    def sig_label_to_hard_PCL(self, sig_pls, pseudo_label_s, en_label_threshold):
        prediction_entropy = sig_pls.clone()
        prediction_entropy = self.get_entropy(prediction_entropy)
        high_entropy = prediction_entropy.clone()
        high_entropy[high_entropy >= en_label_threshold] = 1
        high_entropy[high_entropy < en_label_threshold] = 0

        # pseudo_label = high_entropy.int()
        pseudo_label = pseudo_label_s.int() | high_entropy.int()
        pseudo_label = pseudo_label.float()

        pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
        if torch.cuda.is_available():
            pseudo_labels = pseudo_labels.cuda()
        pseudo_labels[pseudo_label[:, 0] == 1] = 1
        # pseudo_labels[pseudo_label[:, 1] == 1] = 2
        pseudo_labels = pseudo_labels.unsqueeze(dim=1)
        
        low_entropy = prediction_entropy.clone()
        low_entropy_clo = prediction_entropy.clone()
        low_entropy[low_entropy_clo >= en_label_threshold] = 0
        low_entropy[low_entropy_clo < en_label_threshold] = 1
        pseudo_label_low = pseudo_label_s.int() & low_entropy.int()
        pseudo_label_low = pseudo_label_low.float()
        pseudo_labels_low = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
        if torch.cuda.is_available():
            pseudo_labels_low = pseudo_labels_low.cuda()
        pseudo_labels_low[pseudo_label_low[:, 0] == 1] = 1
        # pseudo_labels_low[pseudo_label_low[:, 1] == 1] = 2
        pseudo_labels_low = pseudo_labels_low.unsqueeze(dim=1)
        en_map = prediction_entropy.clone()
        return pseudo_labels, en_map, pseudo_labels_low

    def augment(self, x, factor):
        # Flip
        if factor == 0:
            return x.flip(-1)  # horizontal
        elif factor == 1:
            return x.flip(-2)  # vertical
        # Rotate
        elif factor == 2:
            return x.flip(-1).transpose(-2, -1)  # 90
        elif factor == 3:
            return x.flip(-1).flip(-2)  # 180
        elif factor == 4:
            return x.transpose(-2, -1).flip(-1)  # 270
        # Identity
        elif factor == 5:
            return x

    def inverse_augment(self, output, factor):
        inverse_map = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2, 5: 5}
        inverse_factor = inverse_map[factor]
        return self.augment(output, inverse_factor)
    
    def logit2mask_save(self, pred_logit, threshold=0.5, on_hot=True):
        pred = torch.sigmoid(pred_logit.clone())
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        if not on_hot:
            pseudo_labels = torch.zeros([pred.size()[0], pred.size()[2], pred.size()[3]])
            if torch.cuda.is_available():
                pseudo_labels = pseudo_labels.cuda()
            pseudo_labels[pred[:, 0] == 1] = 255
            # pseudo_labels[pred[:, 1] == 1] = 255

            return pseudo_labels.unsqueeze(dim=1)

        return pred
    
    def run(self, epoch):
        metric_dict = ['Dice', 'ASD']

        # Valid on Target
        metrics_test = [[], []]

        for batch, data in enumerate(self.target_test_loader):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            

            for tr_iter in range(self.iters):
                self.model.load_state_dict(self.target_model.state_dict())

                self.t_model.eval()
                self.model.train()

                with torch.no_grad():
                    pred_logit_t, fea_t = self.t_model(x, tsne=True)
                    pred_t = self.logit2mask_en(pred_logit_t.detach(), threshold=self.thr_t_mask, en_threshold=self.thr_pcl_en)
                    mask_en = self.logit2masked_self(pred_logit_t.detach(),pred_t.detach())

                # FFT bank
                if len(self.memory_bank.memory.keys()) >= self.neighbor:
                    low_freq = self.get_low_freq(x, self.fft_size)
                    init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor, desc=False)
                    # Update the Memory Bank
                    self.memory_bank.push(keys=low_freq.detach().cpu().numpy(), logits=low_freq.detach().cpu().numpy())
                    masked_x = self.FFT_transform_swap(x, init_data, self.fft_size)
                else:
                    low_freq = self.get_low_freq(x, self.fft_size)
                    self.memory_bank.push(keys=low_freq.detach().cpu().numpy(), logits=low_freq.detach().cpu().numpy())
                    masked_x = x.clone()
                    
                for c in range(mask_en.size(1)):
                    masked_x[:, c, :, :] *= (1 - mask_en[:, c, :, :])
                pred_logit, fea = self.model(masked_x, tsne=True)
                consistency_loss = self.fft_l1(torch.abs(torch.fft.fft2(fea)), torch.abs(torch.fft.fft2(fea_t.detach())))
                loss = self.alpha_seg*self.seg_cost(pred_logit, pred_t) + self.alpha_con*consistency_loss

                self.SaRA_optimizer.zero_grad()
                loss.backward()
                self.SaRA_optimizer.step()
                
                self.target_model.load_state_dict(self.model.state_dict())
                self.model.eval()
                with torch.no_grad():
                    pred_logit_t, fea_t = self.model(x, tsne=True)
                    pred_t = self.logit2mask(pred_logit_t.detach(), threshold=self.thr_pcl_pre)
                    
                    pred_logit_t_refine_list=[pred_logit_t.detach()]
                    for factor in [0,1,2,3,4]:
                        augmented_x = self.augment(x, factor)
                        pred_logit, _ = self.model(augmented_x, tsne=True)
                        output = self.inverse_augment(pred_logit, factor)
                        pred_logit_t_refine_list.append(output.detach())
                    pred_logit_t_refine = torch.stack(pred_logit_t_refine_list).mean(0)
                    
                self.model.train()
                
                pred_logit, pcl_fea = self.model(x, tsne=True) #, pcl=True
                t_seg_hard_PCL, t_en_map, t_seg_hard_PCL_low = self.sig_label_to_hard_PCL(pred_logit_t.detach(), pred_t.detach().clone(), self.thr_pcl_en)
                prediction = self.logit2mask(pred_logit.detach(), threshold=self.thr_pcl_pre, on_hot=False)
                contrast_loss = self.contrast_criterion(
                    F.normalize(pcl_fea, p=2, dim=1),
                    t_seg_hard_PCL_low.detach(),
                    prediction.detach(), 
                    plabel=True,
                    en_map=t_en_map.detach(),
                    low=t_seg_hard_PCL.detach()
                )
                loss = self.gamma_con*contrast_loss + self.gamma_ent*(softmax_entropy(pred_logit, pred_logit_t_refine.detach())).mean()
                self.SaRA_optimizer.zero_grad()
                loss.backward()
                self.SaRA_optimizer.step(target_phase=True)
                
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                pred_logit, feature = self.model(x, tsne=True)
                
            save_pred = self.logit2mask_save(pred_logit.detach().clone(), threshold=0.5, on_hot=False)[0][0].cpu().numpy()
            save_image = Image.fromarray(save_pred.astype(np.uint8))
            save_path = os.path.basename(data['name'][0]).replace(os.path.splitext(data['name'][0])[1], '_'+str(epoch)+os.path.splitext(data['name'][0])[1])
            domain_path = os.path.join(self.img_save_path, data['name'][0].split(self.dataset_root)[-1].split('/')[1])
            if not os.path.exists(domain_path):
                os.makedirs(domain_path) 
            save_image.save(os.path.join(domain_path,save_path))
            
            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            # print(metrics)
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]
            
        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='BIDMC',
                        help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL/')
    parser.add_argument('--Target_Dataset', type=list)
    parser.add_argument('--exp_name', type=str, default='TACT')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=384)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=1e-4) # 0.05
    parser.add_argument('--sparse_thr', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    parser.add_argument('--gamma_con', type=float, default=0.1)
    parser.add_argument('--gamma_ent', type=float, default=0.1)

    parser.add_argument('--alpha_seg', type=float, default=1.0)
    parser.add_argument('--alpha_con', type=float, default=0.1)

    parser.add_argument('--thr_t_mask', type=float, default=0.5)
    parser.add_argument('--thr_pcl_en', type=float, default=0.75)
    parser.add_argument('--thr_pcl_pre', type=float, default=0.5)

    # Loss function
    parser.add_argument('--lossmap', type=str, default=['dice', 'bce'])
    parser.add_argument('--lossweight', type=str, default="{'dice':1.0, 'bce':1.0}")

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Path
    parser.add_argument('--path_save_log', type=str, default='/root/autodl-tmp/save/logs/ProstateSlice')
    parser.add_argument('--model_root', type=str, default='/root/autodl-tmp/save/models/ProstateSlice')
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/dataset/ProstateSlice')
    parser.add_argument('--work_root', type=str, default='/root/code/TACT/Prostate')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    config = parser.parse_args()
    config.lossweight = ast.literal_eval(config.lossweight)

    config.Target_Dataset = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]
    config.Target_Dataset.remove(config.Source_Dataset)

    # set random seeds
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    TTA = TACT(config)
    
    print("++++++++++++++++++++ run TTA ++++++++++++++++++++")
    print(config.exp_name+"---"+config.Source_Dataset)
    for i in range(1):
        TTA.run(i)

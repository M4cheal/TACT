import os
import torch
import numpy as np
import argparse, sys, datetime
from utils.config import Logger
from torch.autograd import Variable
from utils.metrics import calculate_metrics
from networks.ResUnet import ResUnet
from torch.utils.data import DataLoader
from dataloaders.ProstateSlice_dataloader import ProstateSlice_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from utils.utils import save_code

torch.set_num_threads(1)


class SourceOnly:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, config.exp_name, config.Source_Dataset)
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        save_code(config.work_root,os.path.join(log_root, time_now, 'code'))
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

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

        # GPU
        self.device = config.device

        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, in_ch=self.in_ch).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

    def run(self):
        metric_dict = ['Dice', 'ASD']

        # Valid on Target
        metrics_test = [[], []]

        for batch, data in enumerate(self.target_test_loader):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            # Inference
            self.model.eval()
            with torch.no_grad():
                pred_logit, _ = self.model(x)
                
            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)
            
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
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
    parser.add_argument('--exp_name', type=str, default='Baseline_Only_Source')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=384)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

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

    config = parser.parse_args()

    config.Target_Dataset = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = SourceOnly(config)
    TTA.run()

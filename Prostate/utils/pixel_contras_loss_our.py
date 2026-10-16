import torch
import torch.nn.functional as F

from abc import ABC
from torch import nn


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-1,
                 device='cuda:0',
                 memory=True,
                 memory_size=100,
                 pixel_update_freq=1,
                 pixel_classes=3,
                 dim=256):

        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

        self.device = device
        # TODO: memory param
        # memory param
        self.memory = memory
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq

        if self.memory:
            self.segment_queue = torch.randn(pixel_classes, self.memory_size, dim)
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
            self.pixel_queue = torch.zeros(pixel_classes, self.memory_size, dim)
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

    def _dequeue_and_enqueue_signal(self, keys, labels):
        # batch_size = keys.shape[0]
        n_view = keys.shape[1]
        feat_dim = keys.shape[2]
        # 8, 128, 32, 32
        # 8, 1,   32, 32
        keys = keys.permute(2, 0, 1)
        labels = labels.unsqueeze(-1).repeat(1, n_view).unsqueeze(0)
        # 24, 3, 128 -> 128, 24, 3
        # 24 -> 1, 24, 3

        this_feat = keys.contiguous().view(feat_dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]
        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()
            lb = int(lb.item())
            # segment enqueue and dequeue
            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            ptr = int(self.segment_queue_ptr[lb])
            self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, self.pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.pixel_queue_ptr[lb])

            if ptr + K >= self.memory_size:
                self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                self.pixel_queue_ptr[lb] = 0
            else:
                self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x >= 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
    def _anchor_sampling(self, X, y_hat, y, plabel=False, en_map=None):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            # this_y_y = y[ii]

            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            """ # if plabel:
            #     this_classes_pl = []
            #     for x_cls_id in this_classes:
            #         if x_cls_id == 0:
            #             if (this_y == x_cls_id).nonzero().shape[0] > self.max_views:
            #                 this_classes_pl.append(x_cls_id)
            #         else:
            #             this_en = (en_map[ii][x_cls_id - 1] * (((this_y == x_cls_id) & (this_y_y != x_cls_id)) * 1.0))
            #             # this_en = (en_map[ii][x_cls_id - 1] * ((this_y == x_cls_id) * 1.0))
            #             k_num = int((this_en > 0).sum().item())
            #             if k_num > self.max_views:
            #                 this_classes_pl.append(x_cls_id)
            #     this_classes = this_classes_pl
            # else:
            #     this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views] """
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        # if plabel:
        #     # n_view = max(int(((en_map>=0.5)*1.0).sum(2).min().cpu().item()), self.max_views)
        #     n_view = 3

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                """ # if plabel:
                #     if cls_id == 0:
                #         hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                #         easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                #
                #         num_hard = hard_indices.shape[0]
                #         num_easy = easy_indices.shape[0]
                #
                #         if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                #             num_hard_keep = n_view // 2
                #             num_easy_keep = n_view - num_hard_keep
                #         elif num_hard >= n_view / 2:
                #             num_easy_keep = num_easy
                #             num_hard_keep = n_view - num_easy_keep
                #         elif num_easy >= n_view / 2:
                #             num_hard_keep = num_hard
                #             num_easy_keep = n_view - num_hard_keep
                #         else:
                #             print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                #             raise Exception
                #
                #         perm = torch.randperm(num_hard)
                #         hard_indices = hard_indices[perm[:num_hard_keep]]
                #         perm = torch.randperm(num_easy)
                #         easy_indices = easy_indices[perm[:num_easy_keep]]
                #         indices = torch.cat((hard_indices, easy_indices), dim=0)
                #
                #         X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                #         y_[X_ptr] = cls_id
                #         X_ptr += 1
                #     else:
                #         condition_mask = (this_y_hat == cls_id) & (this_y != cls_id)
                #         this_en = (en_map[ii][cls_id - 1] * (((this_y_hat == cls_id) & (this_y != cls_id)) * 1.0))
                #         # k_num = int((this_en > 0).sum().item())
                #         topk_indices = this_en.topk(k=n_view, dim=0, largest=True, sorted=True).indices
                #         hard_indices = topk_indices[condition_mask[topk_indices]]
                #         # hard_indices = (en_map[ii][cls_id-1]*((this_y_hat == cls_id)*1.0)).topk(k=n_view, dim=0, largest=True, sorted=True).indices
                #         # easy_indices = en_map[ii][cls_id-1].topk(k=n_view-3, dim=0, largest=False, sorted=True).indices
                #         # indices = torch.cat((hard_indices, easy_indices), dim=0)
                #         X_[X_ptr, :, :] = X[ii, hard_indices, :].squeeze(1)
                #         y_[X_ptr] = cls_id
                #         X_ptr += 1
                # else: """
                if plabel and cls_id!=0:
                    condition = (this_y_hat == cls_id) & (this_y != cls_id)
                    # 使用 torch.nonzero() 获取满足条件的样本索引
                    selected_indices = torch.nonzero(condition).squeeze(1)

                    # 使用排序函数对满足条件的样本根据熵值进行排序
                    sorted_indices = selected_indices[torch.argsort(en_map[ii][cls_id - 1][selected_indices], descending=True)]
                    
                    # hard_indices = sorted_indices[:n_view].unsqueeze(-1)
                    hard_indices = sorted_indices[:min(n_view, sorted_indices.shape[0])].unsqueeze(-1)
                else:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                # 分配硬样本和容易样本
                if num_hard >= n_view:
                    # 硬样本数量大于等于n_view，优先选择硬样本
                    num_hard_keep = n_view
                    num_easy_keep = 0
                elif num_hard + num_easy >= n_view:
                    # 硬样本不足，选择所有硬样本，并从容易样本中补充
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this should never be touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception
                """ if num_hard >= n_view and num_easy >= n_view:
                    num_hard_keep = n_view
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception """
                
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return X_, y_

    def _sample_negative(self, Q):
        class_num, memory_size, feat_size = Q.shape

        x_ = torch.zeros((class_num * memory_size, feat_size)).float().to(self.device)
        y_ = torch.zeros((class_num * memory_size, 1)).float().to(self.device)

        sample_ptr = 0
        for c in range(class_num):
            if c == 0:
                continue
            this_q = Q[c, :memory_size, :]
            x_[sample_ptr:sample_ptr + memory_size, ...] = this_q
            y_[sample_ptr:sample_ptr + memory_size, ...] = c
            sample_ptr += memory_size
        return x_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)  # (anchor_num × n_view) × 1
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)  # (anchor_num × n_view) × feat_dim

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # (anchor_num × n_view) × (anchor_num × n_view)
        mask = torch.eq(y_anchor, y_contrast.T).float().to(self.device)
        # (anchor_num × n_view) × (anchor_num × n_view)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).\
            scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None, plabel=False, en_map=None, low=None):
        if plabel:
            queue_feats = feats.detach()
            queue_labels = low.detach()
        else:
            queue_feats = feats.detach()
            queue_labels = labels.detach()
        # TODO: 改为锚点版本
        queue = self.pixel_queue if self.memory else None
        # queue = self.segment_queue if self.memory else None
        # queue = torch.cat((self.segment_queue, self.pixel_queue), dim=1) if self.memory else None

        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')

        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        if plabel:
            en_map = torch.nn.functional.interpolate(en_map, (feats.shape[2], feats.shape[3]))
            en_map = en_map.contiguous().view(batch_size, en_map.shape[1], -1)

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])



        # feats: N×(HW)×C
        # labels: N×(HW)
        # predict: N×(HW)
        feats_, labels_ = self._anchor_sampling(feats, labels, predict, plabel, en_map)
        loss = self._contrastive(feats_, labels_, queue=queue)

        if self.memory:
            # queue_feats = feats_.detach()
            # queue_labels = labels_.detach()
            self._dequeue_and_enqueue(queue_feats, queue_labels)

        return loss
import torch
import torch.nn as nn
import numpy as np
from lib.core.config import cfg


class WCEL_Loss(nn.Module):
    """
    Weighted Cross-entropy Loss Function.
    """
    def __init__(self):
        super(WCEL_Loss, self).__init__()
        self.weight = cfg.DATASET.WCE_LOSS_WEIGHT
        self.weight /= np.sum(self.weight, 1, keepdims=True)

    def forward(self, pred_logit, gt_bins, gt):
        self.weight = torch.tensor(self.weight, dtype=torch.float32, device=pred_logit.device)
        classes_range = torch.arange(cfg.MODEL.DECODER_OUTPUT_C, device=gt_bins.device, dtype=gt_bins.dtype)
        log_pred = torch.nn.functional.log_softmax(pred_logit, 1)
        log_pred = torch.t(torch.transpose(log_pred, 0, 1).reshape(log_pred.size(1), -1))

        gt_reshape = gt_bins.reshape(-1, 1)
        one_hot = (gt_reshape == classes_range).to(dtype=torch.float, device=pred_logit.device)
        weight = torch.matmul(one_hot, self.weight)
        weight_log_pred = weight * log_pred

        valid_pixels = torch.sum(gt > 0.).to(dtype=torch.float, device=pred_logit.device)
        loss = -1 * torch.sum(weight_log_pred) / valid_pixels
        return loss
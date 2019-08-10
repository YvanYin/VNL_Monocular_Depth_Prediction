import torch
import torch.nn as nn
import numpy as np
from . import lateral_net as lateral_net
from lib.utils.net_tools import get_func
from lib.models.image_transfer import bins_to_depth, kitti_merge_imgs
from lib.core.config import cfg


class MetricDepthModel(nn.Module):
    def __init__(self):
        super(MetricDepthModel, self).__init__()
        self.loss_names = ['Weighted_Cross_Entropy', 'Virtual_Normal']
        self.depth_model = DepthModel()

    def forward(self, data):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.a_real = data['A'].cuda()
        self.b_fake_logit, self.b_fake_softmax = self.depth_model(self.a_real)
        return {'b_fake_logit': self.b_fake_logit, 'b_fake_softmax': self.b_fake_softmax}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
            pred_depth = bins_to_depth(out['b_fake_softmax'])
            b_fake = pred_depth
            return {'b_fake': b_fake}

    def inference_kitti(self, data):
        #crop kitti images into 3 parts
        with torch.no_grad():
            self.a_l_real = data['A_l'].cuda()
            _, b_l_classes = self.depth_model(self.a_l_real)
            self.b_l_fake = bins_to_depth(b_l_classes)

            self.a_m_real = data['A_m'].cuda()
            _, b_m_classes = self.depth_model(self.a_m_real)
            self.b_m_fake = bins_to_depth(b_m_classes)

            self.a_r_real = data['A_r'].cuda()
            _, b_r_classes = self.depth_model(self.a_r_real)
            self.b_r_fake = bins_to_depth(b_r_classes)

            out = kitti_merge_imgs(self.b_l_fake, self.b_m_fake, self.b_r_fake, torch.squeeze(data['B_raw']).shape, data['crop_lmr'])
            return {'b_fake': out}


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        bottom_up_model = 'lateral_net.lateral_' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(bottom_up_model)()
        self.decoder_modules = lateral_net.fcn_topdown(cfg.MODEL.ENCODER)

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax


def cal_params(model):
    model_dict = model.state_dict()
    paras = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum = 0

    for key in model_dict.keys():
        print(key)
        if 'layer5' not in key:
            if 'running' not in key:
                print(key)
                ss = model_dict[key].size()
                temp = 1
                for s in ss:
                    temp = temp * s
                print(temp)
                sum = sum + temp
    print(sum)
    print(paras)

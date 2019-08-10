import os
import torch
from lib.core.config import cfg
import numpy as np
import logging
logger = logging.getLogger(__name__)


def load_pretrained_imagenet_resnext_weights(model):
    """Load pretrained weights
    Args:
        num_layers: 50 for res50 and so on.
        model: the generalized rcnnn module
    """
    model_state_dict = model.state_dict()
    weights_file = os.path.join(cfg.ROOT_DIR, cfg.MODEL.MODEL_REPOSITORY, 'MobileNetV2-ImageNet', cfg.MODEL.PRETRAINED_WEIGHTS)
    pretrianed_state_dict = convert_state_dict(torch.load(weights_file), model_state_dict)

    for k, v in pretrianed_state_dict.items():
        if k in model_state_dict.keys():
            model_state_dict[k].copy_(pretrianed_state_dict[k])
        else:
            logger.info('Weight %s is not in MobileNetV2 model.' % k)
    logger.info('Pretrained MobileNetV2 weight has been loaded')


def convert_state_dict(src_dict, model_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    res_block_n = np.array([1, 4, 7, 14, 18])
    for k, v in src_dict.items():
        toks = k.split('.')
        id_n = int(toks[1])
        if id_n < 18 and '17.conv.7' not in k and 'classifier' not in k:
            res_n = np.where(res_block_n > id_n)[0][0] + 1
            n = res_n - 2 if res_n >= 2 else 0
            res_n_m = 0 if id_n - res_block_n[n] < 0 else id_n - res_block_n[n]
            toks[0] = 'res%s' % res_n
            toks[1] = '%s' % res_n_m
            name = '.'.join(toks)
            dst_dict[name] = v
    return dst_dict

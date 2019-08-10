import os
import torch
from lib.core.config import cfg
import logging
logger = logging.getLogger(__name__)

def load_pretrained_imagenet_resnext_weights(model):
    """Load pretrained weights
    Args:
        num_layers: 50 for res50 and so on.
        model: the generalized rcnnn module
    """
    weights_file = os.path.join(cfg.ROOT_DIR, cfg.MODEL.MODEL_REPOSITORY, 'ResNeXt-ImageNet', cfg.MODEL.PRETRAINED_WEIGHTS)
    pretrianed_state_dict = convert_state_dict(torch.load(weights_file))

    model_state_dict = model.state_dict()

    for k, v in pretrianed_state_dict.items():
        if k in model_state_dict.keys():
            model_state_dict[k].copy_(pretrianed_state_dict[k])
        else:
            print('Weight %s is not in ResNeXt model.' % k)
    logger.info('Pretrained ResNeXt weight has been loaded')


def convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    res_id = 1
    map1 = ['conv1.', 'bn1.', ' ', 'conv2.', 'bn2.']
    map2 = [[' ', 'conv3.', 'bn3.'], ['shortcut.conv.', 'shortcut.bn.']]
    for k, v in src_dict.items():
        toks = k.split('.')
        if int(toks[0]) == 0:
            name = 'res%d.' % res_id + 'conv1.' + toks[-1]
        elif int(toks[0]) == 1:
            name = 'res%d.' % res_id + 'bn1.' + toks[-1]
        elif int(toks[0]) >=4 and int(toks[0]) <= 7:
            name_res = 'res%d.%d.' % (int(toks[0])-2, int(toks[1]))
            if len(toks) == 7:
                name = name_res + map1[int(toks[-2])] + toks[-1]
            elif len(toks) == 6:
                name = name_res + map2[int(toks[-3])][int(toks[-2])] + toks[-1]
        else:
            continue
        dst_dict[name] = v

    return dst_dict

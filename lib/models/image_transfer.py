import torch
import cv2
from lib.core.config import cfg
import numpy as np
import torch.nn.functional as F

def bins_to_depth(depth_bin):
    """
    Transfer n-channel discrate depth bins to 1-channel conitnuous depth
    :param depth_bin: n-channel output of the network, [b, c, h, w]
    :return: 1-channel depth, [b, 1, h, w]
    """
    if type(depth_bin).__module__ != torch.__name__:
        depth_bin = torch.tensor(depth_bin, dtype=torch.float32).cuda()
    depth_bin = depth_bin.permute(0, 2, 3, 1) #[b, h, w, c]
    if type(cfg.DATASET.DEPTH_BIN_BORDER).__module__ != torch.__name__:
        cfg.DATASET.DEPTH_BIN_BORDER = torch.tensor(cfg.DATASET.DEPTH_BIN_BORDER, dtype=torch.float32).cuda()
    depth = depth_bin * cfg.DATASET.DEPTH_BIN_BORDER
    depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
    depth = 10 ** depth
    depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]
    return depth


def resize_image(img, size):
    if type(img).__module__ != np.__name__:
        img = img.cpu().numpy()
    img = cv2.resize(img, (size[1], size[0]))
    return img


def kitti_merge_imgs(left, middle, right, img_shape, crops):
    """
    Merge the splitted left, middle and right parts together.
    """
    left = torch.squeeze(left)
    right = torch.squeeze(right)
    middle = torch.squeeze(middle)
    out = torch.zeros(img_shape, dtype=left.dtype, device=left.device)
    crops = torch.squeeze(crops)
    band = 5

    out[:, crops[0][0]:crops[0][0] + crops[0][2] - band] = left[:, 0:left.size(1)-band]
    out[:, crops[1][0]+band:crops[1][0] + crops[1][2] - band] += middle[:, band:middle.size(1)-band]
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[2][0] + crops[2][2]] += right[:, crops[1][0] + crops[1][2] - 2*band-crops[2][0]:]

    out[:, crops[1][0]+band:crops[0][0] + crops[0][2] - band] /= 2.0
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[1][0] + crops[1][2] - band] /= 2.0
    out = out.cpu().numpy()

    return out
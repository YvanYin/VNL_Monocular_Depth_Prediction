from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.metric_depth_model import MetricDepthModel
from lib.utils.net_tools import load_ckpt
import torch
import os
import numpy as np
from lib.core.config import cfg, merge_cfg_from_file
import matplotlib.pyplot as plt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms
logger = setup_logging(__name__)
import cv2


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = MetricDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    path = '/home/yvan/3Dvideos_testing/3Dvideo_1/1'
    files = os.listdir(path)
    imgs_list = [i for i in files if '.py' not in i]
    for i in imgs_list:
        print(i)
        with torch.no_grad():
            img = cv2.imread(os.path.join(path, i))
            img_resize = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            pred_depth, _ = model.module.depth_model(img_torch)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_scale = (pred_depth / pred_depth.max() * 60000).astype(np.uint16)

            model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
            #image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name + '_randoms')
            image_dir = os.path.join(path + '_randoms')
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            plt.imsave(os.path.join(image_dir, i.replace('.', '-d.')), pred_depth, cmap='rainbow')
            cv2.imwrite(os.path.join(image_dir, i), img)
            cv2.imwrite(os.path.join(image_dir, i.split('.')[0] + '-raw.png'), pred_depth_scale)
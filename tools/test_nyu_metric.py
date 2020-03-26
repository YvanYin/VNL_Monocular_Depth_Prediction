import os
import cv2
import torch
import numpy as np
from lib.core.config import cfg
from lib.utils.net_tools import load_ckpt
from tools.parse_arg_test import TestOptions
from lib.core.config import merge_cfg_from_file
from data.load_dataset import CustomerDataLoader
from lib.models.image_transfer import resize_image
from lib.utils.evaluate_depth_error import evaluate_err
from lib.models.metric_depth_model import MetricDepthModel
from lib.utils.logging import setup_logging, SmoothedValue
import matplotlib.pyplot as plt

logger = setup_logging(__name__)

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

    # test
    smoothed_absRel = SmoothedValue(test_datasize)
    smoothed_rms = SmoothedValue(test_datasize)
    smoothed_logRms = SmoothedValue(test_datasize)
    smoothed_squaRel = SmoothedValue(test_datasize)
    smoothed_silog = SmoothedValue(test_datasize)
    smoothed_silog2 = SmoothedValue(test_datasize)
    smoothed_log10 = SmoothedValue(test_datasize)
    smoothed_delta1 = SmoothedValue(test_datasize)
    smoothed_delta2 = SmoothedValue(test_datasize)
    smoothed_delta3 = SmoothedValue(test_datasize)
    smoothed_whdr = SmoothedValue(test_datasize)

    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_squaRel': smoothed_squaRel, 'err_rms': smoothed_rms,
                         'err_silog': smoothed_silog, 'err_logRms': smoothed_logRms, 'err_silog2': smoothed_silog2,
                         'err_delta1': smoothed_delta1, 'err_delta2': smoothed_delta2, 'err_delta3': smoothed_delta3,
                         'err_log10': smoothed_log10, 'err_whdr': smoothed_whdr}

    for i, data in enumerate(data_loader):
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])
        img_path = data['A_paths']
        invalid_side = data['invalid_side'][0]
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth = pred_depth / data['ratio'].cuda() # scale the depth
        pred_depth = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        smoothed_criteria = evaluate_err(pred_depth, data['B_raw'], smoothed_criteria, mask=(45, 471, 41, 601), scale=10.)

        # save images
        model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
        image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        img_name = img_path[0].split('/')[-1]
        #plt.imsave(os.path.join(image_dir, 'd_' + img_name), pred_depth, cmap='rainbow')
        #cv2.imwrite(os.path.join(image_dir, 'rgb_' + img_name), data['A_raw'].numpy().squeeze())

        print('processing (%04d)-th image... %s' % (i, img_path))


    print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
    print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
        smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))
    print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
    print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
    print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
    print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
    print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
    print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
    print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))

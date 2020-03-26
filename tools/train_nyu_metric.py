from data.load_dataset import CustomerDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err
from lib.models.metric_depth_model import *
from lib.core.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue
import math
import traceback
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions

logger = setup_logging(__name__)

import sys
import datetime
import time
import os

# st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if not os.path.exists(os.path.join(BASE_DIR, 'trains/bt4')):
#     os.makedirs(os.path.join(BASE_DIR, 'trains/bt4'))
# sys.stdout = open(os.path.join(BASE_DIR, 'trains/bt4', st + '.log'), 'w')


def train(train_dataloader, model, epoch, loss_func,
          optimizer, scheduler, training_stats, val_dataloader=None, val_err=[], ignore_step=-1):
    """
    Train the model in steps
    """
    model.train()
    epoch_steps = math.ceil(len(train_dataloader) / cfg.TRAIN.BATCHSIZE)
    base_steps = epoch_steps * epoch + ignore_step if ignore_step != -1 else epoch_steps * epoch
    for i, data in enumerate(train_dataloader):
        if ignore_step != -1 and i > epoch_steps - ignore_step:
            return
        scheduler.step()  # decay lr every iteration
        training_stats.IterTic()
        out = model(data)
        losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data, epoch)
        optimizer.optim(losses)
        step = base_steps + i + 1
        training_stats.UpdateIterStats(losses)
        training_stats.IterToc()
        training_stats.LogIterStats(step, epoch, optimizer.optimizer, val_err[0])

        # validate the model
        if step % cfg.TRAIN.VAL_STEP == 0 and step != 0 and val_dataloader is not None:
            model.eval()
            val_err[0] = val(val_dataloader, model)
            # training mode
            model.train()

        # save checkpoint
        if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0:
            save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])


def val(val_dataloader, model):
    """
    Validate the model.
    """
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel}
    for i, data in enumerate(val_dataloader):
        invalid_side = data['invalid_side'][0]
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth = pred_depth / data['ratio'].cuda()
        pred_depth = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        smoothed_criteria = validate_err(pred_depth, data['B_raw'], smoothed_criteria, (45, 471, 41, 601))
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue()}


if __name__=='__main__':
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()
    train_opt.print_options(train_args)

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0
    val_opt.print_options(val_args)

    train_dataloader = CustomerDataLoader(train_args)
    train_datasize = len(train_dataloader)
    gpu_num = torch.cuda.device_count()
    merge_cfg_from_file(train_args)

    val_dataloader = CustomerDataLoader(val_args)
    val_datasize = len(val_dataloader)

    # Print configs
    print_configs(cfg)

    # tensorboard logger
    if train_args.use_tfboard:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # training status for logging
    training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,
                                   tblogger if train_args.use_tfboard else None)

    # total iterations
    total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters
    cfg.TRAIN.GPU_NUM = gpu_num

    # load model
    model = MetricDepthModel()

    if gpu_num != -1:
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))
        model.cuda()

    optimizer = ModelOptimizer(model)
    loss_func = ModelLoss()

    val_err = [{'abs_rel': 0}]
    ignore_step = -1

    lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lr_optim_lambda)

    # load checkpoint
    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        ignore_step = train_args.start_step - train_args.start_epoch * math.ceil(train_datasize / train_args.batchsize)

    if gpu_num != -1:
        model = torch.nn.DataParallel(model)
    try:
        for epoch in range(train_args.start_epoch, train_args.epoch):
            # training
            train(train_dataloader, model, epoch, loss_func, optimizer, scheduler, training_stats,
                  val_dataloader, val_err, ignore_step)
            ignore_step = -1

    except (RuntimeError, KeyboardInterrupt):

        logger.info('Save ckpt on exception ...')
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard:
            tblogger.close()

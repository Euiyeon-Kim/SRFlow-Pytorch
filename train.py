import os
import math
from collections import defaultdict

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from utils import util
from utils.metric import Measure
from config import config as _config
from models.SRFlow import SRFlowModel
from data.dataloader import get_train_dataloader, get_valid_dataloader


def train(config):
    log_dir = f'{config.path.exp_path}/logs'
    state_dir = f'{config.path.exp_path}/state'
    chkpt_dir = f'{config.path.exp_path}/chkpt'
    valid_dir = f'{config.path.exp_path}/valids'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Define Model & Dataloader
    model = SRFlowModel(config=config)
    train_dataloader = get_train_dataloader(config)
    valid_dataloader = get_valid_dataloader(config)

    cur_step = 0
    start_epoch = 0
    train_size = int(math.ceil(len(train_dataloader) / config.dataset.batch_size))
    total_iters = config.train.n_iter
    total_epochs = int(math.ceil(total_iters / train_size))

    measure = Measure(use_gpu=True)

    print(colored(f'Start training from epoch: {start_epoch}, iter: {cur_step}', 'green'))

    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_dataloader):
            cur_step += 1
            if cur_step > total_iters:
                break

            model.feed_data(train_data)
            model.update_learning_rate(cur_step, warmup_iter=config.train.warmup_iter)

            try:
                nll = model.optimize_parameters(cur_step)
                print(f"Train Epoch {epoch} / Iter {cur_step} || {nll}")
                writer.add_scalar("train/nll", nll, cur_step)
                writer.add_scalar("train/lr", model.get_current_lr(), cur_step)
                writer.flush()

            except RuntimeError as e:
                print("Skipping ERROR caught in nll = model.optimize_parameters(cur_step): ")
                print(e)

            if cur_step % config.train.val_freq == 0:
                os.makedirs(f'{valid_dir}/{cur_step}', exist_ok=True)
                avg_psnr = defaultdict(lambda: 0.0)
                avg_ssim = defaultdict(lambda: 0.0)
                avg_lpips = defaultdict(lambda: 0.0)
                idx = 0
                nlls = []
                for val_data in valid_dataloader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['PATH'][0]))[0]
                    os.makedirs(f'{valid_dir}/{cur_step}/{img_name}', exist_ok=True)

                    model.feed_data(val_data)
                    nll = model.test()
                    if nll is None:
                        nll = 0
                    nlls.append(nll)
                    visuals = model.get_current_visuals()

                    sr_imgs = {}
                    # Save SR images
                    if hasattr(model, 'heats'):
                        for heat in model.heats:
                            for i in range(model.n_sample):
                                sr_img = util.tensor2img(visuals['SR', heat, i])
                                sr_imgs[heat] = sr_img
                                save_img_path = f'{valid_dir}/{cur_step}/{img_name}/Heat({heat})_{i}_SR.png'
                                util.save_img(sr_img, save_img_path)
                    else:
                        sr_img = util.tensor2img(visuals['SR'])
                        sr_imgs['1.0'] = sr_img
                        save_img_path = f'{valid_dir}/{cur_step}/{img_name}/SR.png'
                        util.save_img(sr_img, save_img_path)

                    # Save LQ images
                    save_img_path_lq = f'{valid_dir}/{cur_step}/{img_name}/LQ.png'
                    lq_img = util.tensor2img(visuals['LQ'])
                    util.save_img(lq_img, save_img_path_lq)

                    # Save GT images
                    gt_img = util.tensor2img(visuals['GT'])
                    save_img_path_gt = f'{valid_dir}/{cur_step}/{img_name}/GT.png'
                    util.save_img(gt_img, save_img_path_gt)

                    for k, v in sr_imgs.items():
                        psnr, ssim, lpips = measure.measure(v, gt_img)
                        avg_psnr[k] += psnr
                        avg_ssim[k] += ssim
                        avg_lpips[k] += lpips

                avg_nll = sum(nlls) / len(nlls)
                print(f'Valid || NLL:{avg_nll}')
                writer.add_scalar('valid/nll', avg_nll, cur_step)

                for k, v in avg_psnr.items():
                    writer.add_scalar(f'{k}/psnr', avg_psnr[k]/idx, cur_step)
                    writer.add_scalar(f'{k}/ssim', avg_ssim[k]/idx, cur_step)
                    writer.add_scalar(f'{k}/lpips', avg_lpips[k]/idx, cur_step)

                writer.flush()

            if cur_step % config.train.save_freq == 0:
                model.save(cur_step)
                model.save_training_state(epoch, cur_step)


    writer.close()


if __name__ == '__main__':
    train(_config)


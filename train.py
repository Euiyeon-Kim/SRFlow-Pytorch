import os
import math

import torch
from termcolor import colored

from config import config as _config
from models.SRFlow import SRFlowModel
from utils.timer import Timer, TickTock
from data.dataloader import get_train_dataloader


def train(config):
    log_dir = f'{config.path.exp_path}/logs'
    chkpt_dir = f'{config.path.exp_path}/chkpt'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    # Define Model & Dataloader
    model = SRFlowModel(config=config)
    train_dataloader = get_train_dataloader(config)

    cur_step = 0
    start_epoch = 0
    train_size = int(math.ceil(len(train_dataloader) / config.dataset.batch_size))
    total_iters = config.train.n_iter
    total_epochs = int(math.ceil(total_iters / train_size))

    timer = Timer()
    timer_data = TickTock()
    print(colored(f'Start training from epoch: {start_epoch}, iter: {cur_step}', 'green'))

    for epoch in range(start_epoch, total_epochs + 1):
        timer_data.tick()

        for _, train_data in enumerate(train_dataloader):
            timer_data.tock()

            cur_step += 1
            if cur_step > total_iters:
                break

            model.feed_data(train_data)
            model.update_learning_rate(cur_step, warmup_iter=config.train.warmup_iter)

            try:
                nll = model.optimize_parameters(cur_step)
                print(nll)

            except RuntimeError as e:
                print("Skipping ERROR caught in nll = model.optimize_parameters(cur_step): ")
                print(e)

            exit()


if __name__ == '__main__':
    train(_config)


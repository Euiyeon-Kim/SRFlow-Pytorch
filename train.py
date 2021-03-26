import os
import math

import torch
from termcolor import colored

from config import config as _config
from models.SRFlow import SRFlowModel
from utils.timer import Timer, TickTock


def train(config):
    log_dir = f'{config.path.exp_path}/logs'
    chkpt_dir = f'{config.path.exp_path}/chkpt'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    current_step = 0
    start_epoch = 0
    # train_size = int(math.ceil(len(train_set) / config.dataset.batch_size))
    train_size = int(math.ceil(100 / config.dataset.batch_size))
    total_iters = config.train.n_iter
    total_epochs = int(math.ceil(total_iters / train_size))

    model = SRFlowModel(config=config)
    model.feed_data(train_data)
    model.update_learning_rate(current_step, warmup_iter=config.train.warmup_iter)
    try:
        nll = model.optimize_parameters(current_step)
    except RuntimeError as e:
        print("Skipping ERROR caught in nll = model.optimize_parameters(current_step): ")
        print(e)

    # timer = Timer()
    # print(colored(f'Start training from epoch: {start_epoch}, iter: {current_step}', 'green'))
    # timerData = TickTock()
    # for epoch in range(start_epoch, total_epochs + 1):
    #     timerData.tick()
    #     for _, train_data in enumerate(train_loader)



if __name__ == '__main__':
    train(_config)


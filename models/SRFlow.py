# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE
import os
from collections import OrderedDict
from termcolor import colored

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models.SRFlowArch import SRFlowNet
import models.lr_scheduler as lr_scheduler


class SRFlowModel:
    def __init__(self, config):
        super(SRFlowModel, self).__init__()
        self.config = config
        self.schedulers = []
        self.optimizers = []
        self.optimizer_G = None
        self.device = torch.device('cuda' if config.train.gpu_ids is not None else 'cpu')

        self.heats = config.val.heats
        self.n_sample = config.val.n_sample
        self.hr_size = config.patch_size[0]
        self.lr_size = self.hr_size // config.scale
        self.var_L = None
        self.real_H = None

        self.netG = SRFlowNet(config=config, scale=config.scale).to(self.device)
        if config.train.dist:
            self.rank = torch.distributed.get_rank()
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.rank = -1  # non dist training
            self.netG = DataParallel(self.netG)

        if config.train.resume:
            self.load()

        if config.is_train:
            self.netG.train()
            self.init_optimizer_and_scheduler()
            self.log_dict = OrderedDict()

    def init_optimizer_and_scheduler(self):
        # optimizers
        optim_params_rrdb = []
        optim_params_other = []

        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            # print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_rrdb.append(v)
                    # print('opt', k)
                else:
                    optim_params_other.append(v)
                if self.rank <= 0:
                    colored(f'Params [{k}] will not optimize.', 'red')

        # print('rrdb params', len(optim_params_rrdb))

        # Set optimizer
        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": self.config.train.lr_G, 'beta1': self.config.train.beta1,
                 'beta2': self.config.train.beta2, 'weight_decay': self.config.train.weight_decay_G},
                {"params": optim_params_rrdb, "lr": self.config.train.lr_RRDB,
                 'beta2': self.config.train.beta2, 'weight_decay': self.config.train.weight_decay_G},
            ],
        )
        self.optimizers.append(self.optimizer_G)

        # Set scheduler
        if self.config.train.lr_scheme == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLRRestart(optimizer, self.config.train.lr_steps,
                                                    restarts=self.config.train.restarts,
                                                    weights=self.config.train.restart_weights,
                                                    gamma=self.config.train.lr_gamma,
                                                    clear_state=self.config.train.clear_state,
                                                    lr_steps_invese=self.config.train.lr_steps_inverse)
                )

        elif self.config.train.lr_scheme == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRRestart(
                        optimizer, self.config.train.T_period, eta_min=self.config.train.eta_min,
                        restarts=self.config.train.restarts, weights=self.config.train.restart_weights)
                )

        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # Set up warm up learning rate
        if cur_iter < warmup_iter:
            # Get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # Modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # Set learning rate
            self._set_lr(warm_up_lr_l)

    def feed_data(self, data, need_gt=True):
        self.var_L = data['LQ'].to(self.device, torch.float)  # LQ
        if need_gt:
            self.real_H = data['GT'].to(self.device, torch.float)  # GT

    def optimize_parameters(self, step):
        train_RRDB_delay = self.config.netG.RRDBencoder.train_delay
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.config.train.n_iter) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB()

        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        losses = {}
        weight_fl = self.config.train.weight_fl or 1
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        weight_l1 = self.config.train.weight_l1 or 0
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            l1_loss = (sr - self.real_H).abs().mean()
            losses['l1_loss'] = l1_loss * weight_l1

        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer_G.step()

        mean = total_loss.item()
        return mean

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed:
            torch.manual_seed(seed)
        if self.config.netG.flow.split.enable:
            C = self.netG.module.flowUpsamplerNet.C
            H = int(self.config.scale * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
            W = int(self.config.scale * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = self.config.netG.flow.L or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z

    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            print('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            print(s)

    def load_network(self, load_path, network, strict=True, submodule=None):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        if not (submodule is None or submodule.lower() == 'none'.lower()):
            network = network.__getattr__(submodule)
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def load(self):
        if self.config.path.G_weight_path is not None:
            if os.path.exists(self.config.path.netG_weight_path):
                self.load_network(self.config.path.netG_weight_path, self.netG, strict=True, submodule=None)
                print(colored(f'Loaded generator weight from [{self.config.path.netG_weight_path}]', 'green'))
            else:
                raise FileNotFoundError(
                    colored(f'Generator weight doesn\'t exists [{self.config.path.netG_weight_path}]', 'red'))
        else:
            raise ValueError(colored(f'Need pretrained SR network weights to load', 'red'))

    def save_network(self, network, network_label, iter_label):
        save_path = f'{self.config.path.exp_path}/chkpt/{iter_label}_{network_label}'
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

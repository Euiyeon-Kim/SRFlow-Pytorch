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


class SRFlowModel:
    def __init__(self, config):
        super(SRFlowModel, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if config.train.gpu_ids is not None else 'cpu')

        self.heats = config.val.heats
        self.n_sample = config.val.n_sample
        self.hr_size = config.patch_size[0]
        self.lr_size = self.hr_size // config.scale

        self.netG = SRFlowNet(config=config, scale=config.scale).to(self.device)
        if config.train.dist:
            self.rank = torch.distributed.get_rank()
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.rank = -1  # non dist training
            self.netG = DataParallel(self.netG)

        if config.train.resume:
            self.load()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def feed_data(self, data, need_gt=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_gt:
            self.real_H = data['GT'].to(self.device)  # GT

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
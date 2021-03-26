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
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import numpy as np
import torch
from torch import nn as nn

import models.modules.Split
from models.modules import flow
from models.modules.Split import Split2d
from models.modules.glow_arch import f_conv2d_bias
from models.modules.FlowStep import FlowStep


class FlowUpsamplerNet(nn.Module):
    def __init__(self, K, actnorm_scale=1.0, flow_coupling=None, LU_decomposed=False, config=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.L = config.netG.flow.L
        self.K = config.netG.flow.K
        if isinstance(self.K, int):
            self.K = [K for K in [K, ] * (self.L + 1)]

        self.config = config
        H, W, self.C = config.patch_size
        self.check_image_shape()

        if config.scale == 16:
            self.levelToName = {0: 'fea_up16', 1: 'fea_up8', 2: 'fea_up4', 3: 'fea_up2', 4: 'fea_up1'}
        elif config.scale == 8:
            self.levelToName = {0: 'fea_up8', 1: 'fea_up4', 2: 'fea_up2', 3: 'fea_up1', 4: 'fea_up0'}
        elif config.scale == 4:
            self.levelToName = {0: 'fea_up4', 1: 'fea_up2', 2: 'fea_up1', 3: 'fea_up0', 4: 'fea_up-1'}

        affine_in_nc = self.get_affine_inch()
        flow_permutation = config.netG.flow.flow_permutation
        norm_config = None # opt_get(opt, ['network_G', 'flow', 'norm'])

        conditional_channels = {}
        n_rrdb = self.get_n_rrdb_channels()
        n_bypass_channels = None # opt_get(opt, ['network_G', 'flow', 'levelConditional', 'n_channels'])
        conditional_channels[0] = n_rrdb
        for level in range(1, self.L + 1):
            # Level 1 gets conditionals from 2, 3, 4 => L - level
            # Level 2 gets conditionals from 3, 4
            # Level 3 gets conditionals from 4
            # Level 4 gets conditionals from None
            n_bypass = 0 if n_bypass_channels is None else (self.L - level) * n_bypass_channels
            conditional_channels[level] = n_rrdb + n_bypass

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            self.arch_additional_flow_affine(H, LU_decomposed, W, actnorm_scale)
            self.arch_flow_step(H, self.K[level], LU_decomposed, W, actnorm_scale, flow_coupling, flow_permutation,
                                norm_config, config, n_conditinal_channels=conditional_channels[level])

            # Split
            self.arch_split(H, W, level, self.L)

        if config.netG.flow.split.enable:
            self.f = f_conv2d_bias(affine_in_nc, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affine_in_nc, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = config.patch_size[0] / H
        self.scaleW = config.patch_size[1] / W

    def get_n_rrdb_channels(self):
        blocks = self.config.netG.RRDBencoder.stackRRDB.blocks
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def arch_flow_step(self, H, K, LU_decomposed, W, actnorm_scale, flow_coupling, flow_permutation,
                       norm_config, config, n_conditinal_channels=None):

        cond_aff_config = self.get_cond_aff_setting()
        if cond_aff_config is not None:
            cond_aff_config['in_channels_rrdb'] = n_conditinal_channels

        for k in range(K):
            position_name = get_position_name(H, self.config.scale)
            if norm_config:
                norm_config['position'] = position_name

            self.layers.append(
                FlowStep(in_ch=self.C,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         position=position_name,
                         LU_decomposed=LU_decomposed,
                         ac_config=cond_aff_config,
                         norm_config=norm_config,
                         config=config))
            self.output_shapes.append([-1, self.C, H, W])

    def get_cond_aff_setting(self):
        cond_aff = self.config.netG.flow.condAff or None
        cond_aff = self.config.netG.flow.condFtAffine or cond_aff
        return cond_aff

    def arch_split(self, H, W, L, levels):
        correct_splits = self.config.netG.flow.split.correct_splits
        correction = 0 if correct_splits else 1
        if self.config.netG.flow.split.enable and L < levels - correction:
            logs_eps = self.config.netG.flow.split.logs_eps
            consume_ratio = self.config.netG.flow.split.consume_ratio
            position_name = get_position_name(H, self.config.scale)
            position = position_name if self.config.netG.flow.split.conditional else None
            cond_channels = self.config.netG.flow.split.cond_channels or 0
            t = self.config.netG.flow.split.type or 'Split2d'

            if t == 'Split2d':
                split = models.modules.Split.Split2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                     cond_channels=cond_channels, consume_ratio=consume_ratio,
                                                     config=self.config)
            self.layers.append(split)
            self.output_shapes.append([-1, split.num_channels_pass, H, W])
            self.C = split.num_channels_pass

    def arch_additional_flow_affine(self, H, LU_decomposed, W, actnorm_scale):
        if 'additionalFlowNoAffine' in self.config.netG.flow:
            n_additional_flow_no_affine = int(self.config.netG.flow.additionalFlowNoAffine)
            for _ in range(n_additional_flow_no_affine):
                self.layers.append(
                    FlowStep(in_ch=self.C,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed,
                             config=self.config))
                self.output_shapes.append([-1, self.C, H, W])

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(flow.SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def get_affine_inch(self):
        affine_in_nc = self.config.netG.RRDBencoder.stackRRDB.blocks
        affine_in_nc = (len(affine_in_nc) + 1) * 64
        return affine_in_nc

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, \
            "image_shape should be HWC, like (64, 64, 3) self.C == 1 or self.C == 3"

    def forward(self, gt=None, rrdb_results=None, z=None, epses=None, logdet=0.,
                reverse=False, eps_std=None, y_onehot=None):
        if reverse:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses
            sr, logdet = self.decode(rrdb_results, z, eps_std, epses=epses_copy, logdet=logdet, y_onehot=y_onehot)
            return sr, logdet
        else:
            assert gt is not None
            assert rrdb_results is not None
            z, logdet = self.encode(gt, rrdb_results, logdet=logdet, epses=epses, y_onehot=y_onehot)
            return z, logdet

    def encode(self, gt, rrdb_results, logdet=0.0, epses=None, y_onehot=None):
        fl_fea = gt
        reverse = False
        level_conditionals = {}
        bypasses = {}

        L = self.config.netG.flow.L
        for level in range(1, L + 1):
            bypasses[level] = torch.nn.functional.interpolate(gt, scale_factor=2 ** -level,
                                                              mode='bilinear', align_corners=False)

        for layer, shape in zip(self.layers, self.output_shapes):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))

            if level > 0 and level not in level_conditionals.keys():
                level_conditionals[level] = rrdb_results[self.levelToName[level]]

            level_conditionals[level] = rrdb_results[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, rrdb_results=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d(epses, fl_fea, layer, logdet, reverse, level_conditionals[level],
                                                      y_onehot=y_onehot)
            else:
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse)

        z = fl_fea

        if not isinstance(epses, list):
            return z, logdet

        epses.append(z)
        return epses, logdet

    def forward_preFlow(self, fl_fea, logdet, reverse):
        if hasattr(self, 'preFlow'):
            for l in self.preFlow:
                fl_fea, logdet = l(fl_fea, logdet, reverse=reverse)
        return fl_fea, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, rrdb_results, y_onehot=None):
        ft = None if layer.position is None else rrdb_results[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, y_onehot=y_onehot)

        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def decode(self, rrdb_results, z, eps_std=None, epses=None, logdet=0.0, y_onehot=None):
        z = epses.pop() if isinstance(epses, list) else z
        fl_fea = z
        # debug.imwrite("fl_fea", fl_fea)
        bypasses = {}
        level_conditionals = {}
        if not self.config.netG.flow.levelConditional.conditional:
            for level in range(self.L + 1):
                level_conditionals[level] = rrdb_results[self.levelToName[level]]

        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            size = shape[2]
            level = int(np.log(160 / size) / np.log(2))
            # size = fl_fea.shape[2]
            # level = int(np.log(160 / size) / np.log(2))

            if isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea, layer,
                                                              rrdb_results[self.levelToName[level]], logdet=logdet,
                                                              y_onehot=y_onehot)
            elif isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True, rrdb_results=level_conditionals[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        sr = fl_fea

        assert sr.shape[1] == 3
        return sr, logdet

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, rrdb_results, logdet, y_onehot=None):
        ft = None if layer.position is None else rrdb_results[layer.position]
        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, y_onehot=y_onehot)
        return fl_fea, logdet


def get_position_name(H, scale):
    downscale_factor = 160 // H
    position_name = 'fea_up{}'.format(scale / downscale_factor)
    return position_name


if __name__ == '__main__':
    from config import config as _config
    flowUpsamplerNet = FlowUpsamplerNet(_config.netG.flow.hidden_channels,
                                        _config.netG.flow.K,
                                        flow_coupling=_config.netG.flow.coupling.name,
                                        config=_config)
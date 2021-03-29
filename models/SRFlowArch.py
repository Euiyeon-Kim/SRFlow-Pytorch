import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBencoder import RRDBNet
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow


class SRFlowNet(nn.Module):
    def __init__(self, gc=32, scale=4, config=None):
        super(SRFlowNet, self).__init__()
        self.config = config

        self.RRDB = RRDBNet(gc, scale, config)
        self.RRDB_training = True
        self.set_rrdb_training(True)

        self.quant = self.config.netG.flow.augment.noise_quant

        self.flowUpsamplerNet = FlowUpsamplerNet(config.netG.flow.K,
                                                 flow_coupling=config.netG.flow.coupling.name,
                                                 config=config)
        self.i = 0

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None,
                reverse_with_grad=False, lr_enc=None, add_gt_noise=False, y_label=None):
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, y_onehot=y_label)
        else:
            assert lr.shape[1] == 3
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True):
        if lr_enc is None:
            lr_enc = self.preprocess_encoder_output(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)
        z = gt

        if add_gt_noise:
            z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdb_results=lr_enc, gt=z, logdet=logdet, reverse=False,
                                              epses=epses, y_onehot=y_onehot)
        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        objective = objective + flow.GaussianDiag.logp(None, None, z)

        nll = (-objective) / float(np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def preprocess_encoder_output(self, lr):
        rrdb_results = self.RRDB(lr, get_steps=True)
        block_idxs = self.config.netG.RRDBencoder.stackRRDB.blocks
        if len(block_idxs) > 0:
            concat = torch.cat([rrdb_results["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if self.config.netG.RRDBencoder.stackRRDB.concat:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdb_results.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdb_results.keys():
                    keys.append('fea_up-1')
                if self.config.scale >= 8:
                    keys.append('fea_up8')
                if self.config.scale == 16:
                    keys.append('fea_up16')
                for k in keys:
                    h = rrdb_results[k].shape[2]
                    w = rrdb_results[k].shape[3]
                    rrdb_results[k] = torch.cat([rrdb_results[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdb_results

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.config.scale ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.preprocess_encoder_output(lr)

        x, logdet = self.flowUpsamplerNet(rrdb_results=lr_enc, z=z, eps_std=eps_std,
                                          reverse=True, epses=epses, logdet=logdet)

        return x, logdet


if __name__ == '__main__':
    from torchsummary import summary
    from config import config as _config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srflow = SRFlowNet(config=_config).to(device)
    summary(srflow, (3, 160, 160))
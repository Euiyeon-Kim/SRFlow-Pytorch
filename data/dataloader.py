import os
from glob import glob

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data


def random_crop(hr, size_hr):
    h, w, _ = hr.shape
    start_h = np.random.randint(low=0, high=(h - size_hr) + 1) if h > size_hr else 0
    start_w = np.random.randint(low=0, high=(w - size_hr) + 1) if w > size_hr else 0
    hr_patch = hr[start_h:start_h+size_hr, start_w:start_w+size_hr, :]
    return hr_patch


def center_crop(hr, size_hr):
    h, w, _ = hr.shape
    start_h = (h - size_hr) // 2
    start_w = (w - size_hr) // 2
    hr_patch = hr[start_h:start_h + size_hr, start_w:start_w + size_hr, :]
    return hr_patch


class CelebADataset(data.Dataset):
    def __init__(self, config, data_root, mode):
        super(CelebADataset, self).__init__()
        self.config = config
        self.mode = mode

        self.crop_size = config.patch_size[0]
        self.scale = config.scale
        self.lr_size = self.crop_size // self.scale

        self.gt_paths = sorted(glob(f'{data_root}/*'))

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, item):
        gt = np.asarray(Image.open(self.gt_paths[item]).convert('RGB'))

        # Random Crop
        if self.mode == 'train':
            hr = random_crop(gt, self.crop_size)
            # Augmentation
            if self.config.dataset.augment.use_flip:
                hr = np.flip(hr, axis=np.random.choice([True, False])).copy() if np.random.choice([True, False]) else hr
            if self.config.dataset.augment.use_rot:
                hr = np.rot90(hr, np.random.choice([1, 2, 3])).copy() if np.random.choice([True, False]) else hr

        # Center Crop
        else:
            hr = center_crop(gt, self.crop_size)

        # Downscaling
        lr = np.asarray(Image.fromarray(hr).resize((self.lr_size, self.lr_size), Image.BICUBIC))

        lr = torch.Tensor(np.moveaxis(lr, -1, 0) / 255.0)
        hr = torch.Tensor(np.moveaxis(hr, -1, 0) / 255.0)

        return {'LQ': lr, 'GT': hr, 'PATH': self.gt_paths[item]}


def get_train_dataloader(config):
    dataset = CelebADataset(config, config.dataset.train_root_dir, 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.dataset.batch_size,
                                             shuffle=config.dataset.augment.use_shuffle,
                                             num_workers=config.dataset.num_workers, drop_last=True)
    return dataloader


def get_valid_dataloader(config):
    dataset = CelebADataset(config, config.dataset.valid_root_dir, 'valid')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=config.dataset.num_workers, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    from config import config as _config
    _config.dataset.train_root_dir = '../datasets/CelebA'
    dataset = CelebADataset(_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    for _, train_data in enumerate(dataloader):
        print(train_data['LQ'].shape, train_data['GT'].shape)
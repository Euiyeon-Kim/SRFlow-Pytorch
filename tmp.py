import os

import numpy as np
from PIL import Image

from config import config
from data.dataloader import get_train_dataloader


if __name__ == '__main__':
    os.makedirs('lr', exist_ok=True)
    os.makedirs('hr', exist_ok=True)

    dataloader = get_train_dataloader(config)
    for i, data in enumerate(dataloader):
        lr = np.moveaxis(data['LQ'].numpy()[0], 0, -1)
        hr = np.moveaxis(data['GT'].numpy()[0], 0, -1)
        Image.fromarray(lr).save(f'lr/{i}.png')
        Image.fromarray(hr).save(f'hr/{i}.png')

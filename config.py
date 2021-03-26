from dotmap import DotMap

config = DotMap({
    'scale': 8,
    'patch_size': (160, 160, 3),                    # H, W, C
    'path': {
        'netG_weight_path': 'SRFlow_CelebA_8X.pth'
    },

    'train': {
        'dist': False,
        'gpu_ids': None,
        'n_iter': 200000,
        'resume': True,
    },
    'netG': {
        'in_nc': 3,
        'out_nc': 3,
        'RRDBencoder': {
            'nf': 64,
            'nb': 8,
            'train_delay': 0.5,
            'RRDB_channels': 64,
            'stackRRDB': {
                'blocks': [1, 3, 5, 7],             # Index of RRDB encoder to concat features
                'concat': True,                     # Concat RRDB features for rich description of LR
            },
            'fea_up0': True,
            'fea_up_1': False,
        },
        'flow': {
            'K': 16,
            'L': 4,
            'coupling': {
                'name': 'CondAffineSeparatedAndCond',
                'hidden_channels': 64,
                'eps': 0.0001
            },
            'additionalFlowNoAffine': 2,
            'flow_permutation': 'invconv',
            'split': {
                'enable': True,
                'correct_splits': False,
                'logs_eps': 0,
                'consume_ratio': 0.5,
            },
            'augment': {
                'noise': True,
                'noise_quant': 32,
            }
        }
    },
    'val': {
        'heats': [0.0, 0.5, 0.75, 1.0],
        'n_sample': 3,
    }
})

if __name__ == '__main__':
    print(not config.netG.flow.levelConditional.conditional)
    print(config.netG.flow.split.type or 'Split2d')
    exit()


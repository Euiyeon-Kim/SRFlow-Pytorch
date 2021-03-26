from dotmap import DotMap

EXP_NAME = 'tmp'

config = DotMap({
    'is_train': True,
    'scale': 8,
    'patch_size': (160, 160, 3),                    # H, W, C
    'path': {
        'exp_path': f'exps/{EXP_NAME}',
        'netG_weight_path': 'resources/SRFlow_CelebA_8X.pth'
    },
    'dataset': {
        'batch_size': 16,
    },

    'train': {
        'dist': False,
        'gpu_ids': None,
        'n_iter': 200000,
        'resume': True,
        # Optimizer
        'weight_decay_G': 0,
        'lr_G': 5e-4,
        'lr_RRDB': 5e-4,
        'lr_scheme': 'MultiStepLR',
        'warmup_iter': -1,
        'beta1': 0.9,
        'beta2': 0.99,
        'lr_steps_rel': [0.5, 0.75, 0.9, 0.95],
        'lr_gamma': 0.5,
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

n_iter = config.train.n_iter
if config.train.T_period_rel:
    config.train.T_period = [int(x * n_iter) for x in config.train.T_period_rel]
if config.train.restarts_rel:
    config.train.restarts = [int(x * n_iter) for x in config.train.restarts_rel]
if config.train.lr_steps_rel:
    config.train.lr_steps = [int(x * n_iter) for x in config.train.lr_steps_rel]
if config.train.lr_steps_inverse_rel:
    config.train.lr_steps_inverse = [int(x * n_iter) for x in config.train.lr_steps_inverse_rel]

if __name__ == '__main__':
    print(config.train.weight_l1 or 0)
    exit()



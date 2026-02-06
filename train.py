import os
import sys

sys.path.append('..')

from config import get_config
from bridge.runners.ipf import IPFSequential


def main():
    args = get_config()

    print('=== DSB Training: HES -> CD30 ===')
    print(f'Image size : {args.data.image_size}')
    print(f'Batch size : {args.batch_size}')
    print(f'Num iter   : {args.num_iter}')
    print(f'Num IPF    : {args.n_ipf}')
    print(f'Num steps  : {args.num_steps}')
    print(f'Device     : {args.device}')
    print(f'Transfer   : {args.transfer}')
    print(f'Data dir   : {args.data_dir}')
    print('Directory  : ' + os.getcwd())

    ipf = IPFSequential(args)
    ipf.train()


if __name__ == '__main__':
    main()

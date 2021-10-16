"""Some routines for executing scripts."""

# pylint: disable = import-outside-toplevel, c-extension-no-member

import argparse
import os
import sys

from utils.cfg_parser import ConfigObject, dump_yaml, parse_yaml, copy_data_to_cache

VERSION = "0.1.0"

def parse_args():
    """Function for parsing command line args and merging them with yaml.j2 config."""

    parser = argparse.ArgumentParser(description='ISDS Mindspore research code.')

    parser.add_argument('--version', action='version', version=f'{VERSION}')
    parser.add_argument('--device', type=str, default='Ascend', choices=["CPU", "GPU", "Ascend"],
                        help='Computing device.')
    parser.add_argument('--profile', type=int, default=0, help='Profiling mode.')
    parser.add_argument('--export_file', type=str, default='',
                        help='Exporting mode. Path to save exported model')
    parser.add_argument('--config', type=str, default='', help='Configuration file')
    parser.add_argument('--seed', type=int, default=1, help="Random seed.")
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained model')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch for resumed training.')
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training.")
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help="Batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help="Batch size for eval.")
    parser.add_argument('--export_batch_size', type=int, nargs='+', default=32,
                        help="Batch size for exported models.")
    parser.add_argument('-d', '--dataset', type=str, default="imagenet", help="Dataset name.")
    parser.add_argument('--stat', type=int, default=0, help="Save training statistics.")
    parser.add_argument('--train_url', type=str, default='', help='train_url')

    args = parser.parse_args()

    pretrained = args.pretrained
    if pretrained.startswith('s3://'):
        pretrained_cache = pretrained.replace('s3://', '/cache/')
        copy_data_to_cache(pretrained, pretrained_cache)
        pretrained = pretrained_cache
    args.pretrained = pretrained


    if os.path.basename(sys.argv[0]) in ["export.py"]:
        device_num = 1
    else:
        device_num = int(os.getenv('RANK_SIZE'))
        print('........device_num={}'.format(device_num))

    if args.device == "GPU":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        os.environ["RANK_ID"] = str(rank)
        os.environ["DEVICE_ID"] = str(rank)

    data = {
        "DEVICE_NUM": device_num,
        "VERSION": VERSION,
        "NUM_EPOCHS": args.num_epochs,
        "START_EPOCH": args.start_epoch,
        "DEVICE": args.device,
        "TRAIN_BATCH_SIZE": args.batch_size,
        "EVAL_BATCH_SIZE": args.eval_batch_size,
        "DATASET": args.dataset,
        "STAT": args.stat
        }
    yaml = parse_yaml(args.config, data)
    # print('yaml:', yaml)
    if os.path.basename(sys.argv[0]) not in ["export.py"]:
        dump_yaml(yaml, "config.yaml")

    args = args.__dict__
    args.update(yaml)
    args = ConfigObject(args)

    return args

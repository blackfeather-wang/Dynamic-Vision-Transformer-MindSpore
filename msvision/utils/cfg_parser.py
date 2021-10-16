import yaml
import sys
import os
import re

from .cloud_copy_cache import copy_data_to_cache


class ConfigObject:
    def __init__(self, entries):
        for a, b in entries.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [ConfigObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, ConfigObject(b) if isinstance(b, dict) else b)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_yaml(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return y

def parse_replace_roma(fp, copy_to_cache=False, cloud_args={}):
    y = parse_yaml(fp)
    for y_key in y.keys():
        y_val = y[y_key]
        if type(y_val) is str and y_val.startswith('s3://'):
            # copy to /cache and replace to /cache
            y_val_cache = y_val.replace('s3://', '/cache/')
            if copy_to_cache:
                # ugly for rec with idx
                if y_val_cache.endswith('.rec'):
                    y_val_rec_idx = y_val.replace('.rec', '.idx')
                    y_val_cache_rec_idx = y_val_cache.replace('.rec', '.idx')
                    print('copy {} to {}'.format(y_val_rec_idx,y_val_cache_rec_idx))
                    copy_data_to_cache(y_val_rec_idx, y_val_cache_rec_idx)
                copy_data_to_cache(y_val, y_val_cache)
            y[y_key] = y_val_cache
    return y


def merge_args(args, args_yml_fn):
    if os.path.exists(args_yml_fn):
        args_dict = args.__dict__
        args_yml = parse_replace_roma(args_yml_fn, copy_to_cache=False)
        args_dict_merge = dict(args_dict, **args_yml)
        args = ConfigObject(args_dict_merge)
    elif len(args_yml_fn) != 0:
        print('yml file {} is not existed'.format(args_yml_fn))
        exit(0)

    sys_args = sys.argv[1:]
    for arg in sys_args:
        if re.match('^--(.*)=(.*)$', arg):
            arg = arg.replace('--', '')
            key, val = arg.split('=')
            if key not in ['local_rank', 'world_size', 'args_yml_fn']:
                default_value = getattr(args, key)
                print('set {} from {}:{} to {}'.format(key, type(default_value), default_value ,val))
                setattr(args, key, type(default_value)(val))
        else:
            print('unmatched, arg: {}'.format(arg))

    return args

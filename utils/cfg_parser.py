"""Some utils for work with configurations."""

import os

import yaml
from jinja2 import Environment, FileSystemLoader

from .cloud_copy_cache import copy_data_to_cache


class ConfigObject:
    """Class which represents configuration."""
    def __init__(self, entries):
        for key, val in entries.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [ConfigObject(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, ConfigObject(val) if isinstance(val, dict) else val)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

def parse_yaml(file_path, data):
    """Parse and render yaml file."""
    path, name = os.path.split(file_path)
    env = Environment(loader=FileSystemLoader(path))
    template = env.get_template(name)
    cont = template.render(data)
    parsed_yaml = yaml.safe_load(cont)
    return parsed_yaml

def dump_yaml(data, filename):
    """Dump rendered yaml file."""
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def parse_replace_roma(file_path, env, copy_to_cache=False):
    """Parse yaml on ModelArts."""
    yaml_cfg = parse_yaml(file_path, env)
    for y_key in yaml_cfg.keys():
        y_val = yaml_cfg[y_key]
        if isinstance(y_val, str) and y_val.startswith('s3://'):
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
            yaml_cfg[y_key] = y_val_cache
    return yaml_cfg

def merge_args(args_dict, args_yml_fn, specified_args, data):
    """Merge configs from different sources."""
    if os.path.exists(args_yml_fn):
        args_yml = parse_replace_roma(args_yml_fn, data, copy_to_cache=False)
        args_dict.update(args_yml)
        args_dict.update(specified_args)
        args = ConfigObject(args_dict)
    elif len(args_yml_fn) != 0:
        print('yml file {} is not existed'.format(args_yml_fn))
        exit(0)
    return args

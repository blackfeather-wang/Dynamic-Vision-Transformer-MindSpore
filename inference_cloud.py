"""train_imagenet."""
import os
import argparse
import random
import numpy as np
import moxing as mox
import time
import sys
import multiprocessing

exec_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_path = exec_path #os.path.join(exec_path, '..')
sys.path.insert(0, project_path)

from msvision.utils.cfg_parser import parse_replace_roma


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--data_url', type=str, default=None, help='data_url')
    parser.add_argument('--train_url', type=str, default='./', help='train_url')
    parser.add_argument('--args_yml_fn', type=str, default='configs/resnet50.yml', help='Faker args_yml_fn')
    parser.add_argument('--task_id', type=int, default=0, help='task_id')
    parser.add_argument('--config', type=str, default='', help='args_yml_fn')

    parser.add_argument('--setup_environment', type=int, default=1, help='if copy datasets and upgrade packages')
    parser.add_argument('--upgrade_packages', type=int, default=0, help='upgrade_packages')
    parser.add_argument('--packages_path', type=str, default='', help='upgrade_packages')
    parser.add_argument('--custom_name', type=str, default='', help='upgrade_packages')

    args, unknown_args = parser.parse_known_args()
    args.args_yml_fn = os.path.join(project_path, args.args_yml_fn)
    args.config = os.path.join(project_path, args.config)

    # device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    # local_rank = int(os.getenv('RANK_ID'))    # local_rank
    # world_size = int(os.getenv('RANK_SIZE'))  # world_size
    device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    if device_id != 0:
        exit(0)
    local_rank = 0    # local_rank
    world_size = 1  # world_size
    print('device_id={}, local_rank={}, world_size={}'.format(device_id, local_rank, world_size))

    # install torch
    os.system("sudo chown -R work:work  /home/work")
    os.system('python -m pip install -U pip==8.0.1; pip install -U pip')
    #os.system('cd {}; pip install -r requirements.txt'.format(project_path))
    os.system('cd {}; pip install torch-1.4.0+cpu-cp37-cp37m-linux_x86_64.whl'.format(project_path))
    os.system('cd {}; pip install torchvision-0.5.0+cpu-cp37-cp37m-linux_x86_64.whl'.format(project_path))
    os.system('pip install jinja2')
    os.system('pip install tqdm')
    os.system('cat /cache/user-job-dir/mindspore_isds/configs/cloud/vit_base32_adamw_originhead_test.yml.j2')

    # os.system('export DEVICE_NUM={}'.format(world_size))
    # device_num = os.getenv('DEVICE_NUM')
    # print('........device_num={}'.format(device_num))

    s3_rank_ready_file = os.path.join(args.train_url, 'rank_{}_task_{}.txt'.format(local_rank, args.task_id))
    if mox.file.exists(s3_rank_ready_file):
        mox.file.remove(s3_rank_ready_file, recursive=False)
        time.sleep(10)

    if args.setup_environment and device_id == 0:
        if args.upgrade_packages:
            cmd = 'cd {};python training_cloud/upgrade_packages.py'.format(project_path)
            cmd += ' --packages_path={}'.format(args.packages_path)
            cmd += ' --custom_name={}'.format(args.custom_name)
            os.system(cmd)
        parse_replace_roma(args.args_yml_fn, copy_to_cache=True)

    mox.file.write(s3_rank_ready_file, '{}'.format(local_rank))
    while True:
        all_rank_exist = True
        for rank_item in range(world_size):
            rank_fn_item = os.path.join(args.train_url, 'rank_{}_task_{}.txt'.format(rank_item, args.task_id))
            if not mox.file.exists(rank_fn_item):
                all_rank_exist = False
        if all_rank_exist:
            break
        else:
            time.sleep(5)

    core_per_proc = int(multiprocessing.cpu_count() / 8)
    cmdopt = 'taskset -c {}-{}'.format(core_per_proc * device_id, core_per_proc * (device_id + 1) - 1)
    #cmd = 'cd {};{} python train.py --args_yml_fn={}'.format(project_path, cmdopt, args.args_yml_fn)
    cmd = 'cd {};{} python train.py --config={}'.format(project_path, cmdopt, args.config)
    cmd += ' --train_url={}'.format(args.train_url)
    for it in unknown_args:
        it = it.replace('--', '')
        key, val = it.split('=')
        cmd += ' --{}={}'.format(key, val)
    print(cmd)
    os.system(cmd)

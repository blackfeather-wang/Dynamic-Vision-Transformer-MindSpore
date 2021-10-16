import os
import time
import moxing as mox
import glob
from pathlib import Path
import argparse


def cmd_exec(cmd, just_print=False):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n{}:INFO:{}".format(t, cmd))
    if not just_print:
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--packages_path', type=str, default='', help='upgrade_packages')
    parser.add_argument('--custom_name', type=str, default='', help='upgrade_packages')
    args, unknown_args = parser.parse_known_args()

    run_package_local_path = args.packages_path.replace('s3://', '/cache/')
    mox.file.copy_parallel(args.packages_path, run_package_local_path)

    cmd_exec('chmod +x {}'.format(os.path.join(run_package_local_path, '*.run')))
    cmd_exec("python --version")
    cmd_exec("ls -alt /usr/local/Ascend")

    cmd_exec("before upgrade, version:", just_print=True)
    cmd_exec('sudo cat /usr/local/Ascend/driver/version.info')
    cmd_exec('sudo cat /usr/local/Ascend/fwkacllib/version.info')
    cmd_exec('sudo cat /usr/local/Ascend/opp/version.info')

    # 1. fwkacllib and opp upgrade
    cmd_exec("sudo chmod -R 777 /usr/local/Ascend")
    cmd_exec("sudo rm -rf /usr/local/Ascend/fwkacllib")
    cmd_exec("cd {};sudo ./Ascend-fwkacllib*.run --full --quiet".format(run_package_local_path))
    cmd_exec("ls -alt /usr/local/Ascend/fwkacllib")

    cmd_exec("cd {};sudo ./Ascend-opp*.run --uninstall --quiet".format(run_package_local_path))
    cmd_exec("cd {};sudo ./Ascend-opp*.run --full --quiet".format(run_package_local_path))

    cmd_exec("after upgrade, version:", just_print=True)
    cmd_exec('sudo cat /usr/local/Ascend/fwkacllib/version.info')
    cmd_exec('sudo cat /usr/local/Ascend/opp/version.info')

    if os.path.exists('/usr/local/Ascend/nnae'):
        cmd_exec('remove nnae and re-link /usr/local/Ascend to /usr/local/Ascend/nnae/latest', just_print=True)
        cmd_exec('sudo rm -rf /usr/local/Ascend/nnae/*')
        cmd_exec('sudo ln -s /usr/local/Ascend /usr/local/Ascend/nnae/latest')

    # 2. custom
    if len(args.custom_name) > 0 and os.path.exists(os.path.join(run_package_local_path, args.custom_name)):
        src_dir = os.path.join(run_package_local_path, args.custom_name, '*')
        dst_dir = '/usr/local/Ascend/fwkacllib/data/tiling/ascend910/custom'
        cmd_exec('before replace custom:', just_print=True)
        cmd_exec('ls -alt {}'.format(dst_dir))
        cmd_exec('sudo cp {} {}'.format(src_dir, dst_dir))
        cmd_exec('after replace custom:', just_print=True)
        cmd_exec('ls -alt {}'.format(dst_dir))

    # 3. topi and te
    pip_source = '-i http://100.125.33.126:8888/repository/pypi/simple --trusted-host=100.125.33.126'
    # pip_source = '-i http://192.168.2.228:8888/repository/pypi/simple --trusted-host=192.168.2.228'

    cmd_exec('sudo chown -R work:HwHiAiUser /usr/local/ma/python3.7/lib/python3.7')
    if os.path.exists('/usr/local/Ascend/fwkacllib'):
        cmd_exec('pip uninstall /usr/local/Ascend/fwkacllib/lib64/te-*.whl -y')
        cmd_exec('pip install /usr/local/Ascend/fwkacllib/lib64/te-*.whl {}'.format(pip_source))
        cmd_exec('pip uninstall /usr/local/Ascend/fwkacllib/lib64/topi-*.whl -y')
        cmd_exec('pip install /usr/local/Ascend/fwkacllib/lib64/topi-*.whl {}'.format(pip_source))
    else:
        cmd_exec('skip reinstall topi and te', just_print=True)

    # 4. install NUMA
    NUMA_name = list(Path(run_package_local_path).glob('numactl*'))
    if len(NUMA_name) > 0:
        cmd_exec("cd {};sudo rpm -ivh numactl-libs-*.rpm".format(run_package_local_path))
        cmd_exec("cd {};sudo rpm -ivh numactl-devel-*.rpm".format(run_package_local_path))
        cmd_exec("cd {};sudo rpm -ivh numactl-*.rpm".format(run_package_local_path))
        cmd_exec("numactl --hardware")
    else:
        cmd_exec('cannot find NUMA rpm, skip.', just_print=True)

    # 5. mindspore
    cmd_exec('before reinstall mindspore, version:', just_print=True)
    cmd_exec('pip list | grep mindspore')

    cmd_exec('pip uninstall -y mindspore_ascend')
    mindspore_name = list(Path(run_package_local_path).glob('mindspore*'))

    if len(mindspore_name) > 0:
        mindspore_name = os.path.split(mindspore_name[0])[-1]
        cmd_exec('start install: {}'.format(mindspore_name), just_print=True)
        cmd_exec('cd {};pip install {} {}'.format(run_package_local_path, mindspore_name, pip_source))

        cmd_exec('after reinstall mindspore, version:', just_print=True)
        cmd_exec('pip list | grep mindspore')
    else:
        cmd_exec('cannot find new mindspore whl, skip.', just_print=True)

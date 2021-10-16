"""Train script."""

# pylint: disable = protected-access

import builtins
import os
import time

import numpy as np
from scipy.special import softmax
import random
import torch
from mindspore import Tensor, context
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.train.callback import SummaryCollector
from mindspore.train.model import ParallelMode, Model
from mindspore.train.serialization import load_checkpoint
from mindspore.nn import WithLossCell, TrainOneStepCell
import mindspore.common.dtype as mstype
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from networks.vit import ViT
from networks.vit_dvt import Vit_Dvt
from args import parse_args
from nn.callbacks import StateMonitor
from nn.losses.cross_entropy import CrossEntropySmoothMixup2, CrossEntropySmoothMixup
from nn.metrics import ClassifyCorrectCell2, DistAccuracy2, ClassifyCorrectCell, DistAccuracy
from utils import dump_net, dynamic_call
from utils.logging import get_logger
from utils.set_loglevel import set_loglevel
from dvt_inference import dynamic_evaluate

def __init_env(args):
    device_num = int(os.getenv('RANK_SIZE'))
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id = int(os.getenv('RANK_ID'))
    print('device_num:{}, device_id:{}, rank_id:{}'.format(device_num, device_id, rank_id))

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device, save_graphs=True)
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device, save_graphs=False)
    context.set_context(max_call_depth=2000)
    if args.device == "Ascend":
        set_loglevel('error')
        if device_num > 1:
            os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
        context.set_context(device_id=device_id)

    #assert context.get_auto_parallel_context("enable_parallel_optimizer")



    assert not context.get_auto_parallel_context("enable_parallel_optimizer")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    set_seed(args.seed)

    args.logger = get_logger(".", rank=rank_id, device_num=device_num)
    # builtins.print = args.logger.info

    if args.profile:
        profiler = dynamic_call(args.profiler)
        args.num_epochs = 1
        args.eval.offset = 3
    else:
        profiler = None

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()

    return args, profiler, device_num, rank_id

def __main():
    args = parse_args()


    args, profiler, device_num, rank_id = __init_env(args)
    args.pretrained = '/data/cgf/mindspore_t2t_vit_6_21_1/misc/weights/dvt_21_6_28_10_16_1/deit_dvt_12_49_196-300_625.ckpt' # deit_dvt_12_49_196_n_f_w_r_adamw_originhead_dataaug_mixup_inference

    print(args)

    network = dynamic_call(args.network)
    network = network.to_float(mstype.float16)
    # network.to_float(mstype.float16)
    dump_net(network, "layers.txt")

    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, network)
        args.logger.info(f'load model {args.pretrained} success')

    train_dataset = dynamic_call(args.train_dataset)
    eval_dataset = dynamic_call(args.eval_dataset)


    time0 = time.time()
    print("========================", flush=True)
    # train_step = 0
    train_less_token_outputs = []
    train_outputs = []
    train_targets = []
    current_step = 0
    old_progress = 0
    t_end = time.time()
    network.set_train(mode=False)
    for x in train_dataset:
        current_step += 1
        image = Tensor(x[0])
        target = Tensor(x[1]).asnumpy()
        less_token_output, output = network(image)

        less_token_output = less_token_output.asnumpy()
        output = output.asnumpy()

        less_token_output = softmax(less_token_output, axis=1)
        output = softmax(output, axis=1)

        train_less_token_outputs.append(less_token_output)
        train_outputs.append(output)
        train_targets.append(target)

        if current_step % 100 == 0 or current_step == 1:
            time_used = time.time() - t_end
            fps = args.eval_batch_size * (current_step - old_progress) / time_used
            args.logger.info('iter[{}/{}], {:.2f} imgs/sec'.format(
                current_step, args.val_len // args.train_batch_size, fps))
            t_end = time.time()
            old_progress = current_step

        if current_step * args.train_batch_size == args.val_len:
            break
    train_less_token_outputs = np.concatenate(train_less_token_outputs, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)
    train_pred = np.stack((train_less_token_outputs, train_outputs), axis=0)
    print('train_pred.shape:', train_pred.shape)
    train_targets = np.array(train_targets).flatten()
    print('train_targets.shape:', train_targets.shape)
    print("========================", flush=True)
    time1 = time.time()
    args.logger.info('train time used={:.2f}s'.format(time1 - time0))

    time0 = time.time()
    print("========================", flush=True)
    # test_step = 0
    test_less_token_outputs = []
    test_outputs = []
    test_targets = []
    current_step = 0
    old_progress = 0
    t_end = time.time()
    network.set_train(mode=False)
    for x in eval_dataset:
        current_step += 1
        image = Tensor(x[0])
        target = Tensor(x[1]).asnumpy()

        less_token_output, output = network(image)  # graph mode mindspore

        less_token_output = less_token_output.asnumpy()
        output = output.asnumpy()

        less_token_output = softmax(less_token_output, axis=1)
        output = softmax(output, axis=1)

        test_less_token_outputs.append(less_token_output)
        test_outputs.append(output)
        test_targets.append(target)

        if current_step % 100 == 0 or current_step == 1:
            time_used = time.time() - t_end
            fps = args.eval_batch_size * (current_step - old_progress) / time_used
            args.logger.info('iter[{}/{}], {:.2f} imgs/sec'.format(
                current_step, args.val_len // args.eval_batch_size, fps))
            t_end = time.time()
            old_progress = current_step

        if current_step * args.eval_batch_size == args.val_len:
            break
    test_less_token_outputs = np.concatenate(test_less_token_outputs, axis=0)
    test_outputs = np.concatenate(test_outputs, axis=0)
    test_pred = np.stack((test_less_token_outputs, test_outputs), axis=0)
    print('test_pred.shape:', test_pred.shape)
    test_targets = np.array(test_targets).flatten()
    print('test_targets.shape:', test_targets.shape)
    print("========================", flush=True)
    time1 = time.time()
    args.logger.info('test time used={:.2f}s'.format(time1 - time0))

    flops1 = 1.145
    flops2 = 4.608
    flops = [flops1, flops1 + flops2]
    dynamic_evaluate(train_pred, train_targets, test_pred, test_targets, flops)

    if args.profile:
        profiler.analyse()

if __name__ == '__main__':
    __main()

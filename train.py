"""Train script."""

# pylint: disable = protected-access

import builtins
import os
import time

import numpy as np
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

def __init_env(args):
    device_num = int(os.getenv('RANK_SIZE'))
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id = int(os.getenv('RANK_ID'))

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

    print(args)

    network = dynamic_call(args.network)
    # network.to_float(mstype.float16)
    dump_net(network, "layers.txt")

    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, network)
        args.logger.info(f'load model {args.pretrained} success')

    train_dataset = dynamic_call(args.train_dataset)
    eval_dataset = dynamic_call(args.eval_dataset)
    lr_schedule = dynamic_call(args.lr_schedule)

    optimizer = dynamic_call(args.optimizer, inject_args={
        "params": network.trainable_params(),
        "learning_rate": Tensor(lr_schedule)
        })


    model = dynamic_call(args.train_model, inject_args={
        "network": network,
        "optimizer": optimizer
        })

    if args.device == "Ascend":
        time0 = time.time()
        time1 = time.time()
        args.logger.info('compile time used={:.2f}s'.format(time1 - time0))

    state_monitor = StateMonitor(data_size=args.train_batches_num,
                                 tot_batch_size=args.global_batch_size,
                                 lrs=lr_schedule,
                                 model=model,
                                 eval_dataset=eval_dataset,
                                 eval_interval=args.eval.interval,
                                 eval_offset=args.eval.offset,
                                 logger=args.logger,
                                 device_num=device_num,
                                 rank=rank_id,
                                 device=args.device,
                                 train_url=args.train_url
                                 )
    callbacks = []
    if rank_id == 0:
        callbacks.append(dynamic_call(args.checkpoint_callback))
        if args.stat == 1:
            callbacks.append(SummaryCollector(".", collect_freq=10, collect_specified_data={
                'collect_metric': True,
                'collect_input_data': False,
                'collect_graph': True,
                'histogram_regular': "x",
                'collect_dataset_graph': True
            }))
    callbacks.append(state_monitor)

    # train and eval
    time0 = time.time()
    model.train(args.num_epochs, train_dataset, callbacks=callbacks, sink_size=args.sink_size,
                dataset_sink_mode=args.dataset_sink_mode)
    time1 = time.time()
    args.logger.info('training time used={:.2f}s'.format(time1 - time0))

    args.logger.info(f'last_metric[{state_monitor.best_acc}]')
    args.logger.info(f'mean_fps[{state_monitor.mean_fps}]')

    if args.profile:
        profiler.analyse()

if __name__ == '__main__':
    __main()

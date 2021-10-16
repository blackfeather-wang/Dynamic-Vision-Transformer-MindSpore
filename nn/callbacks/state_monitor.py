"""Training state logging callback."""

# pylint: disable = invalid-name, too-many-instance-attributes, too-many-arguments

import sys
import time
import os

import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from tqdm import tqdm


class StateMonitor(Callback):
    """Training state logging callback."""

    def __init__(self, data_size, tot_batch_size=None, lrs=None,
                 model=None, eval_dataset=None, eval_interval=None, eval_offset=None,
                 logger=None, device_num=1, rank=0, device="Ascend", terminate_training=True, train_url=''):
        super().__init__()
        self.data_size = data_size
        self.tot_batch_size = tot_batch_size
        self.lrs = lrs
        self.epoch_num = 0
        self.loss = 0
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_interval = eval_interval
        self.eval_offset = eval_offset
        self.best_acc = -1
        self.mean_fps = 0.0
        self.epoch_time = 0
        self.device_num = device_num
        self.rank = rank
        self.tqdm_bar = None
        self.device = device
        self.terminate_training = terminate_training
        self.ckpts = []
        self.train_url = train_url
        self.logger = logger
        if logger is None:
            self.print_fn = print
        else:
            self.print_fn = logger.info

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.loss = loss
        if (self.rank == 0) and (self.device == "GPU"):
            self.tqdm_bar.update(1)

    def epoch_begin(self, run_context):
        # print('epoch_begin----in----')
        if (self.rank == 0) and (self.device == "GPU"):
            self.tqdm_bar = tqdm(total = self.data_size, file=sys.stdout, unit=" step",
                            desc=f"epoch [{self.epoch_num}]", dynamic_ncols=True)
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        if (self.rank == 0) and (self.device == "GPU"):
            self.tqdm_bar.close()
        epoch_seconds = (time.time() - self.epoch_time)
        per_step_seconds = epoch_seconds / self.data_size

        print_str = "epoch[{}]".format(self.epoch_num)
        print_str += ', epoch time: {:.2f}s'.format(epoch_seconds)
        print_str += ', per step time: {:.4f}s'.format(per_step_seconds)
        print_str += ', loss={:.6f}'.format(self.loss)

        if self.lrs is not None:
            lr = self.lrs[(self.epoch_num + 1) * self.data_size - 1]
            print_str += ', lr={:.6f}'.format(lr)

        if self.tot_batch_size is not None:
            fps = self.tot_batch_size * self.data_size / (epoch_seconds * self.device_num)
            # ignore first epoch, because usually it is slow
            if self.epoch_num > 0:
                self.mean_fps = (self.mean_fps * (self.epoch_num - 1) + fps) / self.epoch_num
            print_str += ', fps_per_device={:.2f}'.format(fps)

        if (self.epoch_num + 1) % self.eval_interval == self.eval_offset:
            eval_start = time.time()
            output = self.model.eval(self.eval_dataset)
            eval_seconds = time.time() - eval_start

            # print_str += ', accuracy={:.6f}'.format(float(output["acc"]))
            print_str += ', accuracy={}'.format(output["acc"])
            print_str += ', eval_cost={:.2f}'.format(eval_seconds)

        self.print_fn(print_str)

        if (self.epoch_num + 1) % self.eval_interval == self.eval_offset:
            # if len(output["acc"]) > 1:
            if isinstance(output["acc"], tuple):
                output_acc0 = float(output["acc"][1])
            else:
                output_acc0 = float(output["acc"])
            if output_acc0 > self.best_acc:
                self.best_acc = output_acc0

            if self.terminate_training and \
                (self.best_acc / output_acc0 >= 10) or \
                ((self.epoch_num >= 4) and (output_acc0 <= 0.01)):
                self.print_fn("Model diverged. Training has been terminated.")
                run_context.request_stop()

        if self.terminate_training:
            if isinstance(self.loss, np.float32) and \
                (np.isnan(self.loss) or np.isinf(self.loss)):
                self.print_fn("Invalid loss. Training has been terminated.")
                run_context.request_stop()
        if self.rank == 0:
            try:
                ckpt_dir = '/cache/checkpoints'
                ckpts = os.listdir(ckpt_dir)
                # self.print_fn('ckpts:{}'.format(ckpts))
                for ckpt in ckpts:
                    abs_ckpt = os.path.join(ckpt_dir, ckpt)
                    if ckpt in self.ckpts:
                        pass
                    else:
                        # copy to roma
                        if self.train_url != '':
                            import moxing as mox
                            roma_weights_fp = os.path.join(self.train_url, ckpt)
                            roma_weights_dirname = os.path.dirname(roma_weights_fp)
                            if not mox.file.exists(roma_weights_dirname):
                                mox.file.make_dirs(roma_weights_dirname)
                            os.system("python -c 'import moxing as mox; mox.file.copy(\"{}\", \"{}\")' &".format(abs_ckpt, roma_weights_fp))
                            self.print_fn("save weight success, local_weights_fp:{}, roma_weights_fp:{}".format(abs_ckpt,roma_weights_fp))
                        self.ckpts.append(ckpt)
                if self.logger is not None:
                    self.logger.copy_log_to_s3(self.train_url)
            except:
                pass

        self.epoch_num += 1

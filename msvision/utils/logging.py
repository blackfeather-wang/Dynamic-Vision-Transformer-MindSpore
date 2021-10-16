import logging
import os
import numpy as np
import sys
from datetime import datetime

logger_name = 'mindspore-benchmark'


class LOGGER(logging.Logger):
    def __init__(self, logger_name, rank=0):
        super(LOGGER, self).__init__(logger_name)
        self.log_fn = None
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', "%Y-%m-%d %H:%M:%S")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.log_fn = log_fn
    
    def copy_log_to_s3(self, train_url):
        try:
            import moxing as mox
            roma_log_fp = os.path.join(train_url, self.log_fn)
            roma_log_dirname = os.path.dirname(roma_log_fp)
            if not mox.file.exists(roma_log_dirname):
                mox.file.make_dirs(roma_log_dirname)
            mox.file.copy(self.log_fn, roma_log_fp)
        except:
            pass

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        if isinstance(args, (list, tuple)):
            for value in args:
                self.info('--> {}'.format(value))
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                self.info('--> {}: {}'.format(key, args_dict[key]))
        self.info('')


def get_logger(path, rank=0):
    logger = LOGGER(logger_name, rank)
    logger.setup_logging_file(path, rank)
    return logger

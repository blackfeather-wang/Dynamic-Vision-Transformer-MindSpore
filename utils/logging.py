"""Some logging utilities."""

# pylint: disable = logging-fstring-interpolation

import logging
import os
import sys

try:
    import moxing as mox
    MODEL_ARTS = True
except ImportError:
    MODEL_ARTS = False

LOGGER_NAME = 'mindspore-benchmark'

class LOGGER(logging.Logger):
    """Logger nheritor with additional methods for S3 storage."""

    def __init__(self, logger_name, rank=0, device_num=8):
        super().__init__(logger_name)
        self.log_fn = None
        self.rank = rank
        if rank % device_num == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          "%Y-%m-%d %H:%M:%S")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        """Setup logging file."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        #log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        log_name = 'log__{}.log'.format(rank)
        log_fn = os.path.join(log_dir, log_name)
        file_handler = logging.FileHandler(log_fn)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
        self.log_fn = log_fn

    def copy_log_to_s3(self, train_url):
        """Copy log to S3 storage."""
        if MODEL_ARTS:
            roma_log_fp = os.path.join(train_url, self.log_fn)
            roma_log_dirname = os.path.dirname(roma_log_fp)
            if not mox.file.exists(roma_log_dirname):
                mox.file.make_dirs(roma_log_dirname)
            mox.file.copy(self.log_fn, roma_log_fp)

    def info(self, msg, *args, **kwargs):
        """Print info data."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

def get_logger(path, rank=0, device_num=8):
    """Create file logger."""
    logger = LOGGER(LOGGER_NAME, rank, device_num)
    if rank == 0:
        logger.setup_logging_file(path, rank)
    return logger

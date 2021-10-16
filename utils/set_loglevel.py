import os

def set_loglevel(level='info'):
    print('set log level to {}'.format(level))
    for device_idx in range(8):
        os.system('su root -c "adc --host 127.0.0.1:22118 --log \
                  \\"SetLogLevel(0)[{}]\\" --device {}"'.format(level, device_idx))

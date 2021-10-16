import os

def set_loglevel(level='info'):
    print('set device global log level to {}'.format(level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -g {}'.format(level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -g {} -d 4'.format(level))
    event_log_level = 'enable' if level == 'info' else 'disable'
    print('set device event log level to {}'.format(event_log_level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -e {}'.format(event_log_level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -e {} -d 4'.format(event_log_level))

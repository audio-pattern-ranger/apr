'''
APR Configuration
'''
import logging
import os
import yaml


# Default values
DEFAULT_CONFIGURATION = {
    'loglevel': 'INFO',
    'workspace': '_workspace',  # Distro: '/var/cache/apr',
    # 'models': MUST_CONFIGURE,
    # Monitor
    'record_age': '1500',
    'record_duration': '00:30:00',
    # 'record_mic': MUST_CONFIGURE,
    'record_cam': '/dev/video0',
    'record_cam_options': ['-video_size', '1920x1080', '-framerate', '5'],
    'record_cam_timestamp': [
        '-vf', 'drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf:text="%{localtime}":fontcolor=red@0.9:x=7:y=7:fontsize=72'],  # noqa:E501
    'record_compression': 'medium',
    'target_accuracy': 85,
    }

# Storage for loaded configuration
loaded_configuration = None


def load_configuration():
    '''
    Check for a config file in the current directory or at /etc/apr/config.conf
    '''
    global loaded_configuration
    if loaded_configuration is None:
        loaded_configuration = DEFAULT_CONFIGURATION

    # Find best configuration file
    if os.environ.get('APR_CONFIG'):
        config_path = os.environ['APR_CONFIG']
    elif os.path.exists('config.yml'):
        config_path = 'config.yml'
    elif os.path.exists('/etc/apr/config.yml'):
        config_path = '/etc/apr/config.yml'
    else:
        logging.info('No configuration found; using defaults.')
        return False

    # Load values from selected configuration
    with open(config_path, 'r') as fh:
        logging.debug('Loading configuration from %s', config_path)
        config_values = yaml.safe_load(fh)
        if config_values:
            loaded_configuration.update(config_values)

    # Put 'nomatch' at the beginning of all model lists
    loaded_configuration['models'] = \
        ['nomatch'] + loaded_configuration['models']
    if len(loaded_configuration['models']) != 2:
        # TODO: This requires:
        # - collate_fn(?) should skip when an unexpected directory is found
        # - net = M5(n_input=transformed.shape[0], n_output=len(label2idx))
        raise Exception('Only ONE model is currently supported.')

    # Configure logging
    log = logging.getLogger()
    log_level = getattr(logging, loaded_configuration.get('loglevel'), 'INFO')
    log.setLevel(log_level)
    log_format = logging.Formatter('%(levelname)s:%(message)s')
    for h in log.handlers:
        h.setFormatter(log_format)


def get(key, default=None):
    '''
    Return a configuration value.
    '''
    return loaded_configuration.get(key, default)

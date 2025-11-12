'''
Disturbance Tracker - Bootstrap and Options
'''
import argparse
import json
import logging


# NOTE: Copied from src/state/config.go
DTRACK_DEFAULTS = {
    'workspace': '_workspace',
    'inspect_models': [],
    'train_epochs': 200,
    'train_batch_size': 16,
    'train_learning_rate': 0.0001,
    'train_patience': 10,
}


def bootstrap():
    '''
    Read cli flags into environment and return configuration options
    '''
    opts = read_arguments()

    # Convert golang log level to python (-V, -v, )
    if opts.very_verbose:
        level = 'TRACE'
    elif opts.verbose:
        level = 'DEBUG'
    else:
        level = 'INFO'
    configure_logging(level)

    # Load configuration file and merge with defaults + flags
    with open(opts.config_path, 'r', encoding='utf-8') as fh:
        config = json.load(fh)

    return {**DTRACK_DEFAULTS, **config, **vars(opts)}


def configure_logging(log_level):
    '''
    Configure the root logger with a specific format and level.
    '''
    # Add a custom 'TRACE' log level
    logging.addLevelName(5, 'TRACE')
    logging.Logger.trace = lambda self, msg, *args, **kwargs: self._log(
            5, msg, args, **kwargs)
    logging.trace = lambda msg, *args, **kwargs: logging.log(
            5, msg, *args, **kwargs)

    log = logging.getLogger()
    log.setLevel(log_level)

    # Add a default StreamHandler if none are configured
    if not log.hasHandlers():
        handler = logging.StreamHandler()
        log.addHandler(handler)

    # Set a consistent format for all handlers
    log_format = logging.Formatter('%(levelname)5s: %(message)s')
    for h in log.handlers:
        h.setFormatter(log_format)


def read_arguments():
    '''
    Parse and return command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        usage='dtrack [-h] ai.<action> [other_options]',
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=30))

    parser.add_argument(
        '-c',
        dest='config_path',
        action='store',
        metavar='<config>',
        default='./config.json',
        help='Specify path of the configuration file')
    parser.add_argument(
        '-v',
        dest='verbose',
        action='store_true',
        help='Enable verbose logging.')
    parser.add_argument(
        '-V',
        dest='very_verbose',
        action='store_true',
        help='Enable trace-level logging (more than -v).')

    # Arguments for ai/inspect.py
    group_inspect = parser.add_argument_group('options for action [inspect]')
    group_inspect.add_argument(
        '-i',
        dest='inspect_path',
        metavar='<input>',
        help='Path to a .dat audio file or a directory of files')

    return parser.parse_args()

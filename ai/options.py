'''
Disturbance Tracker - Bootstrap and Options
'''
import argparse
import json
import logging


# NOTE: Partially replicates "Application_Configuration" from src/config.go
DTRACK_DEFAULTS = {
        "workspace": "_workspace",
        "inspect_models": [],
        "train_target": 0.95,
        "train_rate": 0.001,
        "train_momentum": 0.9,
        "train_dropout": 0.2,
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
    Configure log format and output log level
    '''
    # Log Level: Trace
    logging.addLevelName(5, "TRACE")
    logging.Logger.trace = log_trace_real
    logging.trace = log_trace

    log = logging.getLogger()
    log.setLevel(log_level)

    # If no handlers, add a default StreamHandler
    if not log.hasHandlers():
        handler = logging.StreamHandler()
        log.addHandler(handler)

    # Log Format
    log_format = logging.Formatter('%(levelname)5s: %(message)s')
    for h in log.handlers:
        h.setFormatter(log_format)


def log_trace(message, *args, **kwargs):
    '''
    Wrapper to support access via .trace()
    '''
    logging.log(5, message, *args, **kwargs)


def log_trace_real(self, message, *args, **kwargs):
    '''
    Log a message if level is set to TRACE
    '''
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)  # pylint: disable=W0212


def read_arguments():
    '''
    Returns all options read from argument parser
    '''
    parser = argparse.ArgumentParser(
            usage='dtrack [-h] ai.<action> [other_options]',
            formatter_class=lambda prog: argparse.HelpFormatter(
                prog, max_help_position=30))

    # NOTE: Replicates "Application Flags" from src/flags.go
    parser.add_argument(
        '-c',
        dest='config_path',
        action='store',
        metavar='<config>',
        default='./config.json',
        help='Specify path of the configuration file')
    parser.add_argument(
        '-k',
        dest='keep_temp',
        action='store_true',
        help='Keep temporary files.')
    parser.add_argument(
        '-v',
        dest='verbose',
        action='store_true',
        help='Enable verbose logging.')
    parser.add_argument(
        '-V',
        dest='very_verbose',
        action='store_true',
        help='Like -v, but more.')

    # Extra arguments for ai/inspect.py
    group_inspect = parser.add_argument_group('options for action [inspect]')
    group_inspect.add_argument(
        '-i',
        dest='inspect_path',
        metavar='<input>',
        help='Path to DTrack-Formatted MKV file (or directory of files)')

    return parser.parse_args()

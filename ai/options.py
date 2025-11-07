'''
Disturbance Tracker - Bootstrap and Options
'''
import argparse
import json
import logging

DTRACK_DEFAULTS = {
    "workspace": "_workspace",
    "inspect_models": [],
    # Old sklearn params (can be removed or kept for legacy)
    "train_target": 0.95,
    # New PyTorch model parameters
    "train_epochs": 100,
    "train_batch_size": 16,
    "train_learning_rate": 0.0001,
    "train_patience": 10, # For early stopping
}

def bootstrap():
    """
    Read CLI flags, configure logging, and return final configuration options.

    This function serves as the entry point for setting up the application's
    environment by merging defaults, a configuration file, and command-line
    arguments.

    Returns:
        dict: A dictionary containing all configuration options.
    """
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
    try:
        with open(opts.config_path, 'r', encoding='utf-8') as fh:
            config = json.load(fh)
    except FileNotFoundError:
        logging.warning("Configuration file not found at %s. Using defaults.", opts.config_path)
        config = {}

    return {**DTRACK_DEFAULTS, **config, **vars(opts)}


def configure_logging(log_level):
    """
    Configure the root logger with a specific format and level.

    Args:
        log_level (str): The desired logging level (e.g., 'INFO', 'DEBUG', 'TRACE').
    """
    # Add a custom 'TRACE' log level
    logging.addLevelName(5, "TRACE")
    logging.Logger.trace = lambda self, msg, *args, **kwargs: self._log(5, msg, args, **kwargs)
    logging.trace = lambda msg, *args, **kwargs: logging.log(5, msg, *args, **kwargs)

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
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
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
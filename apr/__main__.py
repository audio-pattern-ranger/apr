#!/usr/bin/env python3
'''
Primary entry point
'''
import logging
import os
import importlib

# APR
import apr.config
import apr.options


def main():
    '''
    Prepare and launch application.
    '''
    # Use config from command line
    config_path = apr.options.get('config_path')
    if config_path:
        os.environ['APR_CONFIG'] = str(config_path)
    # Load application configuration
    apr.config.load_configuration()

    # Determine appropriate action
    action_alias = {'train': 'model'}
    action = apr.options.get('action')
    selected_action = action_alias.get(action, action)
    if not selected_action:
        raise Exception('No action (-a) provided!')

    # Kick off main process
    router = importlib.import_module(f'apr.{selected_action}')
    if not router:
        raise Exception('Unable to load selected action!')
    try:
        router.entry_point()
    except KeyboardInterrupt:
        # Notify of possibly-incomplete job
        if selected_action == 'monitor':
            logging.warning('^C pressed; Final video may be corrupt')
        # Exit gracefully when requested
        pass


if __name__ == '__main__':
    main()

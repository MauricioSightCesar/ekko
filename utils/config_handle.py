import argparse

import yaml

from utils.experiment_io import get_run_dir

def load_config(config_name=None, default_file_name="config", run_id=None):
    parser = argparse.ArgumentParser(description='Execute train validation step')
    parser.add_argument('--config', required=False, help='YAML File containing the configs')
    parser.add_argument('--run_id', required=False, help='Run ID for the experiment')
    args = parser.parse_args()

    if run_id:
        path = get_run_dir(run_id) / f"{default_file_name}.yaml"

    elif 'run_id' in args and args.run_id is not None:
        run_id = args.run_id
        path = get_run_dir(run_id) / f"{default_file_name}.yaml"
    
    else:
        model_name = default_file_name if args.config is None else args.config
        config_name = model_name if config_name is None else config_name
        path = f"configs/{config_name}.yaml"

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary, joining keys with sep.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
import argparse
import yaml
from typing import Dict, Tuple, List
import logging
from pathlib import Path

from haystack.utils import SquadData
from box import Box

from src import BASE_CONFIG_DEFAULT_PATH
from configs import CONFIG_PATH

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_queries_and_ids(data: Dict) -> Tuple[List[str], List[int]]:
    """Get queries and ids from a Squad-like dict of samples."""
    if 'validation' in data:
        data = data['validation']
    if 'train' in data:
        data = data['train']

    data_df = SquadData.to_df(data['data'])
    data_df_unique = data_df[["title", "context", "question", 'id']].drop_duplicates()

    return (
        data_df_unique['question'].to_list(),
        data_df_unique['id'].to_list(), #astype(int)
    )


def load_config(args: argparse.Namespace) -> Dict:
    """Load config.

    Default settings in BASE_CONFIG_DEFAULT_PATH are overridden by
    the actual config.
    """
    with open(BASE_CONFIG_DEFAULT_PATH, 'r') as fp:
        base_config = yaml.load(fp, Loader=yaml.SafeLoader)

    with open(args.config_path, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    base_config.update(config)
    return base_config


def augment_config(args: argparse.Namespace, config: Box) -> Box:
    """Augment the run config with values extracted from args or the config itself."""
    # overwrite pipeline config if it was supplied as an argument
    _log.info(
        f"Loading pipeline config "
        f"{f'from `{args.pipeline_config_path}`' if args.pipeline_config_path else 'found in config'}"
    )
    if args.pipeline_config_path:
        pipeline_config_path = args.pipeline_config_path
        _log.info(f"Overwriting pipeline config with `{pipeline_config_path}`")
    else:
        _log.info(
            f"Using pipeline config from default config: `{config['pipeline_config']}`"
        )
        pipeline_config_path = CONFIG_PATH / config["pipeline_config"]

    config["pipeline_config_path"] = pipeline_config_path
    config["dataset_name"] = Path(config["data_path"]).parent

    return config

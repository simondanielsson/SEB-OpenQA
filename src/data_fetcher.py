from typing import Dict
import json
import logging
from pathlib import Path

from box import Box
from haystack.utils import fetch_archive_from_http
from data import DATA_PATH

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

# mapping from dataset name to dataset name expected by the loader.
# this is only needed when the expected name is different
#   from the standard name.
MAP_DATASET_NAME_TO_DOWNLOAD_PATH = {
    "nq_subset": "https://s3.eu-central-1.amazonaws.com/deepset"
                 ".ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
}


def load_file_or_download(config: Box) -> Dict:
    """Load file from disk if it exists, else possibly download and load."""
    try:
        with open(DATA_PATH / config.data_path) as fp:
            data = json.load(fp)
    except OSError:
        if config.download_if_not_found:
            downloaded_dataset_path = download_data(config)
            _log.info(f"Loading downloaded dataset")
            with open(downloaded_dataset_path) as fp:
                data = json.load(fp)
        else:
            raise ValueError(f'Could not find file {config.data_path} and download is not activated.')

    return data


def download_data(config: Box) -> Path:
    """Download data from the web and save to disk."""
    _log.info("Path does not exist: trying to fetch from web")
    output_dir = (DATA_PATH / config.data_path).parent

    dataset_identifier = str(Path(config.data_path).parent)
    download_path = MAP_DATASET_NAME_TO_DOWNLOAD_PATH.get(
        dataset_identifier, None,
    )
    if download_path is None:
        raise ValueError(
            f"Could not download {dataset_identifier}: "
            f"only datasets {list(MAP_DATASET_NAME_TO_DOWNLOAD_PATH.keys())}"
            f" support download."
        )

    fetch_archive_from_http(download_path, output_dir)
    output_file_path = output_dir / Path(download_path).stem
    _log.info(f"Successfully downloaded dataset to `{str(output_file_path)}`")

    return output_file_path


DATA_LOADERS = {
    'file': load_file_or_download,
}


def load_data(config):
    """Load data using a dataset name and source type.

    Can be loaded from either HuggingFace or tensorflow_datasets.
    :param config: Configuration for this run.
    :returns: a dataset
    """
    _log.info("Loading data...")

    data_loader = DATA_LOADERS.get(config.loader_type, None)

    dataset = data_loader(config)

    return dataset


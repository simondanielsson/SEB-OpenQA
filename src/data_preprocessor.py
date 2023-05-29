from transformers import SquadV2Processor
from haystack.nodes import PreProcessor
import logging

_log = logging.getLogger(__name__)


def get_preprocessor(config):
    """Fetch preprocessor for loaded data."""
    _log.info("Fetching data preprocessor...")
    # currently no manual runtime preprocessing has to be done for NQ
    return NoProcessor()


class NoProcessor:
    """Processor without any preprocessing steps."""

    def preprocess(self, dataset):
        return dataset

"""Module to generate predictions on a given dataset.

This module is to be used downstream in one of the
dataset-specific evaluation scripts, found in `evaluation/`.
"""
import argparse
import copy
import datetime
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Union
from collections import abc
import warnings

import yaml
from box import Box
from haystack import Pipeline, Document
from haystack.nodes import PreProcessor
from more_itertools import chunked
from yaml import SafeLoader
from tqdm import tqdm

sys.path.append(os.getcwd())  # add
sys.path.append(os.getcwd() + "/src")  # add

from data import DATA_PATH
from evaluation.output import OUTPUT_PATH
from src import CONFIG_DEFAULT_PATH
from src.data_fetcher import load_data
from src.data_postprocessor import get_postprocessor
from src.data_preprocessor import get_preprocessor
from src.utils import get_queries_and_ids, load_config, augment_config
from src.documentstore_manager import DSManagerBase, get_documentstore_manager
from src.final_evidence_fusion import FinalEvidenceFusionNode

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

warnings.filterwarnings("ignore")

def run_inference_pipeline() -> int:
    """Perform inference_hs with pipeline and data."""
    _log.info("--- Running inference pipeline ---")
    config = _get_config()
    data = load_data(config)
    preprocessed_data = _preprocess_data(data, config)

    pipeline, pipeline_config = _initialize_pipeline(config)
    result = _run_pipeline(pipeline, preprocessed_data, config)

    postprocessed_results = _postprocess_results(result, preprocessed_data, config)

    metadata = result["metadata"]
    _save_results(postprocessed_results, metadata, config, pipeline_config)

    return 0


def _get_config() -> Box:
    """Load experiment config."""
    args = _parse_args()
    _log.info(f"Loading config from `{args.config_path}`")

    config = load_config(args)
    config = augment_config(args, config)

    return Box(config)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on a ORQA pipeline",
    )
    # overwrite default config file
    parser.add_argument(
        "--config_path",
        nargs="?",
        default=CONFIG_DEFAULT_PATH,
        help=f"Config for this run, defaults to {CONFIG_DEFAULT_PATH}.",
    )
    parser.add_argument(
        "--pipeline_config_path",
        nargs="?",
        default=None,
        help=f"Pipeline config for this run, defaults to "
        f"what is found in {CONFIG_DEFAULT_PATH}.",
    )
    return parser.parse_args()


def _preprocess_data(data, config):
    """Preprocess evaluation data."""
    data_preprocessor = get_preprocessor(config)
    return data_preprocessor.preprocess(data)


def _initialize_pipeline(config: Box):
    """Initialize and get experiment pipeline."""
    _log.info(f"Loading pipeline from config `{config.pipeline_config_path}`")
    with open(config.pipeline_config_path, "r") as fp:
        pipeline_config = Box(yaml.load(fp, Loader=SafeLoader))
    
    documentstore_manager = get_documentstore_manager(config, pipeline_config)
    _log.info(f"Fetched document store manager `{documentstore_manager!r}`")
    documentstore_manager.prepare_documentstore()

    pipeline = Pipeline.load_from_yaml(
        config.pipeline_config_path, pipeline_name=config.pipeline_name,
    )

    _setup_document_store(
        config=config,
        pipeline_config=pipeline_config,
        pipeline=pipeline,
        documentstore_manager=documentstore_manager,
    )

    return pipeline, pipeline_config


def _setup_document_store(
    config: Box,
    pipeline_config: Box,
    pipeline: Pipeline,
    documentstore_manager: DSManagerBase,
):
    """Setup document store."""
    label_index = config.additional_pipeline_data.label_index
    doc_index = config.additional_pipeline_data.doc_index
    
    if not config.reset_documentstore:
        document_store = pipeline.get_document_store()
        if pipeline.get_node("Retriever") and not document_store.get_embedding_count(index=doc_index):
            retriever = pipeline.get_node("Retriever")
            _log.info(f"Updating dense embeddings using retriever {retriever!r} on {doc_index}")
            document_store.update_embeddings(retriever, index=doc_index)
        
        _log.info(
            f"Using existing document store:\n\t"
            f"{'n_documents':<10}:{document_store.get_document_count(index=doc_index):>6}\n\t"
            f"{'n_labels':<10}:{document_store.get_label_count(index=label_index):>6}\n\t"
            f"{'n_embeddings':<10}:{document_store.get_embedding_count(index=doc_index):>6}"
        )
        return

    _log.info("Resetting document store...")
    document_store = pipeline.get_document_store()

    _log.debug(f"Deleting documents from indices {doc_index}, {label_index}")
    document_store.delete_documents(index=doc_index)
    document_store.delete_documents(index=label_index)
    
    _log.debug("Loading Preprocessor")
    preprocessor = PreProcessor(**config.preprocessor_params) if config.do_preprocessing else None
    
    if config.write_train_documents:
        _write_documents(document_store, doc_index, preprocessor, config)
        documentstore_manager.save_documentstore(document_store)

    _log.info(
        f"Adding evaluation data to document store, \n\tfrom "
        f"{config.data_path}\n\tusing indices {doc_index}, "
        f"{label_index}\n\tand preprocessor {preprocessor!r}"
    )
    data_path = DATA_PATH / config.data_path
    _log.debug(f"Loading data from {data_path}")
    document_store.add_eval_data(
        filename=data_path,
        doc_index=doc_index,
        label_index=label_index,
        preprocessor=preprocessor,
        batch_size=config.populate_document_store_batch_size,
    )
    _log.info(
        f"Document store contents:\n\t"
        f"{'n_documents':<10}:{document_store.get_document_count(index=doc_index):>6}\n\t"
        f"{'n_labels':<10}:{document_store.get_label_count(index=label_index):>6}\n\t"
        f"{'n_embeddings':<10}:{document_store.get_embedding_count(index=doc_index):>6}"
    )
    
    if pipeline.get_node("Retriever") and not document_store.get_embedding_count(index=doc_index):
        retriever = pipeline.get_node("Retriever")
        _log.info(
            f"Updating embeddings using retriever "
            f"{retriever.__class__.__name__} on {doc_index} and {label_index}"
        )
        document_store.update_embeddings(retriever, index=doc_index)
        #document_store.update_embeddings(retriever, index=label_index)
        _log.info(
            f"Embeddings updated (n_embeddings):"
            f"\n\t{doc_index}: {document_store.get_embedding_count(index=doc_index)}"
        #    f"\n\t{label_index}: {document_store.get_embedding_count(index=label_index)}"
        )
    documentstore_manager.save_documentstore(document_store)


def _write_documents(
    document_store, 
    doc_index: str, 
    preprocessor: Optional[PreProcessor], 
    config: Box
) -> None:
    _log.info(f"Writing documents from `{config.train_documents_path}` to document store...")
    train_documents_path = DATA_PATH / config.train_documents_path
    with open(train_documents_path, 'r') as fp:
        suffix = train_documents_path.suffix
        if suffix == '.jsonl':
            data_generator = (json.loads(sample) for sample in fp)
        elif suffix == '.json':
            data = json.load(fp)
            data_generator = (sample for sample in data['data'])
        else:
            raise ValueError(f'Could not load train data file with suffix `{suffix}`.')

        samples_left = True
        update_samples_left = False
        while samples_left:
            text_batch = []
            for i in range(10000):
                try:
                    text_batch.append(next(data_generator))
                except StopIteration:
                    # final batch might be smaller
                    update_samples_left = True
                    break
            # write documents assumes Document objects
            document_batch = [
                Document(
                    sample['paragraphs'][0]['context'],  # long answers as contexts
                    content_type='text',
                )
                for sample in text_batch
            ]
            if preprocessor:
                _log.info("Preprocessing batch...")
                document_batch = preprocessor.process(document_batch)
            document_store.write_documents(document_batch, index=doc_index)
            if update_samples_left:
                samples_left = False

    _log.info(
        f"Finished writing {document_store.get_document_count(index=doc_index)} "
        f"to index `{doc_index}`."
    )

    
def _run_pipeline(pipeline, data, config):
    """Execute pipeline."""
    _log.info("Executing pipeline...")
    if config.do_eval:
        return _hs_run_evaluation(pipeline, config)
    return _hs_run_inference(pipeline, data, config)


def _hs_run_evaluation(pipeline: Pipeline, config: Box) -> Dict:
    """Run Haystack pipeline in evaluation mode."""
    _log.info("Evaluating pipeline...")
    eval_labels = pipeline.get_document_store().get_all_labels_aggregated(
        index=config.additional_pipeline_data.label_index,
        drop_negative_labels=config.drop_negative_labels,
        drop_no_answers=config.drop_no_answers,
    )
    _log.info(f"Fetched {len(eval_labels)} aggregated labels.")

    eval_config = _get_eval_config(config)

    start_time_eval = time.perf_counter()
    
    eval_func = pipeline.eval_batch if config.eval_batch else pipeline.eval
    _log.info(f"Running evaluation using {'eval_batch' if config.eval_batch else 'eval'} method.")
    try:
        eval_result = eval_func(
            labels=eval_labels,
            **eval_config,
        )
    except Exception as e:
        msg = 'Error in evaluation config'
        raise ValueError(msg) from e
        
    eval_runtime_raw = time.perf_counter() - start_time_eval
    eval_runtime = str(datetime.timedelta(seconds=eval_runtime_raw))
    _log.info(f"Evaluation performed in {eval_runtime}")
    return {
        "eval_result": eval_result,
        "metadata": {"eval_runtime": eval_runtime},
    }


def _get_eval_config(config: Box) -> Union[Dict, Box]:
    eval_config = copy.deepcopy(config.eval_config)

    for component_name in eval_config.params:
        if 'top_k' not in eval_config.params[component_name]:
            continue

        top_ks = eval_config.params[component_name]['top_k']

        if isinstance(top_ks, int):
            # do nothing
            continue
        elif isinstance(top_ks, abc.Sequence):
            eval_config.params[component_name]['top_k'] = max(top_ks)
        elif isinstance(top_ks, str):
            # allow for syntax x..y to mean range(x, y+1): set top_k=y
            eval_config.params[component_name]['top_k'] = int(top_ks.split("..")[1])
        else:
            raise ValueError(f'Could not parse {top_ks}')

    return eval_config



def _hs_run_inference(
    pipeline: Pipeline,
    data: Dict,
    config: Box,
) -> Dict:
    """Run Haystack pipeline in inference mode."""
    _log.info("Performing inference...")
    queries, ids = get_queries_and_ids(data)
    batches = chunked(queries, n=int(config.batch_size))

    _log.info(
        f"Performing {len(queries)} queries in batches of size {config.batch_size}..."
    )
    start_time_inference = time.perf_counter()

    inferences = [
        pipeline.run_batch(batch) 
        for batch in tqdm(
            batches, 
            total=len(queries) // int(config.batch_size),
        )
    ]

    inference_runtime_raw = time.perf_counter() - start_time_inference
    inference_runtime = str(datetime.timedelta(seconds=inference_runtime_raw))

    _log.info(f"Inference performed in {inference_runtime}")
    return {
        "inferences": inferences,
        "ids": ids,
        "metadata": {"inference_runtime": inference_runtime},
    }


def _postprocess_results(result, preprocessed_data, config):
    """Postprocess experiment results."""
    data_postprocessor = get_postprocessor(config)
    return data_postprocessor.postprocess(result, preprocessed_data)


def _save_results(inferences, metadata, config, pipeline_config) -> None:
    """Save postprocessed results to disk."""
    config = _clean_config(config)

    if not config.save_output:
        _log.info(f"Debug mode ({config.save_output=}): not saving outputs...")
        return

    current_time = datetime.datetime.strftime(
        datetime.datetime.now(), format="%Y%m%d-%H%M"
    )
    base_output_path = OUTPUT_PATH / config.dataset_name / current_time
    base_output_path.mkdir(parents=True, exist_ok=True)

    results_name = "evaluation" if config.do_eval else "inferences"

    _log.info(f"Saving results to {base_output_path}")
    with open(base_output_path / f"{results_name}.json", "w") as fp_inferences:
        json.dump(inferences, fp_inferences)
    with open(base_output_path / "config.yaml", "w") as fp_config:
        yaml.dump(config.to_dict(), fp_config, allow_unicode=True)
    with open(base_output_path / "pipeline_config.yaml", "w") as fp_pipeline_config:
        yaml.dump(pipeline_config.to_dict(), fp_pipeline_config, allow_unicode=True)
    with open(base_output_path / "metadata.json", "w") as fp_metadata:
        json.dump(metadata, fp_metadata)


def _clean_config(config: Box) -> Box:
    """Clean config to prepare for saving to disk."""
    # yaml.dump cannot correctly serialize Path objects
    config.pipeline_config_path = str(config.pipeline_config_path)
    config.dataset_name = str(config.dataset_name)
    return config


if __name__ == "__main__":
    sys.exit(run_inference_pipeline())

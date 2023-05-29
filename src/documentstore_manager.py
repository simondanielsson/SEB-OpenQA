"""Module for managing document stores.

Document stores might need preparation before they can be 
initialized. For instance, ElasticsearchDocumentStore must 
have a running Elasticsearch container to be instantiated. 
Document store managers provide the functionality for preparing
the launch of a document store, and the saving of them to disk. 
"""

from pathlib import Path
from abc import ABC
import logging
from typing import Type

from box import Box
from haystack.document_stores import BaseDocumentStore
from haystack.utils import launch_es

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


class DSManagerBase(ABC):
    "Base class for document store managers."

    def __init__(self, config, pipeline_config):
        raise NotImplementedError

    def prepare_documentstore(self) -> None:
        """Prepare document store for setup."""
        raise NotImplementedError

    def save_documentstore(self, document_store: BaseDocumentStore) -> None:
        """Teardown document store after setup."""
        raise NotImplementedError
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{name}={value}' for name, value in self.__dict__.items())})"


class FAISSDSManager(DSManagerBase):
    """Document store manager for FAISS."""

    def __init__(self, config, pipeline_config):
        self.reset_documentstore = config.reset_documentstore
        self.faiss_index_path = config.faiss_index_path
        self.faiss_config_path = config.faiss_config_path
        sql_urls = [
            component.params.sql_url 
            for component in pipeline_config.components 
            if component.name == 'DocumentStore' and 'sql_url' in component.params.keys()
        ]
        self.faiss_db_path = sql_urls[0] if sql_urls else None
        
        
        
    def prepare_documentstore(self) -> None:
        # SQL DB has to be empty to initialize a new FAISS DS
        if self.reset_documentstore:
            if not self.faiss_db_path:
                raise ValueError('A FAISS `sql_url` must be supplied if reseting document store.')
    
            prefix = 'sqlite:///'
            faiss_db_path = (
                self.faiss_db_path[len(prefix):]
                if self.faiss_db_path.startswith(prefix)
                else self.faiss_db_path
            )
            _log.info(f'Deleting old `{faiss_db_path}` and index {self.faiss_index_path} (if it exists)')
            Path(faiss_db_path).unlink(missing_ok=True)
            Path(self.faiss_index_path).unlink(missing_ok=True)

    def save_documentstore(self, document_store: BaseDocumentStore) -> None:
        if self.reset_documentstore:
            faiss_index_path = self.faiss_index_path or 'faiss_index.faiss'
            _log.info(f'Saving FAISS index to `{faiss_index_path}`')
            document_store.save(index_path=faiss_index_path, config_path=self.faiss_config_path)


class ElasticsearchDSManager(DSManagerBase):
    """Document store manager for Elasticsearch."""

    def __init__(self, config, pipeline_config):
        self.timeout = 30
        self.reset_documentstore = config.reset_documentstore

    def prepare_documentstore(self) -> None:
        # The elasticsearch container must be running before loading the pipeline
        if self.reset_documentstore:
            _log.info(
                f"Launching ElasticsearchDocumentStore container, waiting {self.timeout}s...",
            )
            launch_es(sleep=self.timeout)
            _log.info("Done launching ElasticsearchDocumentStore container.")

    def save_documentstore(self, document_store: BaseDocumentStore) -> None:
        # Elasticsearch saves automatically
        return


class InMemoryDSManager(DSManagerBase):

        def __init__(self, config, pipeline_config):
            pass

        def prepare_documentstore(self) -> None:
            pass

        def save_documentstore(self, document_store: BaseDocumentStore) -> None:
            pass



DOCUMENTSTORE_MANAGERS = {
    "ElasticsearchDocumentStore": ElasticsearchDSManager,
    "FAISSDocumentStore": FAISSDSManager,
    "InMemoryDocumentStore": InMemoryDSManager,
}


def get_documentstore_manager(config, pipeline_config) -> Type[DSManagerBase]:
    """Fetch the document store manager type for the experiment."""
    documentstore_type = _get_documentstore_type(pipeline_config, config)
    
    documentstore_manager_type = DOCUMENTSTORE_MANAGERS.get(documentstore_type, None)

    if documentstore_manager_type is None:
        raise ValueError(
            f'Could not find a valid document store manager '
            f'given document store type `{documentstore_type}`'
            f' in {DOCUMENTSTORE_MANAGERS}.'
        )

    return documentstore_manager_type(config, pipeline_config)


def _get_documentstore_type(pipeline_config: Box, config: Box) -> str:
    """Retriever document store type from pipeline config."""
    retriever_name = ""
    for pipeline in pipeline_config.pipelines:
        if pipeline.name == config.pipeline_name:
            for node in pipeline.nodes:
                if 'Retriever' in node.name:
                    retriever_name = node.name
                    break

    documentstore_name = ""
    for component in pipeline_config.components:
        if component.name == retriever_name:
            documentstore_name = component.params.document_store
            break

    documentstore_type = ""
    for component in pipeline_config.components:
        if component.name == documentstore_name:
            documentstore_type = component.type
            break

    _log.info(f"Pipeline uses document store of type {documentstore_type}")
            
    return documentstore_type
    
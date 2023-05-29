import logging
from collections import OrderedDict
from typing import Dict, Generator, Iterable, List, Tuple, Union
from itertools import product
from collections import abc

from box import Box
from haystack.schema import EvaluationResult

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_postprocessor(config: Box):
    """Fetch a data postprocessor"""
    _log.info("Getting postprocessor")
    if config.do_eval:
        _log.info("Loading Haystack evaluation postprocessor...")
        return HSEvaluationPostprocessor(config)

    _log.info("Loading Haystack inference postprocessor...")
    return HSInferencePostprocessor(config)


class HSPostprocessorBase:

    def postprocess(self, result: EvaluationResult, preprocessed_data) -> Dict:
        raise NotImplementedError()


class HSEvaluationPostprocessor(HSPostprocessorBase):

    def __init__(self, config: Box):
        self.eval_config = config.eval_config
        self.document_scope = config.document_scope
        self.answer_scope = config.answer_scope

    def postprocess(self, result: EvaluationResult, preprocessed_data) -> Dict:
        _log.info("Postprocessing evaluation results")
        eval_result = result['eval_result']

        answer_nodes = {node for node, df in eval_result.node_results.items() if len(df[df["type"] == "answer"]) > 0}
        all_top_1_metrics = eval_result.calculate_metrics(
            document_scope=self.document_scope, answer_scope=self.answer_scope, simulated_top_k_reader=1
        )
        answer_top_1_metrics = {node: metrics for node, metrics in all_top_1_metrics.items() if node in answer_nodes}
        
        end_to_end_metrics = {
            f"e2e_read_{reader_k}_retr_{retriever_k}": eval_result.calculate_metrics(
                document_scope=self.document_scope,
                answer_scope=self.answer_scope,
                simulated_top_k_reader=reader_k,
                #simulated_top_k_retriever=retriever_k
            )
            for reader_k, retriever_k in self._get_top_ks()
        }

        calculated_metrics = {
            "raw_eval": eval_result.calculate_metrics(
                document_scope=self.document_scope,
                answer_scope=self.answer_scope,
            ),
            **end_to_end_metrics,
            "top_1": answer_top_1_metrics,
            "upper bound max top_k": eval_result.calculate_metrics(
                document_scope=self.document_scope,
                answer_scope=self.answer_scope,
                eval_mode="isolated",
            ),
            "upper bound top_k=1": eval_result.calculate_metrics(
                document_scope=self.document_scope,
                answer_scope=self.answer_scope,
                eval_mode="isolated",
                simulated_top_k_reader=1,
            ),
        }

        return calculated_metrics

    def _get_top_ks(self) -> Generator[Tuple[int, int], None, None]:
        reader_ks = None
        retriever_ks = None
        for component_name in self.eval_config.params:
            if 'Retriever' in component_name:
                retriever_ks = type(self)._parse_top_ks(self.eval_config.params[component_name].top_k)
            elif 'Reader' in component_name:
                reader_ks = type(self)._parse_top_ks(self.eval_config.params[component_name].top_k)

        if not (reader_ks and retriever_ks):
            raise ValueError(f'Could not parse eval config {self.eval_config}')

        # if ranker is present, it is likely sitting between the retriever and reader.
        # in that case we cannot run with non-default simulated retriever top_k: instead set to default
        if 'Ranker' in self.eval_config.params:
            retriever_ks = (-1 for _ in retriever_ks)
        
        return product(reader_ks, retriever_ks)

    @staticmethod
    def _parse_top_ks(top_ks: Union[str, int, List[int]]) -> Iterable[int]:
        if isinstance(top_ks, int):
            return [top_ks]
        elif isinstance(top_ks, str):
            # allow for syntax '1..5' == range(1, 5+1)
            boundaries = top_ks.split('..')
            start, stop = int(boundaries[0]), int(boundaries[1]) + 1
            return range(start, stop)
        elif isinstance(top_ks, abc.Sequence):
            return top_ks

        raise ValueError(f'Could not parse top_ks {top_ks}')


class HSInferencePostprocessor(HSPostprocessorBase):

    def __init__(self, config: Box):
        super().__init__()

    def postprocess(self, result: Dict, preprocessed_data: List):
        _log.info(f"Postprocessing {len(result['inferences'])} predictions...")
        return OrderedDict(
            zip(
                result['ids'],
                self._get_answers(result['inferences']),
            )
        )

    def _get_answers(self, inferences):
        # reader returns top_k answer candidates:
        # fetch only the most likely answer (i.e. the first entry)
        return [
            answer_candidates[0].answer
            for batch_inferences in inferences
            for answer_candidates in batch_inferences['answers']
        ]


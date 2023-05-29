from typing import Tuple, Any, List, Optional, Dict, Union
from collections import abc
import copy
import logging

from haystack.nodes.base import BaseComponent
from haystack import Document, Answer

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


class FinalEvidenceFusionNode(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self, 
        reader_score_weight: float = 1.0, 
        retriever_score_weight: float = 1.0,
        top_k: Optional[int] = None,
        normalized_no_ans_boost: Optional[float] = None,
        debug: bool = False,
    ):
        """Initialize this node."""
        # Note:`Another look on DPR` performs grid search to find optimal score weights
        self.reader_score_weight = reader_score_weight
        self.retriever_score_weight = retriever_score_weight
        self.top_k = top_k
        self.normalized_no_ans_boost = normalized_no_ans_boost
        self.debug = debug

    def run(
        self,
        inputs: Optional[Union[List[dict], Dict]] = None,
        reader_score_weight: Optional[float] = None,
        retriever_score_weight: Optional[float] = None,
        top_k: Optional[int] = None,
        normalized_no_ans_boost: Optional[float] = None,
        _debug: Optional[dict] = None,
        node_id: str = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """Merge the scores from retriever and reader.
        
        The final score is a linear combination of the retriever and reader scores, with
        weights provided as parameters to this function. Defaults to the weights given at
        component initialization, or equal weight (1) to both components if no weights are
        provided.    
        """
        if reader_score_weight is None:
            reader_score_weight = self.reader_score_weight
        if retriever_score_weight is None:
            retriever_score_weight = self.retriever_score_weight
        if top_k is None:
            top_k = self.top_k
        if normalized_no_ans_boost is None:
            normalized_no_ans_boost = self.normalized_no_ans_boost or 0.0
        if _debug is None:
            _debug = self.debug
        
        if isinstance(inputs, abc.Sequence):
            for input_ in inputs:
                kwargs.update(input_)
        elif isinstance(inputs, abc.Mapping):
            kwargs.update(inputs)
        else:
            raise ValueError(f'cannot process input of type {type(inputs)}')
        
        answers = kwargs.get('answers')
        documents = kwargs.get('documents')
        
        final_evidence_answers = type(self)._create_final_evidence_score(
            answers=answers, 
            documents=documents, 
            reader_score_weight=reader_score_weight, 
            retriever_score_weight=retriever_score_weight,
            top_k=top_k,
            normalized_no_ans_boost=normalized_no_ans_boost,
        )

        output = {
            'query': kwargs.get('query'),
            'no_ans_gap': kwargs.get('no_ans_gap'),
            'answers': final_evidence_answers,
            'documents': documents,
        }
    
        return output, "output_1"
    
    @staticmethod
    def _create_final_evidence_score(
        answers: List[Answer], 
        documents: List[Document], 
        reader_score_weight: float, 
        retriever_score_weight: float,
        top_k: Optional[int],
        normalized_no_ans_boost: float,
    ) -> List[Answer]:
        """Merge the scores from reader and retriever and reorder answers."""
        final_evidence_answers = copy.deepcopy(answers)        
        
        for index, answer in enumerate(final_evidence_answers):
            # currently we only look at the first document in which the context occurs
            # TODO: check if we lose something by not including all documents, 
            #   perhaps we should add the scores from all documents
            
            if hasattr(answer, 'document_id'):
                document_id = answer.document_id
            elif hasattr(answer, 'document_ids'):
                document_ids = answer.document_ids
                if isinstance(document_ids, abc.Sequence):
                    #_log.debug(
                    #    f'Retrieved a list of document ids '
                    #    f'{document_ids}: fetching the first one.'
                    #)
                    document_id = document_ids[0]
                else:
                    document_id = document_ids
            else: 
                raise ValueError(f"Could not find document id in answer's attributes {set(answer.keys())}")
                
            for document in documents:
                if document.id == document_id:
                    retriever_score = document.score
                    break
            else: 
                #_log.debug(
                #    f'Could not find document {document_id} in retriever '
                #    f'output: probably empty answer. Score {answer.score:.2f} is '
                #    f'updated with normalized_no_ans_boost={normalized_no_ans_boost:.2f}'
                #)
                continue
            
            original_answer_score = answer.score # TODO: remove after debugging
            answer.score = (
                reader_score_weight * answer.score 
                + retriever_score_weight * retriever_score
            )
            _log.debug(f'{index} | {document_id} | read + ret:\n\t{reader_score_weight} * {original_answer_score:.2f} + {retriever_score_weight} * {retriever_score:.2f} = {answer.score:.2f}')
        
        sorted_ranked_answers = sorted(
            final_evidence_answers, 
            key=lambda answer: answer.score, 
            reverse=True
        )
        
        return sorted_ranked_answers[:top_k]
        

    def run_batch(
        self,
        inputs: Optional[List[dict]] = None,
        reader_score_weight: Optional[float] = None,
        retriever_score_weight: Optional[float] = None,
        node_id: str = None,
        _debug: bool = None,
        **kwargs,
    ) -> Tuple[Dict[str, List[Any]], str]:
        """Compute final evidence fusion on a batch of queries."""
        result = {'queries': [], 'answers': [], 'no_ans_gaps': []}     
        
        retriever_output, reader_output = inputs
        
        queries = reader_output['queries']      
        for index, query in enumerate(queries):
            if 'answers' in reader_output:
                answers = reader_output['answers'][index]
            else: 
                _log.error(f'No answer to be found in reader output: {reader_output}')
                continue
                
            result_from_query, _ = self.run(
                inputs={
                    'query': query,
                    'documents': retriever_output['documents'][index],
                    'answers': answers,
                    'no_ans_gap': reader_output['no_ans_gaps'][index],
                }, 
                reader_score_weight=reader_score_weight, 
                retriever_score_weight=retriever_score_weight,
                node_id=node_id,
                **kwargs,
            )
            result['queries'].append(result_from_query['query'])
            result['answers'].append(result_from_query['answers'])
            result['no_ans_gaps'].append(result_from_query['no_ans_gap'])
            result['documents'].append(result_from_query['documents'])
        
        return result, "output_1"
    
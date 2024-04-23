from typing import Dict, Callable, List, Union, Set
import pytrec_eval
import json

def eval(
        retrieval_results : Dict[str, List[str]], 
        expected_outputs : Dict[str, List[str]], 
        text_generator: Callable[[Dict[str, List[str]]], Dict[str, str]],
        downstream_metric: Callable[[Dict[str, str], Dict[str, List[str]]], Dict[str, Union[int, float]]],
        retrieval_metrics: Set[str],
        **args,
    ):
    """
    This function returns the eRAG score as explained in "[link to paper]"

    Args:
        retrieval_results ('Dict[str, List[str]]'):
            A dictionary that the key is the text input and the value is a list of 
            retrieval results in 'str' format that should be evaluated.
        
        expected_outputs ('Dict[str, List[str]]'):
            A dictionary that the key is the text input and the value is a list of the expected
            output that the 'text_generator' function should generate for that input.
                    
        text_generator ('Callable[[Dict[str, List[str]]], Dict[str, str]]'):
            A callable object that takes a dictionary of textual input to retrieval list and 
            generates dictionary of textual input to corresponding output.
                    
        downstream_metric ('Callable[[Dict[str, str], Dict[str, List[str]]], Dict[str, Union[int, float]]]'):
            A callable object that takes a dictonary of textual inputs to the corresponding prediction text as the first argument 
            and a dictionary of the textual inputs to corresponding list of gold outputs as the second argument,
            and generates a score based on them for the prediction text. The generated score should be in range [0, 1].

        retrieval_metrics ('Set[str]'):
            The set of Information Retrieval metrics should be used to evaluate the retrieval results. 
            We follow the same format as pytrec_eval library for deifning metrics: "https://github.com/cvangysel/pytrec_eval"
    
    Returns:
        A dictionary containing the per input eRAG score the and aggregated eRAG score.
    """

    assert set(retrieval_results.keys()) == set(expected_outputs.keys()), 'The keys in retrieval results and expected outputs do not match.'

    max_length_retrieval_lists = max(len(lst) for lst in retrieval_results.values())
    flatten_inputs = {
        f'{query}@{i}' : {'query' : query, 'document' : [doc]} for query, documents in retrieval_results.items() for i, doc in enumerate(documents)
    }

    evaluation_scores = dict()

    for i in range(max_length_retrieval_lists):
        current_input = dict()
        current_expected_outputs = dict()
        for query in retrieval_results.keys():
            if f'{query}@{i}' in flatten_inputs.keys():
                item = flatten_inputs[f'{query}@{i}']
                current_input[item['query']] = item['document']
                current_expected_outputs[item['query']] = expected_outputs[item['query']]
        current_generated_outputs = text_generator(current_input)
        assert set(current_input.keys()) == set(current_generated_outputs.keys()), 'The text_generator function did not return outputs for all given inputs.'
        current_evaluation_scores = downstream_metric(current_generated_outputs, current_expected_outputs)
        assert set(current_generated_outputs.keys()) == set(current_evaluation_scores.keys()), 'The downstream_metric function did not return evaluation scores for all given inputs.'
        for query, score in current_evaluation_scores.items():
            evaluation_scores[f'{query}@{i}'] = score
    
    qrel = dict()
    run = dict()

    binary_downstream_metric = True

    for query in retrieval_results.keys():
        run[query] = dict()
        qrel[query] = dict()
        for j in range(len(retrieval_results[query])):
            run[query][str(j)] = len(retrieval_results[query]) - j
            qrel[query][str(j)] = evaluation_scores[f'{query}@{j}']
            
            if qrel[query][str(j)] in [0, 1]:
                qrel[query][str(j)] = int(qrel[query][str(j)])
            if qrel[query][str(j)] not in [0, 1]:
                binary_downstream_metric = False
            if qrel[query][str(j)] > 1 or qrel[query][str(j)] < 0:
                raise RuntimeError('The returning value of the downstream_metric must be in range [0,1].')
            
    if binary_downstream_metric:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, retrieval_metrics)
        results = evaluator.evaluate(run)
    else:
        results = dict()
        for query, labels in qrel.items():
            results[query] = dict()
            for metric in retrieval_metrics:
                if "_" in metric:
                    metric_without_cut = metric[:metric.find("_")]
                    if metric_without_cut not in {'success', 'P'}:
                        raise RuntimeError('The provided retrieval metrics cannot be used with continuous downsream metric. The supported retrieval metrics are ["success", "P"]')
                    cut_value = int(metric[metric.find("_")+1:])
                else:
                    metric_without_cut = metric
                    cut_value = len(labels)
                if metric_without_cut == 'success':
                    max_value = 0
                    for i in range(cut_value):
                        max_value = max(max_value, labels[str(i)])
                    results[query][metric] = max_value
                elif metric_without_cut == 'P':
                    mean_value = 0
                    for i in range(cut_value):
                        mean_value += labels[str(i)]
                    results[query][metric] = mean_value / cut_value
                else:
                    raise RuntimeError('The provided retrieval metrics cannot be used with continuous downsream metric. The supported retrieval metrics are ["success", "P"]')
    final_results = {'per_input' : results, 'aggregated' : dict()}
    for metric in retrieval_metrics:
        values = [value[metric] for key, value in results.items()]
        final_results['aggregated'][metric] = sum(values) / len(values)
    return final_results



    
# eRAG: Evaluating Retrieval Quality in Retrieval-Augmented Generation

This repository contains the codes and packages for the paper titled [Evaluating Retrieval Quality in Retrieval-Augmented Generation]().

Evaluating retrieval-augmented generation (RAG) presents challenges, particularly for retrieval models within these systems. Traditional end-to-end evaluation methods are computationally expensive. Furthermore, evaluation of the retrieval model's performance based on query-document relevance labels shows a small correlation with the RAG system's downstream performance. We propose a novel evaluation approach, \metric, where each document in the retrieval list is individually utilized by the large language model within the RAG system. The output generated for each document is then evaluated based on the downstream task ground truth labels. In this manner, the downstream performance for each document serves as its relevance label. We employ various downstream task metrics to obtain document-level annotations and aggregate them using set-based or ranking metrics. Extensive experiments on a wide range of datasets demonstrate that \metric achieves a higher correlation with downstream RAG performance compared to baseline methods, with improvements in Kendall's $\tau$ correlation ranging from 0.168 to 0.494. Additionally, \metric offers significant computational advantages, improving runtime and consuming up to 50 times less GPU memory than end-to-end evaluation.


## Installation

You can install the codes for evaluating a RAG system with eRAG using the following script:

```
pip install erag
```

## Documentation

To calculate the eRAG score, you should use the eval function, with the following arguments:

- **--retrieval_results**: This is a dictionary in that the keys are the queries, and the values are the list of the text in the retrieved documents for each query. e.g.,

```
retrieval_results = {
    "What position does Harry play on the Gryffindor Quidditch team?" : [
        "Quidditch /ˈkwɪdɪtʃ/ is a fictional sport invented by author J. K. Rowling for her fantasy book series Harry Potter. It first appeared in the novel Harry Potter and the Philosopher's Stone (1997). In the series, Quidditch is portrayed as a dangerous but popular sport played by witches and wizards riding flying broomsticks.",
        "Matches are played on a large oval pitch with three ring-shaped goals of different heights on each side, between two opposing teams of seven players each: three Chasers, two Beaters, the Keeper, and the Seeker. The Chasers and the Keeper respectively score with and defend the goals against the Quaffle; the two Beaters bat the Bludgers away from their teammates and towards their opponents; and the Seeker locates and catches the Golden Snitch, whose capture simultaneously wins the Seeker's team 150 points and ends the game. The team with the most points at the end wins.",
        "Harry Potter plays as Seeker for his house team at Hogwarts. Regional and international Quidditch competitions are mentioned throughout the series. Aspects of the sport's history are revealed in Quidditch Through the Ages, published by Rowling in 2001 to benefit Comic Relief."
    ],
    "Who is the Headmaster of Hogwarts when Harry arrives?" : [
        "Prof. Albus Percival Wulfric Brian Dumbledore is a fictional character in J. K. Rowling's Harry Potter series. For most of the series, he is the headmaster of the wizarding school Hogwarts. As part of his backstory, it is revealed that he is the founder and leader of the Order of the Phoenix, an organisation dedicated to fighting Lord Voldemort, the main antagonist of the series.",
        "Dumbledore was portrayed by Richard Harris in the film adaptations of Harry Potter and the Philosopher's Stone (2001) and Harry Potter and the Chamber of Secrets (2002). Following Harris' death in October 2002, Michael Gambon portrayed Dumbledore in the six remaining Harry Potter films from 2004 to 2011. Jude Law portrayed Dumbledore as a middle-aged man in the prequel films Fantastic Beasts: The Crimes of Grindelwald (2018) and Fantastic Beasts: The Secrets of Dumbledore (2022)."
    ]
}
```

- **--expected_outputs**: This is a dictionary that the keys are the queries, and the values are the list of corrosponding outputs for each query. Note that the keys in *expected_outputs* and *retrieval_results* must be the same. e.g.,

```
expected_outputs = {
    "What position does Harry play on the Gryffindor Quidditch team?" : ["seeker"],
    "Who is the Headmaster of Hogwarts when Harry arrives?" : ["Albus Dumbledore", "Albus Percival Wulfric Brian Dumbledore", "Dumbledore"]
}
```


- **--text_generator**: This is a function that takes a dictionary as the input where the keys are the queries and values are the retrieved documents, and it returns a dictionary where the keys are the queries and the values are the generated string by the generative model in the RAG pipeline for the corresponding query. e.g.,

```
def text_generator(queries_and_documents):
    from openai import OpenAI
    client = OpenAI(api_key="...")
    results = dict()
    for question, documents in queries_and_documents.items():
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"question: {question} context: {' '.join(documents)} answer: ",
                }
            ],
            model="gpt-4-1106-preview",
        )
        results[question] = chat_completion.choices[0].message.content
    return results
```

In this example, we utilize *GPT4* to generate outputs for each input question using the retrieved documents. 

**Note:** For the case that there exists a function that given the query and a set of retrieved documents can generate the output for a single input, you can use the utility function provided to make it work on a batch of inputs:

```
from erag.utils import batchify_text_generator

def text_genrator_single_input(query, documents):
    ...
    return output

batch_text_genrator = batchify_text_generator(text_genrator_single_input)
```

Now, you can use *batch_text_genrator* for eRAG.

- **--downstream_metric**: This is a function that takes two dictionaries as the input; the first argument is a dictionary that its keys are the input queries and values are the generated text by the *text_generator* function, and the second argument is *expected_outputs* explained before. The main job of this function is to evaluate the usefulness of each generated text by comparing it to the expected output. Note that the values should be between 0 and 1. e.g.,

```
from rouge_score import rouge_scorer

def rouge_metric(generated_outputs, expected_outputs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)    
    results = dict()
    for query, gen_output in generated_outputs.items():
        expe_outputs_query = expected_outputs[query]
        max_value = 0
        for exp_output in expe_outputs_query:
            max_value = max(scorer.score(exp_output, gen_output)['rougeL'].fmeasure, max_value)
        results[query] = max_value
    return results
```

**Note:** For the case that there exists a function that given the generated ouput and a list of expected outputs can generate a score for a single input, you can use the utility function provided to make it work on a batch of inputs:

```
from erag.utils import batchify_downstream_metric

def score_single_input(generated_output, expected_outputs):
    ...
    return output

batch_scorer = batchify_downstream_metric(score_single_input)
```

Now, you can use *batch_scorer* for eRAG.

- **--retrieval_metrics**: This is a set of retrieval metric names that eRAG uses to aggregate the scores of individual documents in a ranked list for each query. We follow the [*pytrec_eval*](https://github.com/cvangysel/pytrec_eval) format for naming metrics. Note that when the returned values by *downstream_metric* are not binary (zero and one), the only possible metrics are precision ('P') and Hit Ratio ('success'). e.g., 

```
retrieval_metrics = {'P_10', 'success', 'recall', 'map'}
```


Given all these arguments, you can use the following code to evaluate your results:

```
import erag

results = erag.eval(
    retrieval_results = retrieval_results,
    expected_outputs = expected_output,
    text_generator = text_generator,
    downstream_metric = downstream_metric,
    retrieval_metrics = retrieval_metrics
)

```


## Examples

This [Colab notebook](https://colab.research.google.com/drive/1kMPRGowsVse56iGOei2Xaolk_zFw01S_?usp=sharing) is designed to show some examples of how to use eRAG for evaluating retrieval results in a RAG pipeline.

## Reference

```
@misc{salemi2024evaluating,
      title={Evaluating Retrieval Quality in Retrieval-Augmented Generation}, 
      author={Alireza Salemi and Hamed Zamani},
      year={2024},
      eprint={2404.13781},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgments

This work was supported in part by the Center for Intelligent Information Retrieval, in part by Lowe’s, and in part by an Amazon Research Award, Fall 2022 CFP. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.
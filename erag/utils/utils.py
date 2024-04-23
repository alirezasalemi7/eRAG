
def batchify_text_generator(func):
    def batchified_text_gen(current_input):
        answers = dict()
        for key, value in current_input.items():
            answers[key] = func(key, value)
        return answers
    return batchified_text_gen

def batchify_downstream_metric(func):
    def batchified_metric(generated, expected):
        answers = dict()
        for key, value in generated.items():
            answers[key] = func(value, expected[key])
        return answers
    return batchified_metric


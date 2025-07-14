import random

def build_support_set(x_embedding, human_data, self_data, lambda_weight, k):
    kh = int(lambda_weight * k + 0.5)
    ks = k - kh
    human_subset = human_data[:kh]
    self_subset = self_data[:ks]
    return human_subset + self_subset

def construct_prompt(support_set):
    prompt = ""
    for example in support_set:
        prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
    return prompt

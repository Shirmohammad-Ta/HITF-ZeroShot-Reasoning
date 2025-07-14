from hitf.model_selector import Selector
from hitf.prompt_constructor import build_support_set, construct_prompt
from hitf.marginalizer import marginalize_predictions

def run_HITF(model, x, encoder, selector, human_data, self_data, k=8):
    x_emb = encoder(x)
    lambda_weight = selector(x_emb).item()
    support_set = build_support_set(x_emb, human_data, self_data, lambda_weight, k)
    prompt_variants = [construct_prompt(support_set)]  # Can extend to multiple variants
    prediction = marginalize_predictions(model, x, prompt_variants)
    return prediction

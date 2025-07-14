import torch

def marginalize_predictions(model, x, prompt_variants):
    preds = []
    scores = []

    for prompt in prompt_variants:
        input_text = f"{prompt}Q: {x}\nA:"
        output = model.generate_answer(input_text)
        confidence = model.get_confidence(input_text, output)
        preds.append(output)
        scores.append(confidence)
    
    # Aggregate using weighted majority
    final_scores = {}
    for y, score in zip(preds, scores):
        final_scores[y] = final_scores.get(y, 0) + score
    
    return max(final_scores, key=final_scores.get)

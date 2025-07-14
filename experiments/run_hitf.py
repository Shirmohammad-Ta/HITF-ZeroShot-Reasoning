import json
from transformers import AutoTokenizer, AutoModel
from hitf.runner import run_HITF
from hitf.model_selector import Selector

class DummyEncoder:
    def __call__(self, text):
        return torch.rand(768)  # random embedding

class DummyModel:
    def generate_answer(self, prompt):
        return "42"  # Dummy fixed answer
    def get_confidence(self, prompt, answer):
        return 0.9  # Dummy fixed confidence

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    human_data = load_data("data/human_examples.json")
    self_data = load_data("data/self_generated.json")

    encoder = DummyEncoder()
    selector = Selector()
    model = DummyModel()

    x = "What is the answer to life, the universe, and everything?"
    prediction = run_HITF(model, x, encoder, selector, human_data, self_data, k=4)

    print("Prediction:", prediction)

if __name__ == "__main__":
    import torch
    main()

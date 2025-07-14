# HITF-ZeroShot-Reasoning

**Hybrid Instruction Tuning Framework (HITF)** is a lightweight yet powerful method to enhance the zero-shot reasoning capability of large language models (LLMs).  
It dynamically combines human-annotated and self-generated examples and leverages marginalization over intermediate responses to improve accuracy, logical coherence, and generalization.

## 🔬 Paper Reference
**Title:** Enhancing Zero-Shot Reasoning in Language Models via Hybrid Instruction Marginalization  
**Author:** Shirmohammad Tavangari  


## 🧠 Key Features
- Dynamic data selection (human vs self-generated)
- Intermediate response marginalization
- Zero-shot and few-shot support
- Compatible with GPT, T5, LLaMA

## 🚀 Quickstart
```bash
git clone https://github.com/your-username/HITF-ZeroShot-Reasoning.git
cd HITF-ZeroShot-Reasoning
pip install -r requirements.txt
```

## 🗂️ Folder Structure
```
hitf/
  ├── model_selector.py
  ├── prompt_constructor.py
  ├── marginalizer.py
  └── runner.py
data/
  ├── human_examples.json
  └── self_generated.json
```

## 🧪 Datasets
- SuperGLUE
- MMLU
- GSM8K
- FermiQA

## 📜 License
[MIT License](./LICENSE)

---

Made by [Shirmohammad Tavangari](mailto:s.tavangari@alumni.ubc.ca)

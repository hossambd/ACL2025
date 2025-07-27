# Implicit Hate Target Span Identification in Zero- and Few-Shot Settings with Selective Sub-Billion Parameter Models 

*Hossam Boudraa, Benoit Favre, Raquel Urena*

---

## 🔍 Overview

This repository provides the code, prompts, and benchmark results for identifying **implicit hate speech target spans** using **parameter-efficient masked and generative SLMs (<1B)** in zero-shot, few-shot, and supervised settings.

Implicit hate speech poses a unique challenge as it relies on **contextual, euphemistic, and ideologically coded expressions** rather than explicit slurs. Our work evaluates span-level models across multiple datasets to address this problem with interpretable, efficient, and instruction-tuned architectures.

---

## 🧪 Key Contributions

- A benchmark for **Implicit Target Span Identification (iTSI)** using **SBIC**, **IHC**, and **OffensiveLang** datasets.  
- Performance comparison of multiple sub-1B models in **zero-shot**, **few-shot**, and **fully supervised** conditions.  
- Introduction of **instruction tuning and LoRA fine-tuning** to enhance detection sensitivity.  
- In-depth **error analysis via LDA topic modeling** to reveal systematic model blind spots.  
- Demonstration that **scale isn’t everything**: small, well-instructed models can outperform larger LLMs.

---

## 📊 Results Summary

| Model                  | Params | IHC F1 | SBIC F1 |
|------------------------|--------|--------|---------|
| RoBERTa-Large          | 355M   | 72.5   | 75.8    |
| ModernBERT             | 125M   | 72.2   | 75.1    |
| LLaMA 3.2 (Instruct)   | 1B     | 68.5   | 72.5    |
| SmolLM2-135M (Instruct)| 135M   | 66.0   | 69.8    |

- **Few-shot F1 (10 examples):** 68.2 (SBIC), 64.0 (IHC)  
- **LoRA Rank 16 fine-tuning:** nearly matches full-data fine-tuning with 75% less compute time.

---

## 📁 Datasets

| Dataset         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **SBIC**        | Social Bias Inference Corpus: 34k implicit bias examples (span-annotated via GPT). |
| **IHC**         | Implicit Hate Corpus: 6,300 tweets with implicit hate.                      |
| **OffensiveLang** | Synthetic ChatGPT-based utterances with human-verified span annotations.   |

Annotation strategy: Weak supervision + automatic annotation (GPT-3.5) for span labels.

---

## 💡 Research Questions

- 🔬 Does increasing model size improve implicit hate span detection?
- ⚙️ Does instruction-tuning help?
- 🧪 Is few-shot fine-tuning competitive?
- 🧵 What can topic-based error analysis reveal?

---
<!-- 
## 🛠️ Setup & Training

### Requirements
```bash
pip install torch transformers scikit-learn datasets
```

### Fine-tuning with LoRA (example)
```bash
python train.py \
  --model_name SmolLM2-135M-Instruct \
  --dataset SBIC \
  --tuning lora \
  --lora_r 16 \
  --epochs 40 \
  --lr 1e-2 \
  --batch_size 16
```

### Inference (Zero-Shot with Prompting)
```bash
python zero_shot_infer.py \
  --model llama-3.2-1B-Instruct \
  --prompt_template templates/itsi_prompt.txt \
  --dataset IHC
```

## 🧵 Error Analysis

- LDA-based topic modeling reveals common failure themes:  
  *gender discourse*, *political ideology*, *identity framing*.  
- Many false negatives involve **sarcasm**, **euphemisms**, or **ideologically subtle cues**.
 --

## 🧑‍💻 Contact

`hossam.boudraa@etu.univ-amu.fr`  – [@hossambd](https://github.com/hossambd)  

--->

## ⚠️ Coming Soon

> The code scripts and prompt templates will be added soon.

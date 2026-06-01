# Molweni — MRC on Multiparty Dialogues

## Paper

**Molweni: A Challenge Multiparty Dialogues-based Machine Reading Comprehension Dataset with Discourse Structure**
Li et al., COLING 2020
[Paper](https://aclanthology.org/2020.coling-main.238)

## What it does

Molweni is a Machine Reading Comprehension dataset built on top of Ubuntu IRC multi-party dialogues. Given a dialogue context and a question, the model must extract the answer span (or predict "unanswerable"). It includes 10K+ dialogues annotated with discourse structure (RST-style relations).

## My implementation

- Fine-tuning `bert-base-uncased` on the Molweni MRC task
- Preprocessing with offset mapping, sliding window (stride + overflow chunks), unanswerable → `[CLS]` position
- Standard SQuAD-style start/end logits training

**Differences vs paper:**
- No discourse structure / graph-based component (the paper's full model uses RST-style relations)
- Baseline fine-tuning only; the paper reports stronger results with explicit discourse modeling

## Results

| Metric | Paper (BERT-base) | Mine |
|---|---|---|
| EM | 45.3 | TBD |
| F1 | 58.0 | TBD |

## How to run

```bash
pip install -r requirements.txt

# Download Molweni dataset from https://github.com/HIT-SCIR/Molweni
# Place train.json / dev.json / test.json in molweni/data/

jupyter notebook molweni.ipynb
```

# TopicBERT — Response Selection in Multi-Party Conversations

## Paper

**Topic-Aware Multi-turn Response Selection in Multi-Party Conversations**
Wang et al., EMNLP 2020
DSTC-8 Track 2 (NOESIS-II) — Tasks 2 & 4

## What it does

TopicBERT extends BERT with a topic attention mechanism to dynamically track conversational topics across multi-party IRC dialogues. It jointly trains on response selection (Task 2) and conversation disentanglement (Task 4) using multi-task learning.

## My implementation

- `TopicAttention` module implementing the additive attention equations from the paper
- Pretraining with STP (Sequential Turn Prediction) + MLM on DSTC-8 Ubuntu IRC logs
- `DisentangleHead` with feature vector `[t_r ; t_k ; t_r ⊙ t_k ; t_r − t_k]` for Task 4
- Multi-task fine-tuning: response selection + disentanglement

**Differences vs paper:**
- In `TopicDisentanglementModel`, the current implementation encodes the (parent, child) pair jointly via `[CLS]` rather than encoding them separately. The paper uses distinct `t_k` (parent) and `t_r` (child) representations.
- Evaluated on a small subset of DSTC-8 due to compute constraints (RTX 2060 6GB)

## Results

| Metric | Paper (DSTC-8 Task 2) | Mine |
|---|---|---|
| Recall@1 | — | TBD |
| F1 (Task 4) | — | TBD |

## How to run

```bash
pip install -r requirements.txt

# Download DSTC-8 data from https://github.com/dstc8-track2/NOESIS-II
# Place train.json / dev.json in topic-bert/data/

jupyter notebook topicbert.ipynb
```

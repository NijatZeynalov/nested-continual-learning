# Nested Continual Learning

When a model is trained on Task 1 and then fine-tuned on Task 2, its updated weights often overwrite previously learned representations, causing the model to forget Task 1 (catastrophic forgetting).
This project shows how a nested continual setup preserves prior knowledge while learning the new task.

In our case:
- **Task 1**: cat (0) vs dog (1)  
- **Task 2**: dog (0) vs horse (1) — the shared dog class flips label, which makes forgetting obvious.

Baseline fine-tunes the whole CNN on Task 2 and typically forgets Task 1. Continual freezes the backbone, adds a small adapter + new head for Task 2, and better preserves Task 1.


## Quick start
Install deps (choose the right torch wheel for your CUDA/CPU setup):

```bash
pip install -r requirements.txt
```

Train baseline (defaults: 10 epochs per task, batch size 128):

```bash
python3 baseline_train.py
```

Train continual (nested adapter):

```bash
python3 continual_train.py
```

## What each script does
1) Build loaders for both tasks (CIFAR-10 filtered + remapped labels).  
2) Train Task 1 with a 3-block CNN + classifier head.  
3) Task 2:  
   - **Baseline**: fine-tune the same backbone with a new head → Task1 accuracy usually drops (forgetting).  
   - **Continual**: freeze backbone, add adapter + new head (optionally unfreeze last block) → Task1 accuracy should stay close to pre-Task2.  
4) Save checkpoints and a text summary. `evaluate_models.py` reloads them and prints Task1-after-Task2 vs Task2 accuracies for both setups.

## Example results (CPU, 10+10 epochs, augment on)
- Baseline naive fine-tune: Task1 before 0.7130 → after 0.5255, Task2 0.8455 (clear catastrophic forgetting).
- Continual nested adapter: Task1 before 0.7185 → after 0.7185, Task2 0.8095 (forgetting eliminated, small trade-off on Task2).


## Architecture
- Backbone: Conv → ReLU → MaxPool (32ch), Conv → ReLU → MaxPool (64ch), Conv → ReLU (128ch).
- Head: global average pool + linear (2 logits).
- Adapter (continual Task2 path): Conv(3x3) + Conv(1x1) + ReLU stack inserted before the Task 2 head.

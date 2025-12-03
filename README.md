# Continual Learning with Nested Modules: Preventing Catastrophic Forgetting on CIFAR-10

Minimal setup that contrasts naive two-step fine-tuning against a nested/adaptive continual approach on CIFAR-10 binaries:

- **Task 1**: cat (0) vs dog (1)  
- **Task 2**: dog (0) vs horse (1) — the shared dog class flips label, which makes forgetting obvious.

Baseline fine-tunes the whole CNN on Task 2 and typically forgets Task 1. Continual freezes the backbone, adds a small adapter + new head for Task 2, and better preserves Task 1.

## Files
- `continual_utils.py` — shared loaders, models, train/eval helpers.
- `baseline_train.py` — naive Task1 → Task2 fine-tune, saves `checkpoints/baseline.pth`.
- `continual_train.py` — nested adapter training, saves `checkpoints/continual.pth`.
- `evaluate_models.py` — loads both checkpoints, evaluates Task1/Task2 to show forgetting vs preservation.
- `requirements.txt` — torch, torchvision.

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

Evaluate both checkpoints (uses test splits only):

```bash
python3 evaluate_models.py
```

Common flags:
- `--epochs-task1 / --epochs-task2` — more epochs (e.g., 20–30) make the gap clearer.
- `--batch-size` (default 128), `--lr` (default 1e-3), `--no-augment` to skip crop/flip.
- `--unfreeze-top` (continual) also fine-tunes the last conv block during Task 2.
- `--device {auto,cpu,cuda}` — force CPU/CUDA; use `--device cpu` if CUDA/CuDNN libs are mismatched.

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

## What we ran
- Environment: CPU (forced with `--device cpu`), defaults otherwise (batch 128, lr 1e-3, augment on).
- Commands executed:  
  - `python3 baseline_train.py --device cpu` → saved to `checkpoints/baseline.pth`, summary `baseline_results.txt`.  
  - `python3 continual_train.py --device cpu` → saved to `checkpoints/continual.pth`, summary `continual_results.txt`.  
- Observation: baseline forgets Task1; continual keeps Task1 intact while retaining solid Task2 performance.

## Architecture (conceptual)
- Backbone: Conv → ReLU → MaxPool (32ch), Conv → ReLU → MaxPool (64ch), Conv → ReLU (128ch).
- Head: global average pool + linear (2 logits).
- Adapter (continual Task2 path): Conv(3x3) + Conv(1x1) + ReLU stack inserted before the Task 2 head.

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from continual_utils import (
    ClassifierHead,
    ConvBackbone,
    evaluate,
    make_task_loaders,
    train_one_epoch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Naive two-step training (Task1 then Task2) on CIFAR-10 subsets."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="CIFAR-10 root.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-task1", type=int, default=10)
    parser.add_argument("--epochs-task2", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable random crop/flip for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline.pth",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="baseline_results.txt",
        help="Where to save a text summary.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force device selection; auto picks CUDA if available.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        device = torch.device(args.device)
    print(f"Using device: {device}")

    loaders = make_task_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augment=not args.no_augment,
    )
    train_t1, test_t1 = loaders["task1"]
    train_t2, test_t2 = loaders["task2"]

    backbone = ConvBackbone().to(device)
    head_t1 = ClassifierHead().to(device)
    head_t2 = ClassifierHead().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_t1 = optim.Adam(
        list(backbone.parameters()) + list(head_t1.parameters()), lr=args.lr
    )
    for epoch in range(args.epochs_task1):
        loss, acc = train_one_epoch(
            backbone, head_t1, train_t1, optimizer_t1, criterion, device
        )
        print(f"[Baseline][Task1] Epoch {epoch+1}/{args.epochs_task1} - loss {loss:.4f} acc {acc:.4f}")

    _, acc_t1_before = evaluate(backbone, head_t1, test_t1, device)

    optimizer_t2 = optim.Adam(
        list(backbone.parameters()) + list(head_t2.parameters()), lr=args.lr
    )
    for epoch in range(args.epochs_task2):
        loss, acc = train_one_epoch(
            backbone, head_t2, train_t2, optimizer_t2, criterion, device
        )
        print(f"[Baseline][Task2] Epoch {epoch+1}/{args.epochs_task2} - loss {loss:.4f} acc {acc:.4f}")

    _, acc_t2 = evaluate(backbone, head_t2, test_t2, device)
    _, acc_t1_after = evaluate(backbone, head_t1, test_t1, device)

    history = {
        "task1_before_task2": acc_t1_before,
        "task1_after_task2": acc_t1_after,
        "task2_final": acc_t2,
    }

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head_t1": head_t1.state_dict(),
            "head_t2": head_t2.state_dict(),
            "history": history,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path.resolve()}")

    results_path = Path(args.results_path)
    with results_path.open("w") as f:
        f.write("[Baseline]\n")
        f.write(f"task1_before_task2: {acc_t1_before:.4f}\n")
        f.write(f"task1_after_task2:  {acc_t1_after:.4f}\n")
        f.write(f"task2_final:        {acc_t2:.4f}\n")
    print(f"Saved summary to {results_path.resolve()}")

    print("\nFinal metrics (baseline naive fine-tune):")
    print(f"Task1 before Task2: {acc_t1_before:.4f}")
    print(f"Task1 after Task2:  {acc_t1_after:.4f}")
    print(f"Task2 final:        {acc_t2:.4f}")


if __name__ == "__main__":
    main()

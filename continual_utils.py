from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class RemappedDataset(Dataset):
    """Wraps a Subset and remaps original CIFAR-10 labels to binary targets."""

    def __init__(self, subset: Subset, label_map: Dict[int, int]):
        self.subset = subset
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, target = self.subset[idx]
        return image, self.label_map[int(target)]


def make_task_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    use_augment: bool = True,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """Build train/test loaders for Task1 (cat/dog) and Task2 (dog/horse)."""

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if use_augment
        else [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def _build_loaders(
        class_ids: Tuple[int, int],
        label_map: Dict[int, int],
        download: bool,
    ) -> Tuple[DataLoader, DataLoader]:
        train_base = datasets.CIFAR10(
            root=data_root, train=True, download=download, transform=train_transform
        )
        test_base = datasets.CIFAR10(
            root=data_root, train=False, download=False, transform=test_transform
        )

        train_indices = [i for i, t in enumerate(train_base.targets) if t in class_ids]
        test_indices = [i for i, t in enumerate(test_base.targets) if t in class_ids]

        train_subset = RemappedDataset(Subset(train_base, train_indices), label_map)
        test_subset = RemappedDataset(Subset(test_base, test_indices), label_map)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader

    # CIFAR-10 class ids: cat=3, dog=5, horse=7
    task1_loaders = _build_loaders((3, 5), {3: 0, 5: 1}, download=True)
    task2_loaders = _build_loaders((5, 7), {5: 0, 7: 1}, download=False)
    return {"task1": task1_loaders, "task2": task2_loaders}


class ConvBackbone(nn.Module):
    """Simple CNN backbone with three conv blocks."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Adapter(nn.Module):
    """Lightweight nested adapter added for Task2."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """Global average pooling followed by a linear classifier."""

    def __init__(self, in_channels: int = 128, num_classes: int = 2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)


def forward_pass(
    backbone: ConvBackbone, head: ClassifierHead, x: torch.Tensor, adapter: Adapter = None
) -> torch.Tensor:
    feats = backbone.forward_features(x)
    if adapter is not None:
        feats = adapter(feats)
    return head(feats)


def train_one_epoch(
    backbone: ConvBackbone,
    head: ClassifierHead,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    adapter: Adapter = None,
) -> Tuple[float, float]:
    backbone.train()
    head.train()
    if adapter is not None:
        adapter.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = forward_pass(backbone, head, images, adapter)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        total += batch_size

    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc


def evaluate(
    backbone: ConvBackbone,
    head: ClassifierHead,
    loader: DataLoader,
    device: torch.device,
    adapter: Adapter = None,
) -> Tuple[float, float]:
    backbone.eval()
    head.eval()
    if adapter is not None:
        adapter.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = forward_pass(backbone, head, images, adapter)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            total += batch_size

    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad

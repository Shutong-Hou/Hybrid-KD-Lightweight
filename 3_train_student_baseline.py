import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
import os
import importlib
get_cifar100_dataloaders = importlib.import_module('1_data_prepare').get_cifar100_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Training')
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss/(loop.n + 1e-8))
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct_top5 += correct.any(dim=0).sum().item()
            _, pred1 = outputs.max(1)
            correct_top1 += pred1.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct_top1 / total, 100. * correct_top5 / total

def main(seed):
    set_seed(seed)
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, test_loader = get_cifar100_dataloaders()

    model = mobilenet_v3_small(weights='DEFAULT')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    epochs = 20

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()

    top1, top5 = test(model, test_loader)
    print(f"Student Baseline Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
    torch.save(model.state_dict(), f"checkpoints/student_baseline_seed{seed}.pth")
    return top1, top5

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
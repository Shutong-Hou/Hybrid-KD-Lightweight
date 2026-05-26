import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50, mobilenet_v3_small
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

def kd_loss_fn(student_logits, teacher_logits, targets, T=4, alpha=0.9):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss

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

    teacher = resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 100)
    teacher.load_state_dict(torch.load(f"checkpoints/teacher_seed{seed}.pth", map_location=device))
    teacher = teacher.to(device)
    teacher.eval()

    student = mobilenet_v3_small(weights='DEFAULT')
    student.classifier[-1] = nn.Linear(student.classifier[-1].in_features, 100)
    student = student.to(device)

    optimizer = optim.SGD(student.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    epochs = 20

    for epoch in range(epochs):
        student.train()
        loop = tqdm(train_loader, desc=f'KD Epoch {epoch+1}')
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                s_logits = student(inputs)
                with torch.no_grad():
                    t_logits = teacher(inputs)
                loss = kd_loss_fn(s_logits, t_logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

    top1, top5 = test(student, test_loader)
    print(f"Standard KD Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
    torch.save(student.state_dict(), f"checkpoints/student_kd_seed{seed}.pth")
    return top1, top5

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
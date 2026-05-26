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

kd_hybrid = importlib.import_module('5_train_kd_hybrid')
set_seed = kd_hybrid.set_seed
features = kd_hybrid.features
get_feature_hook = kd_hybrid.get_feature_hook
FeatAdapter = kd_hybrid.FeatAdapter
test = kd_hybrid.test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed):
    set_seed(seed)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, test_loader = get_cifar100_dataloaders()

    teacher = resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 100)
    teacher.load_state_dict(torch.load(f"checkpoints/teacher_seed{seed}.pth", map_location=device))
    teacher = teacher.to(device).eval()
    teacher.layer3.register_forward_hook(get_feature_hook('t_feat'))

    student = mobilenet_v3_small(weights='DEFAULT')
    student.classifier[-1] = nn.Linear(student.classifier[-1].in_features, 100)
    student = student.to(device)
    student.features[11].register_forward_hook(get_feature_hook('s_feat'))

    adapter = FeatAdapter(96, 1024, (14, 14)).to(device)

    optimizer = optim.SGD(list(student.parameters()) + list(adapter.parameters()),
                          lr=0.2, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    epochs = 20

    for epoch in range(epochs):
        student.train()
        adapter.train()
        loop = tqdm(train_loader, desc=f'Ablation Epoch {epoch+1}')
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                s_logits = student(inputs)
                with torch.no_grad():
                    _ = teacher(inputs)
                t_feat = features['t_feat']
                s_feat = features['s_feat']
                if adapter.target_size != t_feat.shape[-2:]:
                    adapter.target_size = t_feat.shape[-2:]
                adapted = adapter(s_feat)
                feat_loss = F.mse_loss(adapted, t_feat)
                ce_loss = F.cross_entropy(s_logits, targets)
                loss = 0.95 * feat_loss + 0.05 * ce_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(student.parameters()) + list(adapter.parameters()), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

    top1, top5 = test(student, test_loader)
    print(f"Ablation (Feat only) Top-1: {top1:.2f}%")
    torch.save(student.state_dict(), f"checkpoints/student_ablation_seed{seed}.pth")
    return top1, top5

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
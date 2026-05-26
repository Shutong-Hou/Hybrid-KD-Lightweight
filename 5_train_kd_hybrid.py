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

features = {}
def get_feature_hook(name):
    def hook(model, input, output):
        features[name] = output
    return hook

class FeatAdapter(nn.Module):
    """1x1卷积 + BN + 自适应池化"""
    def __init__(self, in_channels, out_channels, target_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.target_size = target_size
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, self.target_size)
        x = self.conv(x)
        x = self.bn(x)
        return x

def hybrid_loss_fn(s_logits, t_logits, s_feat, t_feat, targets, adapter, T=4,
                   alpha=0.9, beta=0.02, gamma=0.08, use_feat=True):
    """
    alpha: KD软标签
    beta: 特征对齐（极低）
    gamma: 硬标签CE
    """
    kd_loss = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    ce_loss = F.cross_entropy(s_logits, targets)
    if use_feat:
        adapted = adapter(s_feat)
        feat_loss = F.mse_loss(adapted, t_feat)
        total = alpha * kd_loss + beta * feat_loss + gamma * ce_loss
    else:
        feat_loss = torch.tensor(0.0, device=device)
        total = (alpha + beta) * kd_loss + gamma * ce_loss  # 把beta的权重临时还给KD
    return total, kd_loss, feat_loss, ce_loss

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
    warmup_epochs = 2  # 前2个epoch不使用特征损失

    for epoch in range(epochs):
        student.train()
        adapter.train()
        use_feat = (epoch >= warmup_epochs)
        loop = tqdm(train_loader, desc=f'Hybrid KD Epoch {epoch+1}')
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
                with torch.no_grad():
                    t_logits = teacher(inputs)
                loss, kd_l, feat_l, ce_l = hybrid_loss_fn(
                    s_logits, t_logits, s_feat, t_feat, targets, adapter,
                    use_feat=use_feat
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(student.parameters()) + list(adapter.parameters()), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item(), kd=kd_l.item(), feat=feat_l.item(), ce=ce_l.item())
        scheduler.step()

    top1, top5 = test(student, test_loader)
    print(f"Hybrid KD Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
    torch.save(student.state_dict(), f"checkpoints/student_hybrid_seed{seed}.pth")
    torch.save(adapter.state_dict(), f"checkpoints/adapter_seed{seed}.pth")
    return top1, top5

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 手动填入你的真实数据，避免依赖results.csv
data = {
    'Method': [
        'Teacher',
        'Student\nBaseline',
        'Standard\nKD',
        'Hybrid KD\n(Ours)',
        'Ablation'
    ],
    'Mean': [82.99, 65.73, 69.68, 70.93, 44.23],
    'Std':  [0.15, 0.30, 1.13, 0.36, 1.17]
}

x = np.arange(len(data['Method']))
means = data['Mean']
stds = data['Std']

plt.figure(figsize=(10, 6))
bars = plt.bar(x, means, yerr=stds, capsize=5,
               color=['gray', 'lightblue', 'orange', 'steelblue', 'lightcoral'],
               edgecolor='black')
plt.ylabel('Top-1 Accuracy (%)')
plt.title('Performance Comparison on CIFAR-100 (mean ± std)')
plt.xticks(x, data['Method'], rotation=0)
plt.ylim(0, 90)

# 在柱子上标注数值
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{mean:.2f}±{std:.2f}', ha='center', fontsize=9)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/comparison_bar.png", dpi=300)
plt.show()
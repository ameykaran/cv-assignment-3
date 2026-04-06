import torch
import numpy as np
from dataset import get_dataloaders
from pointnet import PointNetClassifier, evaluate_with_critical_points
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, test_loader = get_dataloaders()
model = PointNetClassifier(num_classes=10)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

target_ns = [10, 50, 100, 150, 200, 300, 400, 500, 800, 1024]
accuracies = []

for n in target_ns:
    print(f"\\n--- Evaluating target_n = {n} ---")
    acc = evaluate_with_critical_points(model, test_loader, device, target_n=n)
    accuracies.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(target_ns, accuracies, marker='o', linestyle='-')
plt.title('Effect of Number of Critical Points on Accuracy')
plt.xlabel('Number of Sampled Critical Points')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.savefig('critical_points_accuracy_curve.png')

print("Experiment complete. Saved critical_points_accuracy_curve.png")

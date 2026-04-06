import torch
import numpy as np
from dataset import get_dataloaders
from pointnet import PointNetClassifier, evaluate_permutation_invariance, extract_critical_points
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, test_loader = get_dataloaders()
model = PointNetClassifier(num_classes=10)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

print("Evaluating...")

# Final test loss and accuracy
loss_fn = torch.nn.CrossEntropyLoss()
test_loss = 0.0
test_correct = 0
total = 0

with torch.no_grad():
    for points, labels in test_loader:
        points = points.to(device)
        labels = labels.to(device)
        logits, _ = model(points)
        loss = loss_fn(logits, labels)
        
        test_loss += loss.item() * labels.shape[0]
        test_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.shape[0]

test_loss /= max(1, total)
test_acc = test_correct / max(1, total)
print(f"Final Class Metrics -- Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Permutation invariance
from pointnet import evaluate_permutation_invariance
percent_changed = evaluate_permutation_invariance(model, test_loader, device)
print(f"Permuted invariant percent changing: {percent_changed:.2f}%")

# Evaluating Robustness
from pointnet import evaluate_with_critical_points
acc_critical = evaluate_with_critical_points(model, test_loader, device, target_n=300)
print(f"Critical point robustness subset accuracy: {acc_critical:.4f}")

# Generating visualizations
points, labels = test_loader.dataset.get_each_class_sample(n_samples_per_class=1)
points = points.to(device)

logits, point_features = model(points)
critical_points_batch = extract_critical_points(point_features, points)

def plot_pcd(ax, pts, c='gray', alpha=1.0, title=''):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=10, alpha=alpha)
    ax.set_title(title)
    ax.set_axis_off()

for i in range(5): # Just get 5
    critical_pts = critical_points_batch[i].cpu().numpy()
    orig_pts = points[i].cpu().numpy()
    
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_pcd(ax1, orig_pts, c='darkgray', title='Original')
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # faint full cloud
    plot_pcd(ax2, orig_pts, c='lightgray', alpha=0.1, title='Critical Points')
    # bright critical points
    ax2.scatter(critical_pts[:,0], critical_pts[:,1], critical_pts[:,2], c='red', s=20)
    
    plt.tight_layout()
    plt.savefig(f'critical_points_{i}.png')
    plt.close()

print("Saved critical points vis.")

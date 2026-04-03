"""
Input: (B, N, 3)

→ Transpose → (B, 3, N)

→ Shared MLP (Conv1D layers)
→ (B, 1024, N)

→ MaxPool over N
→ (B, 1024)

→ Fully Connected layers

→ Output: class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input: (B, N, 3) of point coordinates
        Output: (B, 1024), (B, 1024, N)
        """
        x = x.transpose(1, 2)   # → (B, 3, N)

        features = self.mlp(x)  # → (B, 1024, N)

        # Save BEFORE pooling (needed for critical points)
        point_features = features

        # Global max pooling
        global_feature, _ = torch.max(features, 2)  # → (B, 1024)

        return global_feature, point_features
    
class PointNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = PointNetEncoder()

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        global_feature, point_features = self.encoder(x)
        logits = self.fc(global_feature)

        return logits, point_features

def permute_points(points):
    """
    points: (B, N, 3)
    """
    B, N, _ = points.shape

    permuted = []

    for b in range(B):
        idx = torch.randperm(N)
        permuted.append(points[b][idx])

    return torch.stack(permuted)


def evaluate_permutation_invariance(model, data_loader, device):
    model.eval()

    total_samples = 0
    changed_predictions = 0

    with torch.no_grad():
        for points, _ in data_loader:
            points = points.to(device)
            
            # Original prediction
            logits_orig, _ = model(points)            
            preds_orig = torch.argmax(logits_orig, dim=1)
            
            # Permuted input
            permuted_points = permute_points(points)
            permuted_points = permuted_points.to(device)

            logits_perm, _ = model(permuted_points)
            preds_perm = torch.argmax(logits_perm, dim=1)

            # Compare
            diff = (preds_orig != preds_perm).sum().item()

            changed_predictions += diff
            total_samples += points.shape[0]

    percent_changed = (changed_predictions / total_samples) * 100

    print(f"Percentage of changed predictions: {percent_changed:.2f}%")

    return percent_changed


def eval_model(model, val_loader, loss_fn, device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        for points, labels in tqdm(val_loader):
            points = points.to(device)
            labels = labels.to(device)
            batch_size = labels.shape[0]

            logits, _ = model(points)

            loss = loss_fn(logits, labels)

            val_loss += loss.item() * batch_size
            val_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            val_samples += batch_size

    val_loss /= max(1, val_samples)
    val_acc = val_correct / max(1, val_samples)

    print(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return val_loss, val_acc

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()  # FIXED
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for points, labels in tqdm(train_loader):
            points = points.to(device)      # (B, N, 3)
            labels = labels.to(device)      # (B,)
            batch_size = labels.shape[0]

            logits, _ = model(points)

            loss = loss_fn(logits, labels)  # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_samples += batch_size

        scheduler.step()

        train_loss /= max(1, train_samples)
        train_acc = train_correct / max(1, train_samples)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, "
            f"Acc={train_acc:.4f}"
        )

        # Validation
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        # Logging
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)



from dataset import get_dataloaders, visualize_interactive
train_loader, val_loader, test_loader = get_dataloaders()

model = PointNetClassifier(num_classes=10)
epochs = 50
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# wandb.init(project="pointnet", config={
#     "epochs": epochs,
#     "batch_size": train_loader.batch_size,
#     "learning_rate": lr,
#     "optimizer": "Adam",
#     "loss_function": "CrossEntropyLoss"
#     })
# train_model(model, train_loader, val_loader, epochs, lr=lr, device=device)


loaded_state = torch.load("best_model.pth", map_location=device)
model.load_state_dict(loaded_state)
model = model.to(device)

# evaluate_permutation_invariance(model, test_loader, device=device)

points, labels = next(iter(test_loader))
# points, labels = next(iter(test_loader))
# permuted_points = permute_points(points)

# print(f"Original label: {labels[0].item()}")
# # original
# visualize_interactive(points[0].cpu())
# visualize_interactive(permuted_points[0].cpu())


def extract_critical_points(point_features, input_points):
    """
    point_features: (B, 1024, N)
    input_points:   (B, N, 3)
    """

    B, D, N = point_features.shape

    # Indices of max contributors
    critical_indices = torch.argmax(point_features, dim=2)  # (B, 1024)

    critical_points_batch = []

    for b in range(B):
        idx = critical_indices[b]  # (1024,)

        # Remove duplicates
        unique_idx = torch.unique(idx)

        # Select corresponding points
        critical_pts = input_points[b][unique_idx]  # (K, 3)

        critical_points_batch.append(critical_pts)

    return critical_points_batch



def evaluate_with_critical_points(model, data_loader, device, target_n=2048):
    model.eval()

    total = 0
    correct_critical = 0
    correct_normal = 0

    with torch.no_grad():
        for points, labels in data_loader:
            points = points.to(device)      # (B, N, 3)
            labels = labels.to(device)

            # Normal prediction
            logits, point_features = model(points)
            preds = torch.argmax(logits, dim=1)

            correct_normal += (preds == labels).sum().item()

            # Extract critical points
            critical_points_batch = extract_critical_points(point_features, points)

            new_points_batch = []

            for pts in critical_points_batch:
                K = pts.shape[0]

                # Resample to target_n
                if K >= target_n:
                    idx = torch.randperm(K, device=pts.device)[:target_n]
                else:
                    idx = torch.randint(0, K, (target_n,), device=pts.device)

                new_pts = pts[idx]
                new_points_batch.append(new_pts)

            new_points_batch = torch.stack(new_points_batch).to(device)

            print(f"Original: {points.shape}, Critical-resampled: {new_points_batch.shape}")

            # Re-run model
            logits_new, _ = model(new_points_batch)
            preds_new = torch.argmax(logits_new, dim=1)

            correct_critical += (preds_new == labels).sum().item()
            total += labels.size(0)

        print(f"Avg critical points: {sum(len(p) for p in critical_points_batch)/len(critical_points_batch):.2f}")
    acc_normal = correct_normal / total
    acc_critical = correct_critical / total

    print(f"Accuracy with original points: {acc_normal:.4f}")
    print(f"Accuracy with critical points only: {acc_critical:.4f}")

    return acc_critical


# evaluate_with_critical_points(model, test_loader, device=device, target_n=100)

points, labels = test_loader.dataset.get_each_class_sample(n_samples_per_class=1)
points = points.to(device)

logits, point_features = model(points)
critical_points_batch = extract_critical_points(point_features, points)

for i in range(points.shape[0]):
    critical_points = critical_points_batch[i].cpu()
    print(f"Sample {i}: Original points={points[i].shape[0]}, Critical points={critical_points.shape[0]}")
    visualize_interactive(points[i].cpu(), critical_points)
    

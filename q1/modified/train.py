import torch
import argparse
import os
import math
import numpy as np
import yaml
import random
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import wandb

# Standard detection imports
import detection
from detection.faster_rcnn import FastRCNNPredictor
from dataset.st import SceneTextDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# 1. Geometry & Math Helpers
# ---------------------------------------------------------------------------

def collate_function(data):
    return tuple(zip(*data))

def obb_to_aabb(boxes):
    """[xc, yc, w, h, theta] → [x1, y1, x2, y2]"""
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=1)

def aabb_and_angle_to_corners(boxes_aabb, angles):
    """Reconstruct 4 corners of oriented boxes from AABB + angle."""
    boxes_aabb = boxes_aabb.cpu().numpy()
    angles = angles.cpu().numpy()
    N = len(boxes_aabb)
    corners = np.zeros((N, 4, 2), dtype=np.float32)
    
    for i in range(N):
        x1, y1, x2, y2 = boxes_aabb[i]
        xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        theta_rad = math.radians(angles[i])
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        
        dx = np.array([w/2, w/2, -w/2, -w/2])
        dy = np.array([h/2, -h/2, -h/2, h/2])
        
        for j in range(4):
            corners[i, j, 0] = xc + cos_t * dx[j] - sin_t * dy[j]
            corners[i, j, 1] = yc + sin_t * dx[j] + cos_t * dy[j]
    return corners

def polygon_iou(poly1, poly2):
    p1, p2 = poly1.astype(np.float32), poly2.astype(np.float32)
    area1, area2 = cv2.contourArea(p1), cv2.contourArea(p2)
    _, inter_pts = cv2.intersectConvexConvex(p1, p2)
    if inter_pts is None or len(inter_pts) == 0: return 0.0
    inter_area = cv2.contourArea(inter_pts)
    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0

# ---------------------------------------------------------------------------
# 2. Unified Metric Calculation
# ---------------------------------------------------------------------------

def compute_obb_map(pred_list, gt_list, iou_threshold=0.5):
    all_classes = set()
    for item in pred_list + gt_list:
        all_classes.update(item['labels'].cpu().tolist())
    all_classes.discard(0)

    if not all_classes: return {"map": 0.0, "precision": 0.0, "recall": 0.0, "map_per_class": {}}

    ap_per_class = {}
    all_tp, all_fp, total_gt = 0, 0, 0

    for cls in sorted(all_classes):
        det_scores, det_tp_flags, n_gt_total = [], [], 0

        for p_img, g_img in zip(pred_list, gt_list):
            pmask, gmask = (p_img['labels'] == cls), (g_img['labels'] == cls)
            p_boxes, p_angles, p_scores = p_img['boxes'][pmask], p_img['angles'][pmask], p_img['scores'][pmask]
            g_boxes, g_angles = g_img['boxes'][gmask], g_img['angles'][gmask]
            
            n_gt = int(gmask.sum().item())
            n_gt_total += n_gt
            if len(p_boxes) == 0: continue

            order = p_scores.argsort(descending=True)
            p_boxes, p_angles, p_scores = p_boxes[order], p_angles[order], p_scores[order]
            p_corners = aabb_and_angle_to_corners(p_boxes, p_angles)
            g_corners = aabb_and_angle_to_corners(g_boxes, g_angles)

            matched_gt = set()
            for pi in range(len(p_boxes)):
                best_iou, best_gi = 0.0, -1
                for gi in range(n_gt):
                    if gi in matched_gt: continue
                    iou = polygon_iou(p_corners[pi], g_corners[gi])
                    if iou > best_iou: best_iou, best_gi = iou, gi

                det_scores.append(p_scores[pi].item())
                if best_iou >= iou_threshold:
                    det_tp_flags.append(1)
                    matched_gt.add(best_gi)
                else:
                    det_tp_flags.append(0)

        if n_gt_total == 0:
            ap_per_class[cls] = 0.0
            continue

        order = np.argsort(det_scores)[::-1]
        tp_arr = np.array(det_tp_flags)[order]
        cum_tp, cum_fp = np.cumsum(tp_arr), np.cumsum(1 - tp_arr)
        prec, rec = cum_tp / (cum_tp + cum_fp + 1e-9), cum_tp / (n_gt_total + 1e-9)
        
        ap = 0.0
        for thr in np.linspace(0, 1, 101):
            p_at_r = prec[rec >= thr]
            ap += (np.max(p_at_r) if len(p_at_r) > 0 else 0.0)
        ap_per_class[cls] = ap / 101.0
        
        all_tp += int(tp_arr.sum())
        all_fp += int((1 - tp_arr).sum())
        total_gt += n_gt_total

    return {
        "map": float(np.mean(list(ap_per_class.values()))),
        "precision": all_tp / (all_tp + all_fp + 1e-9),
        "recall": all_tp / (total_gt + 1e-9),
        "map_per_class": ap_per_class
    }

# ---------------------------------------------------------------------------
# 3. Core Engine (Training / Evaluation / Visualization)
# ---------------------------------------------------------------------------

def calculate_batch_loss(batch_losses):
    return (batch_losses['loss_classifier'] + batch_losses['loss_box_reg'] +
            batch_losses['loss_rpn_box_reg'] + batch_losses['loss_objectness'] +
            batch_losses['loss_angle'])

def build_targets(targets, device):
    return [{
        'boxes': obb_to_aabb(t['bboxes'].float()).to(device),
        'labels': t['labels'].long().to(device),
        'angles': t['bboxes'][:, 4].float().to(device)
    } for t in targets]

def save_visuals(image, target, output, epoch, idx, out_dir, mode="val"):
    img_np = np.clip(image.cpu().permute(1, 2, 0).numpy(), 0, 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_np)

    gt_aabb = obb_to_aabb(target['bboxes'].float())
    gt_angles = target['bboxes'][:, 4].float()
    gt_corners = aabb_and_angle_to_corners(gt_aabb, gt_angles)
    for pts in gt_corners:
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor=(0, 1, 0), linewidth=1.5))

    scores = output['scores'].cpu()
    if len(scores) > 0:
        order = scores.argsort(descending=True)[:20]
        p_boxes = output['boxes'].cpu()[order]
        p_angles = output['angles'].cpu()[order]
        p_corners = aabb_and_angle_to_corners(p_boxes, p_angles)
        for pts in p_corners:
            ax.add_patch(plt.Polygon(pts, fill=False, edgecolor=(0, 0, 1), linewidth=1.5))

    ax.legend(handles=[
        Line2D([0], [0], color=(0, 1, 0), lw=2, label='GT'),
        Line2D([0], [0], color=(0, 0, 1), lw=2, label='Pred')
    ], loc='upper right')
    
    ax.set_title(f'Epoch {epoch} | {mode} Visual {idx}')
    ax.axis('off')
    
    if epoch is None:
        save_path = os.path.join(out_dir, f'{mode}_idx{idx:03d}.jpg')
    else:
        save_path = os.path.join(out_dir, f'{mode}_ep{epoch:03d}_idx{idx:03d}.jpg')
        
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    wandb.log({f"{mode}/examples": wandb.Image(save_path)}, step=epoch if epoch is not None else None)

def run_evaluation(model, dataloader, device, hname, epoch, mode="val", vis_limit=8):
    model.eval()
    losses, pred_all, gt_all = [], [], []
    out_dir = f'outputs/{hname}/'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (ims, targets, paths) in enumerate(tqdm(dataloader, desc=mode.capitalize())):
            images = [im.float().to(device) for im in ims]
            formatted_targets = build_targets(targets, device)

            # 1. Loss calculation (requires train mode)
            model.train()
            batch_losses = model(images, formatted_targets)
            losses.append(calculate_batch_loss(batch_losses).item())
            
            # 2. Inference
            model.eval()
            outputs = model(images)

            for j, (out, target) in enumerate(zip(outputs, targets)):
                pred_all.append({k: v.cpu() for k, v in out.items()})
                gt_all.append({
                    'boxes': obb_to_aabb(target['bboxes'].float()),
                    'angles': target['bboxes'][:, 4].float(),
                    'labels': target['labels']
                })
                
                if i < vis_limit:
                    save_visuals(images[j], target, out, epoch, i, out_dir, mode=mode)

    metrics_by_iou = {str(thr): compute_obb_map(pred_all, gt_all, thr) for thr in [0.25, 0.50, 0.75]}
    model.train()
    return float(np.mean(losses)), metrics_by_iou

# ---------------------------------------------------------------------------
# 4. Main Training Routine
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_cfg, ds_cfg = config['train_params'], config['dataset_params']
    
    # Seeds
    seed = train_cfg['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

    # Data
    train_loader = DataLoader(SceneTextDataset('train', ds_cfg['root_dir']), batch_size=1, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(SceneTextDataset('val', ds_cfg['root_dir']), batch_size=1, collate_fn=collate_function)
    test_loader = DataLoader(SceneTextDataset('test', ds_cfg['root_dir']), batch_size=1, collate_fn=collate_function)

    # Model
    model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=600, max_size=1000,
        angle_mode=args.angle_mode, num_angle_bins=args.num_angle_bins,
        angle_loss_weight=args.angle_loss_weight
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=ds_cfg['num_classes'],
        angle_mode=args.angle_mode, num_angle_bins=args.num_angle_bins
    )
    model.to(DEVICE)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-5, momentum=0.9)
    
    hname = f"mode{args.angle_mode}_bins{args.num_angle_bins}_aw{args.angle_loss_weight}"
    wandb.init(project="faster-rcnn-obb", name=hname, config=vars(args))
    
    best_val_loss = float('inf')

    for epoch in range(train_cfg['num_epochs']):
        model.train()
        train_losses = []
        for ims, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            optimizer.zero_grad()
            images = [im.float().to(DEVICE) for im in ims]
            loss_dict = model(images, build_targets(targets, DEVICE))
            loss = calculate_batch_loss(loss_dict)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        val_loss, metrics = run_evaluation(model, val_loader, DEVICE, hname, epoch, mode="val")
        
        # Log to Console & WandB
        print(f"\nEpoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f}")
        print(f"mAP@0.5: {metrics['0.5']['map']:.4f} | Precision: {metrics['0.5']['precision']:.4f}")

        wandb.log({
            "train/loss": np.mean(train_losses), "val/loss": val_loss,
            "val/map_50": metrics['0.5']['map'], "val/precision_50": metrics['0.5']['precision']
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(train_cfg['task_name'], f"{hname}_best.pth")
            os.makedirs(train_cfg['task_name'], exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    # Final Test Suite
    print("\n--- Starting Final Test Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(train_cfg['task_name'], f"{hname}_best.pth")))
    test_loss, test_metrics = run_evaluation(model, test_loader, DEVICE, hname, None, mode="test")
    
    print(f"Final Test Loss: {test_loss:.4f}")
    for thr in ['0.25', '0.5', '0.75']:
        m = test_metrics.get(thr, test_metrics.get(f'{float(thr):.2f}'))
        print(f"Test IoU {thr} -> mAP: {m['map']:.4f}, P: {m['precision']:.4f}, R: {m['recall']:.4f}")

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/st.yaml', type=str)
    args = parser.parse_args()
    args.rpn_post_nms_top_n_train, args.rpn_fg_iou_thresh, args.rpn_positive_fraction = 1000, 0.7, 0.5

    # configs = [('regression', 36, 1.0), ('classification', 36, 1.0)]
    # for mode, bins, weight in configs:
        # args.angle_mode, args.num_angle_bins, args.angle_loss_weight = mode, bins, weight
        # train(args)
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
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
import wandb

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collate_function(data):
    return tuple(zip(*data))


def obb_to_aabb(boxes):
    """[xc, yc, w, h, theta] → [x1, y1, x2, y2]  (theta ignored)."""
    xc, yc = boxes[:, 0], boxes[:, 1]
    w,  h  = boxes[:, 2], boxes[:, 3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def aabb_and_angle_to_corners(boxes_aabb: torch.Tensor, angles: torch.Tensor) -> np.ndarray:
    """
    Reconstruct the 4 corners of an oriented box from its AABB approximation + angle.

    boxes_aabb : [N, 4]  (x1, y1, x2, y2)
    angles     : [N]     degrees
    Returns    : [N, 4, 2]  corner coordinates
    """
    boxes_aabb = boxes_aabb.cpu().numpy()
    angles = angles.cpu().numpy()
    N = len(boxes_aabb)
    corners = np.zeros((N, 4, 2), dtype=np.float32)
    for i in range(N):
        x1, y1, x2, y2 = boxes_aabb[i]
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w  = x2 - x1
        h  = y2 - y1
        theta_rad = math.radians(angles[i])
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        # unrotated half-extents
        dx = [w / 2,  w / 2, -w / 2, -w / 2]
        dy = [h / 2, -h / 2, -h / 2,  h / 2]
        for j in range(4):
            corners[i, j, 0] = xc + cos_t * dx[j] - sin_t * dy[j]
            corners[i, j, 1] = yc + sin_t * dx[j] + cos_t * dy[j]
    return corners


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    Compute IoU between two convex quadrilaterals using cv2.
    poly1, poly2: (4, 2) float arrays.
    """
    p1 = poly1.astype(np.float32)
    p2 = poly2.astype(np.float32)
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    inter_type, inter_pts = cv2.intersectConvexConvex(p1, p2)
    if inter_pts is None or len(inter_pts) == 0:
        return 0.0
    inter_area = cv2.contourArea(inter_pts)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_obb_map(
    pred_boxes_list,   # List[Tensor(N,4)]  AABB predictions
    pred_angles_list,  # List[Tensor(N,)]
    pred_scores_list,  # List[Tensor(N,)]
    pred_labels_list,  # List[Tensor(N,)]
    gt_boxes_list,     # List[Tensor(M,4)]  AABB gt
    gt_angles_list,    # List[Tensor(M,)]
    gt_labels_list,    # List[Tensor(M,)]
    iou_threshold: float = 0.5,
):
    """
    Compute per-class AP and mAP using oriented bounding box IoU.
    Returns a dict with keys: 'map', 'map_per_class', 'precision', 'recall'.
    """
    # Gather all class ids (excluding background 0)
    all_classes = set()
    for lbls in pred_labels_list + gt_labels_list:
        all_classes.update(lbls.cpu().tolist())
    all_classes.discard(0)
    if not all_classes:
        return {"map": 0.0, "map_per_class": {}, "precision": 0.0, "recall": 0.0}

    ap_per_class = {}
    all_tp, all_fp, all_fn = 0, 0, 0

    for cls in sorted(all_classes):
        # Collect all predictions for this class across all images
        det_scores, det_tp_flags = [], []
        n_gt_total = 0

        for img_idx in range(len(pred_boxes_list)):
            # ---- predictions for this class ----
            pmask = (pred_labels_list[img_idx] == cls)
            p_boxes  = pred_boxes_list[img_idx][pmask]
            p_angles = pred_angles_list[img_idx][pmask]
            p_scores = pred_scores_list[img_idx][pmask]

            # ---- ground truth for this class ----
            gmask = (gt_labels_list[img_idx] == cls)
            g_boxes  = gt_boxes_list[img_idx][gmask]
            g_angles = gt_angles_list[img_idx][gmask]
            n_gt = int(gmask.sum().item())
            n_gt_total += n_gt

            if len(p_boxes) == 0:
                continue

            # sort predictions by score (descending)
            order = p_scores.argsort(descending=True)
            p_boxes  = p_boxes[order]
            p_angles = p_angles[order]
            p_scores = p_scores[order]

            matched_gt = set()

            if n_gt == 0:
                for _ in range(len(p_boxes)):
                    det_scores.append(p_scores[_].item())
                    det_tp_flags.append(0)
                continue

            p_corners = aabb_and_angle_to_corners(p_boxes, p_angles)  # (N,4,2)
            g_corners = aabb_and_angle_to_corners(g_boxes, g_angles)  # (M,4,2)

            for pi in range(len(p_boxes)):
                best_iou, best_gi = 0.0, -1
                for gi in range(n_gt):
                    if gi in matched_gt:
                        continue
                    iou = polygon_iou(p_corners[pi], g_corners[gi])
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi

                det_scores.append(p_scores[pi].item())
                if best_iou >= iou_threshold and best_gi not in matched_gt:
                    det_tp_flags.append(1)
                    matched_gt.add(best_gi)
                else:
                    det_tp_flags.append(0)

        if n_gt_total == 0:
            ap_per_class[cls] = 0.0
            continue

        # Sort by score descending and compute AP (11-point interpolation)
        order = np.argsort(det_scores)[::-1]
        tp_arr = np.array(det_tp_flags)[order]
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)

        precision_arr = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall_arr    = cum_tp / (n_gt_total + 1e-9)

        # 101-point interpolation (COCO-style)
        ap = 0.0
        for thr in np.linspace(0, 1, 101):
            p_at_r = precision_arr[recall_arr >= thr]
            ap += (np.max(p_at_r) if len(p_at_r) > 0 else 0.0)
        ap /= 101.0
        ap_per_class[cls] = ap

        # aggregate TP/FP/FN for global P/R
        tp_total = int(tp_arr.sum())
        fp_total = int((1 - tp_arr).sum())
        fn_total = n_gt_total - tp_total
        all_tp += tp_total
        all_fp += fp_total
        all_fn += fn_total

    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    denom_p = all_tp + all_fp
    denom_r = all_tp + all_fn
    global_precision = all_tp / denom_p if denom_p > 0 else 0.0
    global_recall    = all_tp / denom_r if denom_r > 0 else 0.0

    return {
        "map": mean_ap,
        "map_per_class": ap_per_class,
        "precision": global_precision,
        "recall": global_recall,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# def draw_obb(ax, boxes_aabb, angles, color, label_prefix=""):
#     """Draw oriented bounding boxes on a matplotlib Axes."""
#     if len(boxes_aabb) == 0:
#         return
#     corners = aabb_and_angle_to_corners(boxes_aabb, angles)  # (N,4,2)
#     for i, pts in enumerate(corners):
#         poly = plt.Polygon(pts, fill=False, edgecolor=color, linewidth=1.5)
#         ax.add_patch(poly)


#             draw_obb(ax, gt_boxes_aabb,  gt_angles,  color='lime')
#             draw_obb(ax, pred_boxes,     pred_angles, color='red')

#             # Score labels for predictions
#             for bi in range(len(pred_boxes)):
#                 x1, y1, _, _ = pred_boxes[bi].tolist()
#                 ax.text(x1, y1 - 2, f'{pred_scores[bi]:.2f}',
#                         color='red', fontsize=6, va='bottom')




def draw_obb(ax, boxes_aabb, angles, color, label_prefix=""):
    """Draw oriented bounding boxes on a matplotlib Axes."""
    if len(boxes_aabb) == 0:
        return

    corners = aabb_and_angle_to_corners(boxes_aabb, angles)  # (N,4,2)

    for pts in corners:
        poly = plt.Polygon(pts, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(poly)


def draw_pred_obb(ax, ground_truth_boxes, predicted_boxes, pred_scores):
    for i in range(len(predicted_boxes)):

        gt_boxes_aabb = obb_to_aabb(ground_truth_boxes[i].float())
        gt_angles = ground_truth_boxes[i][:, 4].float()

        pred_boxes_aabb = obb_to_aabb(predicted_boxes[i].float())
        pred_angles = predicted_boxes[i][:, 4].float()

        # Draw boxes
        draw_obb(ax, gt_boxes_aabb, gt_angles, color=(0, 255, 0))
        draw_obb(ax, pred_boxes_aabb, pred_angles, color=(0, 0, 255))


def visualize_predictions(
    val_dataset,
    model,
    hname: str,
    epoch: int,
    num_images: int = 8,
    iou_threshold: float = 0.5,
):
    """
    For `num_images` validation images overlay GT (green) and predictions (red).
    Saves to outputs/{hname}/vis/epoch_{epoch:03d}_img_{i}.jpg
    """
    out_dir = f'outputs/{hname}/vis/'
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    count = 0
    with torch.no_grad():
        for ims, targets, im_paths in val_dataset:
            if count >= num_images:
                break
            image = ims[0].float().to(device)
            target = targets[0]

            output = model([image])[0]   # single image → first result dict

            # GT
            # gt_boxes_aabb = obb_to_aabb(target['bboxes'].float())
            # gt_angles = target['bboxes'][:, 4]

            # Predictions
            pred_boxes  = output['boxes'].cpu()
            pred_angles = output['angles'].cpu()
            pred_scores = output['scores'].cpu()

            # Convert image tensor → numpy for display
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(img_np)
            # draw_obb(ax, gt_boxes_aabb,  gt_angles,  color='lime')
            # draw_obb(ax, pred_boxes,     pred_angles, color='red')
            draw_pred_obb(ax, target['bboxes'], pred_boxes, pred_scores)

            # Score labels for predictions
            # for bi in range(len(pred_boxes)):
            #     x1, y1, _, _ = pred_boxes[bi].tolist()
            #     ax.text(x1, y1 - 2, f'{pred_scores[bi]:.2f}',
            #             color='red', fontsize=6, va='bottom')

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lime', lw=2, label='GT'),
                Line2D([0], [0], color='red',  lw=2, label='Pred'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax.set_title(f'Epoch {epoch} | {os.path.basename(im_paths[0])}', fontsize=9)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'{out_dir}epoch_{epoch:03d}_img_{count:03d}.jpg', dpi=100, bbox_inches='tight')
            plt.close(fig)
            count += 1
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_targets(targets, device):
    new_targets = []
    for target in targets:
        t = {}
        t['boxes']  = obb_to_aabb(target['bboxes'].float()).to(device)
        t['labels'] = target['labels'].long().to(device)
        t['angles'] = target['bboxes'][:, 4].float().to(device)
        new_targets.append(t)
    return new_targets


def run_val_epoch(model, val_dataset, device, hname, epoch,
                  vis=True, num_vis_images=8, iou_thresholds=(0.25, 0.5, 0.75)):
    """
    Run one validation pass.
    Returns avg_val_loss and a metrics dict at each requested IoU threshold.
    """
    model.eval()
    val_losses = []

    # Accumulators for OBB-mAP
    pred_boxes_all, pred_angles_all, pred_scores_all, pred_labels_all = [], [], [], []
    gt_boxes_all,   gt_angles_all,   gt_labels_all                    = [], [], []

    vis_count = 0
    out_dir = f'outputs/{hname}/vis/'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for val_step, (ims, targets, im_paths) in enumerate(tqdm(val_dataset, desc='Val')):
            images = [im.float().to(device) for im in ims]
            new_targets = build_targets(targets, device)

            # ---- loss (requires train mode) ----
            model.train()
            batch_losses = model(images, new_targets)
            loss = (batch_losses['loss_classifier']
                  + batch_losses['loss_box_reg']
                  + batch_losses['loss_rpn_box_reg']
                  + batch_losses['loss_objectness']
                  + batch_losses['loss_angle'])
            val_losses.append(loss.item())

            # ---- predictions ----
            model.eval()
            outputs = model(images)

            for i, (out, target) in enumerate(zip(outputs, targets)):
                pred_boxes_all.append(out['boxes'].cpu())
                pred_angles_all.append(out['angles'].cpu())
                pred_scores_all.append(out['scores'].cpu())
                pred_labels_all.append(out['labels'].cpu())

                gt_boxes_all.append(obb_to_aabb(target['bboxes'].float()))
                gt_angles_all.append(target['bboxes'][:, 4].float())
                gt_labels_all.append(target['labels'])

                # Visualize a fixed subset
                if vis and vis_count < num_vis_images:
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np, 0, 1)
                    gt_boxes_aabb = obb_to_aabb(target['bboxes'].float())
                    gt_angles     = target['bboxes'][:, 4]

                    # Sort predictions by score descending, keep top-20
                    TOP_K = 20
                    p_scores = out['scores'].cpu()
                    p_boxes  = out['boxes'].cpu()
                    p_angles = out['angles'].cpu()
                    if len(p_scores) > 0:
                        order    = p_scores.argsort(descending=True)[:TOP_K]
                        p_scores = p_scores[order]
                        p_boxes  = p_boxes[order]
                        p_angles = p_angles[order]

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(img_np)
                    draw_obb(ax, gt_boxes_aabb, gt_angles, color='lime')
                    draw_obb(ax, p_boxes,       p_angles,  color='red')

                    for bi in range(len(p_boxes)):
                        x1, y1, _, _ = p_boxes[bi].tolist()
                        ax.text(x1, y1 - 2, f"{p_scores[bi].item():.2f}",
                                color='red', fontsize=6, va='bottom')

                    from matplotlib.lines import Line2D
                    ax.legend(handles=[
                        Line2D([0],[0], color='lime', lw=2, label='GT'),
                        Line2D([0],[0], color='red',  lw=2, label='Pred'),
                    ], loc='upper right', fontsize=8)
                    ax.set_title(f'Epoch {epoch} | {os.path.basename(im_paths[i])}', fontsize=9)
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{out_dir}epoch_{epoch:03d}_img_{vis_count:03d}.jpg', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    wandb.log({"val/examples": wandb.Image(f'{out_dir}epoch_{epoch:03d}_img_{vis_count:03d}.jpg')}, step=epoch)
                    vis_count += 1

    # ---- compute OBB-mAP at multiple thresholds ----
    # Use string keys to avoid float identity/equality pitfalls with dict lookup
    metrics_by_iou = {}
    for iou_thr in iou_thresholds:
        m = compute_obb_map(
            pred_boxes_all, pred_angles_all, pred_scores_all, pred_labels_all,
            gt_boxes_all,   gt_angles_all,   gt_labels_all,
            iou_threshold=iou_thr
        )
        metrics_by_iou[f'{iou_thr:.2f}'] = m

    model.train()
    return float(np.mean(val_losses)), metrics_by_iou


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config   = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st_train = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    st_val   = SceneTextDataset('val',   root_dir=dataset_config['root_dir'])
    st_test  = SceneTextDataset('test',  root_dir=dataset_config['root_dir'])

    train_dataset = DataLoader(st_train, batch_size=1, shuffle=True,  num_workers=2, collate_fn=collate_function)
    val_dataset   = DataLoader(st_val,   batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function)
    test_dataset  = DataLoader(st_test,  batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function)

    angle_mode_val       = getattr(args, 'angle_mode',        'regression')
    num_angle_bins_val   = getattr(args, 'num_angle_bins',    36)
    angle_loss_weight_val = getattr(args, 'angle_loss_weight', 1.0)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=600,
        max_size=1000,
        rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
        rpn_fg_iou_thresh=args.rpn_fg_iou_thresh,
        rpn_positive_fraction=args.rpn_positive_fraction,
        angle_mode=angle_mode_val,
        num_angle_bins=num_angle_bins_val,
        angle_loss_weight=angle_loss_weight_val,
    )

    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'],
        angle_mode=angle_mode_val,
        num_angle_bins=num_angle_bins_val,
    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        lr=1e-4,
        weight_decay=5e-5,
        momentum=0.9,
    )

    num_epochs = train_config['num_epochs']

    hname = (
        f'mode{angle_mode_val}'
        f'_bins{num_angle_bins_val}'
        f'_aw{angle_loss_weight_val}'
    )

    wandb.init(
        project="faster-rcnn-obb",
        name=f"{hname}",
        config={
            "lr": 1e-4,
            "epochs": train_config['num_epochs'],
            "angle_mode": angle_mode_val,
            "num_angle_bins": num_angle_bins_val,
            "angle_loss_weight": angle_loss_weight_val,
            "rpn_post_nms_top_n_train": args.rpn_post_nms_top_n_train,
            "rpn_fg_iou_thresh": args.rpn_fg_iou_thresh,
            "rpn_positive_fraction": args.rpn_positive_fraction,
        }
    )

    os.makedirs(f'outputs/{hname}/vis/', exist_ok=True)

    best_val_loss = float('inf')

    # ---- history for plotting ----
    history = {
        'train_loss': [], 'val_loss': [],
        'map_50': [], 'precision_50': [], 'recall_50': [],
        'map_25': [], 'map_75': [],
    }

    for epoch in range(num_epochs):
        faster_rcnn_model.train()

        rpn_cls_losses, rpn_loc_losses = [], []
        frcnn_cls_losses, frcnn_loc_losses, angle_losses = [], [], []
        total_train_losses = []

        for ims, targets, _ in tqdm(train_dataset, desc=f'Epoch {epoch} Train'):
            optimizer.zero_grad()
            images      = [im.float().to(device) for im in ims]
            new_targets = build_targets(targets, device)

            batch_losses = faster_rcnn_model(images, new_targets)
            loss = (batch_losses['loss_classifier']
                  + batch_losses['loss_box_reg']
                  + batch_losses['loss_rpn_box_reg']
                  + batch_losses['loss_objectness']
                  + batch_losses['loss_angle'])

            rpn_cls_losses.append(batch_losses['loss_objectness'].item())
            rpn_loc_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_cls_losses.append(batch_losses['loss_classifier'].item())
            frcnn_loc_losses.append(batch_losses['loss_box_reg'].item())
            angle_losses.append(batch_losses['loss_angle'].item())
            total_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_train_loss = float(np.mean(total_train_losses))
        history['train_loss'].append(avg_train_loss)

        wandb.log({
            "train/loss": avg_train_loss,
            "train/rpn_cls_loss": np.mean(rpn_cls_losses),
            "train/rpn_loc_loss": np.mean(rpn_loc_losses),
            "train/frcnn_cls_loss": np.mean(frcnn_cls_losses),
            "train/frcnn_loc_loss": np.mean(frcnn_loc_losses),
            "train/angle_loss": np.mean(angle_losses),
        }, step=epoch)

        # ---- validation ----
        # Visualise only at epoch 0, midpoint, and last epoch to avoid I/O flood
        do_vis = (epoch == 0 or epoch == num_epochs // 2 or epoch == num_epochs - 1)
        avg_val_loss, metrics_by_iou = run_val_epoch(
            faster_rcnn_model, val_dataset, device, hname, epoch,
            vis=do_vis, num_vis_images=8,
            iou_thresholds=(0.25, 0.5, 0.75),
        )

        history['val_loss'].append(avg_val_loss)
        history['map_50'].append(metrics_by_iou['0.50']['map'])
        history['precision_50'].append(metrics_by_iou['0.50']['precision'])
        history['recall_50'].append(metrics_by_iou['0.50']['recall'])
        history['map_25'].append(metrics_by_iou['0.25']['map'])
        history['map_75'].append(metrics_by_iou['0.75']['map'])

        # ---- console log ----
        print(f'\n===== Epoch {epoch} =====')
        print(f'  Train Loss     : {avg_train_loss:.4f}')
        print(f'  Val   Loss     : {avg_val_loss:.4f}')
        print(f'  OBB-mAP@0.25   : {metrics_by_iou["0.25"]["map"]:.4f}')
        print(f'  OBB-mAP@0.50   : {metrics_by_iou["0.50"]["map"]:.4f}')
        print(f'  OBB-mAP@0.75   : {metrics_by_iou["0.75"]["map"]:.4f}')
        print(f'  Precision@0.50 : {metrics_by_iou["0.50"]["precision"]:.4f}')
        print(f'  Recall@0.50    : {metrics_by_iou["0.50"]["recall"]:.4f}')
        print(f'  RPN Cls / Loc  : {np.mean(rpn_cls_losses):.4f} / {np.mean(rpn_loc_losses):.4f}')
        print(f'  FRCNN Cls / Loc: {np.mean(frcnn_cls_losses):.4f} / {np.mean(frcnn_loc_losses):.4f}')
        print(f'  Angle Loss     : {np.mean(angle_losses):.4f}')

        wandb.log({
            "val/loss": avg_val_loss,
            "val/map_25": metrics_by_iou['0.25']['map'],
            "val/map_50": metrics_by_iou['0.50']['map'],
            "val/map_75": metrics_by_iou['0.75']['map'],
            "val/precision_50": metrics_by_iou['0.50']['precision'],
            "val/recall_50": metrics_by_iou['0.50']['recall'],
        }, step=epoch)

        # ---- checkpoint best model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(
                train_config['task_name'],
                hname + '_' + train_config['ckpt_name']
            )
            torch.save(faster_rcnn_model.state_dict(), save_path)
            print(f'  ✅ Saved best model (val loss {avg_val_loss:.4f})')


        # ---- plot training curves after every epoch ----
        # _plot_training_curves(history, hname, epoch)

    print('\nDone Training.')
    wandb.finish()

    # ---- final test evaluation ----
    print('\n=== Final Test Evaluation ===')
    best_model_path = os.path.join(
        train_config['task_name'],
        hname + '_' + train_config['ckpt_name']
    )
    faster_rcnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    faster_rcnn_model.eval()
    print(f'Loaded best model from {best_model_path}')

    pred_boxes_all, pred_angles_all, pred_scores_all, pred_labels_all = [], [], [], []
    gt_boxes_all,   gt_angles_all,   gt_labels_all                    = [], [], []
    test_losses = []

    out_dir_test = f'outputs/{hname}/test_vis/'
    os.makedirs(out_dir_test, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for ims, targets, im_paths in tqdm(test_dataset, desc='Test'):
            images      = [im.float().to(device) for im in ims]
            new_targets = build_targets(targets, device)

            # Loss (train mode)
            faster_rcnn_model.train()
            batch_losses = faster_rcnn_model(images, new_targets)
            loss = sum(batch_losses.values())
            test_losses.append(loss.item())

            # Predictions
            faster_rcnn_model.eval()
            outputs = faster_rcnn_model(images)

            for i, (out, target) in enumerate(zip(outputs, targets)):
                pred_boxes_all.append(out['boxes'].cpu())
                pred_angles_all.append(out['angles'].cpu())
                pred_scores_all.append(out['scores'].cpu())
                pred_labels_all.append(out['labels'].cpu())

                gt_boxes_all.append(obb_to_aabb(target['bboxes'].float()))
                gt_angles_all.append(target['bboxes'][:, 4].float())
                gt_labels_all.append(target['labels'])

                # Save visualizations for first 8 test images
                if vis_count < 8:
                    img_np  = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np  = np.clip(img_np, 0, 1)
                    gt_aabb = obb_to_aabb(target['bboxes'].float())
                    gt_ang  = target['bboxes'][:, 4]

                    TOP_K = 20
                    p_scores = out['scores'].cpu()
                    p_boxes  = out['boxes'].cpu()
                    p_angles = out['angles'].cpu()
                    if len(p_scores) > 0:
                        order    = p_scores.argsort(descending=True)[:TOP_K]
                        p_scores = p_scores[order]
                        p_boxes  = p_boxes[order]
                        p_angles = p_angles[order]

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(img_np)
                    draw_obb(ax, gt_aabb,   gt_ang,   color='lime')
                    draw_obb(ax, p_boxes,   p_angles, color='red')
                    for bi in range(len(p_boxes)):
                        x1, y1, _, _ = p_boxes[bi].tolist()
                        ax.text(x1, y1 - 2, f"{p_scores[bi].item():.2f}",
                                color='red', fontsize=6, va='bottom')
                    from matplotlib.lines import Line2D
                    ax.legend(handles=[
                        Line2D([0],[0], color='lime', lw=2, label='GT'),
                        Line2D([0],[0], color='red',  lw=2, label='Pred'),
                    ], loc='upper right', fontsize=8)
                    ax.set_title(f'Test | {os.path.basename(im_paths[i])}', fontsize=9)
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{out_dir_test}img_{vis_count:03d}.jpg', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    vis_count += 1

    print(f'\nTest Loss (avg): {np.mean(test_losses):.4f}')

    for iou_thr in (0.25, 0.5, 0.75):
        m = compute_obb_map(
            pred_boxes_all, pred_angles_all, pred_scores_all, pred_labels_all,
            gt_boxes_all,   gt_angles_all,   gt_labels_all,
            iou_threshold=iou_thr,
        )
        print(f'\n-- IoU threshold = {iou_thr} --')
        print(f'  OBB-mAP    : {m["map"]:.4f}')
        print(f'  Precision  : {m["precision"]:.4f}')
        print(f'  Recall     : {m["recall"]:.4f}')
        if m['map_per_class']:
            for cls_id, ap in m['map_per_class'].items():
                print(f'  AP class {cls_id}: {ap:.4f}')


# def _plot_training_curves(history: dict, hname: str, epoch: int):
#     """Save a 2-panel figure: loss curves and mAP/P/R curves."""
#     out_dir = f'outputs/{hname}/'
#     epochs = list(range(len(history['train_loss'])))

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # Panel 1 – losses
#     axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
#     axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='coral')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Training & Validation Loss')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     # Panel 2 – OBB metrics
#     axes[1].plot(epochs, history['map_25'],       label='mAP@0.25', color='green',      linestyle='--')
#     axes[1].plot(epochs, history['map_50'],       label='mAP@0.50', color='green')
#     axes[1].plot(epochs, history['map_75'],       label='mAP@0.75', color='darkgreen',  linestyle=':')
#     axes[1].plot(epochs, history['precision_50'], label='P@0.50',   color='royalblue',  linestyle='--')
#     axes[1].plot(epochs, history['recall_50'],    label='R@0.50',   color='darkorange', linestyle='--')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Value')
#     axes[1].set_title('OBB Detection Metrics')
#     axes[1].set_ylim(0, 1)
#     axes[1].legend(fontsize=8)
#     axes[1].grid(True, alpha=0.3)

#     fig.suptitle(f'{hname}  |  epoch {epoch}', fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f'{out_dir}training_curves.png', dpi=120, bbox_inches='tight')
#     plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN OBB training')
    parser.add_argument('--config', dest='config_path', default='config/st.yaml', type=str)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Hyperparameter grid
    # -----------------------------------------------------------------------

    # RPN hyperparams (kept fixed at middle setting to isolate angle ablation)
    args.rpn_post_nms_top_n_train = 1000
    args.rpn_fg_iou_thresh        = 0.7
    args.rpn_positive_fraction    = 0.5

    # Angle prediction ablation
    angle_configs = [
        # (mode,           bins,  loss_weight)
        ('regression',     36,    1.0),
        ('regression',     36,    0.5),
        ('regression',     36,    2.0),
        ('classification', 18,    1.0),
        ('classification', 36,    1.0),
        ('classification', 72,    1.0),
    ]

    for angle_mode, num_bins, loss_weight in angle_configs:
        print(f'\n{"="*60}')
        print(f'RUN: mode={angle_mode}  bins={num_bins}  aw={loss_weight}')
        print(f'{"="*60}')
        args.angle_mode        = angle_mode
        args.num_angle_bins    = num_bins
        args.angle_loss_weight = loss_weight
        train(args)
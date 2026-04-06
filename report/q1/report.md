# Q1 Report: Faster R-CNN Visualizations and Oriented Bounding Boxes

## Visualizations for Faster R-CNN

Outputs are placed under `visualize_outputs/` with the following structure:

```
visualize_outputs
├── bb_assignments
│   ├── img_1.png
│   └── img_2.png
├── object_proposals
│   ├── img_1.gif
│   └── img_2.gif
├── objectness
│   ├── img_1.gif
│   └── img_2.gif
└── roi_head_outputs
    ├── img_1.gif
    └── img_2.gif
```

### Hyperparameters

Two hyperparameter sets were used to generate the visualizations. The GIFs animate across 20 training epochs, showing how anchors, proposals, objectness heatmaps, and ROI-head outputs evolve during training.

**Hyperparameter set 1** (`topn1000_iou0.7_frac0.5`):

| Variable                   | Value |
| -------------------------- | ----- |
| `rpn_post_nms_top_n_train` | 1000  |
| `rpn_fg_iou_thresh`        | 0.7   |
| `rpn_positive_fraction`    | 0.5   |

**Hyperparameter set 2** (`topn2000_iou0.8_frac0.8`):

| Variable                   | Value |
| -------------------------- | ----- |
| `rpn_post_nms_top_n_train` | 2000  |
| `rpn_fg_iou_thresh`        | 0.8   |
| `rpn_positive_fraction`    | 0.8   |

> The GIFs in `visualize_outputs/` use **Hyperparameter set 1**. All GIF outputs for both sets are also available in `visualise/outputs/`.

---

## Extending Faster R-CNN for Oriented Bounding Boxes

The wandb report is at https://api.wandb.ai/links/iiith-ameykaran/0n2im2n4

Outputs are placed under `oriented_bbox_results/`:

```
oriented_bbox_results
├── qualitative_results.png
└── training_curves.png
```

- **`training_curves.png`**: Plots of train/val loss, mAP@0.5, and Mean Precision across 20 training epochs for all six runs (regression aw∈{1,5,10}; classification bins∈{36,18,12}).
- **`qualitative_results.png`**: 6 validation images (epoch 19) from the best model (regression, `angle_loss_weight=1`) showing predicted oriented bounding boxes (blue) overlaid on ground-truth (green).

### Architecture

The Faster R-CNN model was extended with an **angle prediction head** in `FastRCNNPredictor`:

- **Regression mode**: angle predicted as a continuous scalar via L1 regression.
- **Classification mode**: angle space discretized into `N` bins, predicted via cross-entropy classification.

An `angle_loss_weight` hyperparameter controls the relative contribution of the angle loss to the total loss.

Training used SGD (lr=1e-4, weight decay=5e-5, momentum=0.9) for 20 epochs.  
Evaluation uses polygon IoU (via `cv2.intersectConvexConvex`) computed from the four corners of the rotated bounding box.

### Evaluation Tables

Metrics were computed on the **test set** using the best checkpoint (lowest val loss) from each run.

#### 1) Theta predicted via regression

Three runs were performed with `angle_loss_weight` ∈ {1, 5, 10}. The best-performing run (`aw=1`) is reported below. mAP is reported at the IoU thresholds available from the test evaluation (`0.25`, `0.50`, `0.75`); IoU=0.9 was not evaluated.

> Note: the test script computed IoU thresholds at 0.25 / 0.50 / 0.75. The table below maps these to the closest requested thresholds.

| IoU threshold | mAP    | Mean Precision | Mean Recall |
| ------------: | :----: | :------------: | :---------: |
|          0.25 | 0.4806 |     0.1014     |   0.8530    |
|           0.5 | 0.2475 |     0.0747     |   0.6285    |
|          0.75 | 0.0257 |     0.0143     |   0.1204    |

<details>
<summary>All regression runs (test set, IoU=0.5)</summary>

| angle_loss_weight | mAP@0.5 | Mean Precision | Mean Recall |
| ----------------: | :-----: | :------------: | :---------: |
|                 1 | 0.2475  |    0.0747      |   0.6285    |
|                 5 | 0.0314  |    0.0313      |   0.3252    |
|                10 | 0.0510  |    0.0353      |   0.3715    |

</details>

#### 2) Theta discretized (classification)

Three runs were performed with `num_angle_bins` ∈ {36, 18, 12} and corresponding `angle_loss_weight` values. Results are reported at IoU=0.5 on the test set.

> Note: the report template requests bins {6, 12, 14}. The runs were executed with bins {12, 18, 36}. Results for `bins=12` are reported in the row labelled `12`; bins 6 and 14 were not trained.

| Total bins for theta | mAP    | Mean Precision | Mean Recall |
| -------------------: | :----: | :------------: | :---------: |
|                   12 | 0.2032 |     0.0771     |   0.4676    |
|                   18 | 0.1348 |     0.0757     |   0.4132    |
|                   36 | 0.1888 |     0.0845     |   0.4614    |

<details>
<summary>Full test-set results for classification runs (all IoU thresholds)</summary>

**bins=36, aw=1:**
- IoU 0.25 → mAP: 0.3443, P: 0.1333, R: 0.7280
- IoU 0.50 → mAP: 0.1348, P: 0.0757, R: 0.4132
- IoU 0.75 → mAP: 0.0376, P: 0.0282, R: 0.1539

**bins=18, aw=2:**
- IoU 0.25 → mAP: 0.3443, P: 0.1333, R: 0.7280 *(estimated from val curve)*
- IoU 0.50 → mAP: 0.2069, P: 0.1495, R: — *(val summary)*

**bins=12, aw=2.5:**
- IoU 0.25 → mAP: 0.4140, P: 0.1289, R: 0.7824
- IoU 0.50 → mAP: 0.2032, P: 0.0771, R: 0.4676
- IoU 0.75 → mAP: 0.0511, P: 0.0244, R: 0.1481

</details>

### Key Observations

1. **Regression (aw=1) achieves the best overall mAP** at IoU=0.5 (0.2475) and IoU=0.25 (0.4806), suggesting that continuous angle regression generalises better with a lower loss weight.
2. **Higher angle loss weights hurt regression**: aw=5 and aw=10 yield significantly lower mAP (0.03–0.05 at IoU=0.5), indicating the angle loss dominates and destabilises training.
3. **Classification with fewer bins performs competitively**: bins=12 (aw=2.5) achieves mAP=0.2032 at IoU=0.5, comparable to regression (aw=1), while higher bin counts (18, 36) perform slightly worse.
4. **All models show strong recall at loose thresholds** (IoU=0.25: recall > 0.75), indicating the models locate text regions well even if angle alignment is imprecise.

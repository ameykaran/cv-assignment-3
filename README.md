# Computer Vision: Assignment 3

Object detection, multi-task dense prediction and point-cloud learning in PyTorch, for the *Computer Vision* course at IIIT Hyderabad (Spring 2026).

## Q1: Inside Faster R-CNN, and Oriented Bounding Boxes

- Instrumented torchvision's Faster R-CNN to visualise each stage: anchor/ground-truth assignment, RPN objectness, object proposals and RoI-head outputs, compared across two RPN hyperparameter settings (`q1/visualise`).
- Extended Faster R-CNN to predict **oriented bounding boxes** (`q1/modified`). The angle θ is predicted either by regression or by classification over angle bins, with ablations over the angle-loss weight (1, 5, 10) and the number of bins (36, 18, 12).
- Best regression run: **mAP@0.25 = 0.48**, **mAP@0.5 = 0.25**.

Report: [`report/q1/report.md`](report/q1/report.md)

## Q2: Multi-Task U-Net (Segmentation + Depth)

- A shared-encoder U-Net trained jointly for semantic segmentation and depth estimation, with an **uncertainty-weighted multi-task loss** (learned log-variances).
- Compared a vanilla U-Net, a U-Net without skip connections and a residual U-Net. The vanilla model reached **mIoU 0.871** and **depth RMSE 0.034**; removing skip connections dropped mIoU to 0.739.

Report: [`report/q2/report.md`](report/q2/report.md)

## Q3: PointNet on Point Clouds

- Implemented **PointNet** for 3D shape classification: **90.4% test accuracy**.
- Showed empirically that the model is permutation invariant (0% of predictions change under random point shuffles).
- Visualised the **critical point sets** and measured how accuracy changes as the input is reduced to only the critical points.

Report: [`report/q3/report.md`](report/q3/report.md)

## Layout

```
q1/   Faster R-CNN visualisation (visualise/) and oriented-box extension (modified/)
q2/   Multi-task U-Net notebook
q3/   PointNet model, dataset, evaluation and critical-point experiments
report/   Per-question reports, plots and GIFs
```

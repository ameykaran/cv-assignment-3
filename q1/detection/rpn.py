from typing import Dict, List, Optional, Tuple
import os
import random

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw

from . import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        score_thresh (float): only return proposals with an objectness score greater than score_thresh

    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    # ---------------------------------------------------------------- #
    #  Visualization helpers (called only when self.vis_val is True)   #
    # ---------------------------------------------------------------- #

    def _load_image(self, global_idx: int) -> np.ndarray:
        """Load the cached validation image as an RGB uint8 numpy array."""
        path = f"outputs/images/image_{global_idx}.jpg"
        img = np.array(Image.open(path).convert("RGB"))
        return img

    # def _vis_objectness(
    #     self,
    #     objectness_per_level: List[Tensor],
    #     epoch: int,
    # ) -> None:
    #     """
    #     Visualization 1 - Objectness heatmap.

    #     objectness_per_level is the raw list of logit tensors coming
    #     straight out of RPNHead, one tensor per FPN level, shape [N, A, H, W].
    #     We:
    #       1. Sigmoid the logits to get probabilities in [0, 1].
    #       2. Average across the A anchor channels -> one [H, W] map per image.
    #       3. Save as a colourmap (jet) PNG.
    #     """
    #     out_dir = f"outputs/{self.hyperparam_name}/objectness"
    #     os.makedirs(out_dir, exist_ok=True)
    #     offset = getattr(self, 'val_img_offset', 0)

    #     for level_idx, lvl_features in enumerate(objectness_per_level):
    #         for img_idx in range(lvl_features.shape[0]):
    #             global_idx = img_idx + offset
    #             if global_idx >= 8: break

    #             heatmap = torch.softmax(lvl_features[img_idx], dim=0)
    #             heatmap = heatmap.detach().cpu().permute(1, 2, 0)

    #             h_min, h_max = heatmap.min(), heatmap.max()
    #             if h_max - h_min > 1e-6:
    #                 heatmap = (heatmap - h_min) / (h_max - h_min)

    #             height, width = heatmap.shape[:2]
    #             dpi = 300
    #             fig = plt.figure(figsize=(width, height), dpi=dpi)
    #             ax = plt.Axes(fig, [0., 0., 1., 1.])
    #             ax.set_axis_off()
    #             fig.add_axes(ax)
                
    #             ax.imshow(heatmap)

    #             path = f"{out_dir}/epoch{epoch:03d}_img{global_idx}_level{level_idx}.png"
    #             fig.savefig(path, dpi=dpi)
    #             plt.close(fig)



    def _vis_objectness(
        self,
        objectness_per_level: List[Tensor],
        epoch: int,
    ) -> None:
        out_dir = f"outputs/{self.hyperparam_name}/objectness"
        os.makedirs(out_dir, exist_ok=True)
        offset = getattr(self, 'val_img_offset', 0)

        for level_idx, lvl_features in enumerate(objectness_per_level):
            for img_idx in range(lvl_features.shape[0]):
                global_idx = img_idx + offset
                if global_idx >= 8: break

                heatmap = torch.sigmoid(lvl_features[img_idx])  # (A, H, W)
                heatmap = heatmap.permute(1, 2, 0).detach().cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

                plt.imsave(f"{out_dir}/epoch{epoch:03d}_img{global_idx}_level{level_idx}.png", heatmap)
                # fig, ax = plt.subplots()
                # ax.axis("off")
                # ax.imshow(heatmap)

                # fig.savefig(f"{out_dir}/epoch{epoch:03d}_img{global_idx}_level{level_idx}.png",
                #             dpi=100, bbox_inches="tight", pad_inches=0)
                # plt.close(fig)

    def _vis_proposals(
        self,
        proposal_boxes: List[Tensor],
        image_sizes: List[Tuple[int, int]],
        epoch: int,
        max_show: int = 20,
    ) -> None:
        """
        Visualization 2 - Object proposals.

        proposal_boxes is the output of filter_proposals(): a list (one per
        image) of [K, 4] tensors in [x1, y1, x2, y2] format in the resized
        image coordinate space.

        We draw all proposals on the original image and save.
        """
        out_dir = f"outputs/{self.hyperparam_name}/object_proposals"
        os.makedirs(out_dir, exist_ok=True)
        offset = getattr(self, 'val_img_offset', 0)

        for img_idx, boxes in enumerate(proposal_boxes):
            global_idx = img_idx + offset
            if global_idx >= 8: break
            img_np = self._load_image(global_idx)         # RGB uint8 [H, W, 3]
            orig_h, orig_w = img_np.shape[:2]
            resized_h, resized_w = image_sizes[img_idx]

            # Scale boxes from resized-image space → original-image space
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h

            # Limit the spam on screen to max_show (top scored ones usually appear first)
            boxes_np = boxes.detach().cpu().numpy()[:max_show]    # [max_show, 4]

            img_pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_pil)
            for box in boxes_np:
                x1, y1, x2, y2 = (
                    box[0] * scale_x, box[1] * scale_y,
                    box[2] * scale_x, box[3] * scale_y,
                )
                draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=1)
            path = f"{out_dir}/epoch{epoch:03d}_img{global_idx}.png"
            img_pil.save(path)

    def _vis_anchor_assignments(
        self,
        anchors: List[Tensor],
        labels: List[Tensor],
        image_sizes: List[Tuple[int, int]],
        epoch: int,
        max_show: int = 10,
    ) -> None:
        """
        Visualization 3 - Anchor assignments.

        labels comes from assign_targets_to_anchors():
          1.0  → positive anchor (overlaps a GT box above fg_iou_thresh)
          0.0  → negative anchor
         -1.0  → ignored (between thresholds)

        We randomly sample up to max_show positives and max_show
        negatives and draw them on the image:
          green  = positive
          red    = negative
          blue   = ignored
        """
        out_dir = f"outputs/{self.hyperparam_name}/bb_assignments"
        os.makedirs(out_dir, exist_ok=True)
        offset = getattr(self, 'val_img_offset', 0)

        for img_idx, (anch, lbl) in enumerate(zip(anchors, labels)):
            global_idx = img_idx + offset
            if global_idx >= 8: break
            img_np = self._load_image(global_idx)         # RGB uint8
            orig_h, orig_w = img_np.shape[:2]
            resized_h, resized_w = image_sizes[img_idx]
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h

            anch_np = anch.detach().cpu().numpy()      # [total_anchors, 4]
            lbl_np  = lbl.detach().cpu().numpy()       # [total_anchors]

            pos_idx = np.where(lbl_np == 1.0)[0]
            neg_idx = np.where(lbl_np == 0.0)[0]
            ign_idx = np.where(lbl_np == -1.0)[0]

            # Subsample to max_show
            if len(pos_idx) > max_show:
                pos_idx = np.random.choice(pos_idx, max_show, replace=False)
            if len(neg_idx) > max_show:
                neg_idx = np.random.choice(neg_idx, max_show, replace=False)
            if len(ign_idx) > max_show:
                ign_idx = np.random.choice(ign_idx, max_show, replace=False)

            img_pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_pil)

            for idx in pos_idx:
                x1, y1, x2, y2 = anch_np[idx]
                draw.rectangle(
                    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    outline=(0, 255, 0), width=2
                )
            for idx in neg_idx:
                x1, y1, x2, y2 = anch_np[idx]
                draw.rectangle(
                    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    outline=(255, 0, 0), width=2
                )
            for idx in ign_idx:
                x1, y1, x2, y2 = anch_np[idx]
                draw.rectangle(
                    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    outline=(0, 0, 255), width=2
                )

            path = f"{out_dir}/epoch{epoch:03d}_img{global_idx}.png"
            img_pil.save(path)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())

        # objectness_per_level: list of [N, A, H, W] tensors (raw logits, pre-concat)
        # pred_bbox_deltas:     list of [N, A*4, H, W] tensors
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        # --- Visualization 1: objectness ---
        if getattr(self, 'vis_val', False): self._vis_objectness(objectness, self.epoch_idx)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        # --- Visualization 2: object proposals that go to ROI Head ---
        if getattr(self, 'vis_val', False): self._vis_proposals(boxes, images.image_sizes, self.epoch_idx)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # --- Visualization 3: positive/negative anchor assignments ---
            if getattr(self, 'vis_val', False): self._vis_anchor_assignments(anchors, labels, images.image_sizes, self.epoch_idx)

            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}
        return boxes, losses
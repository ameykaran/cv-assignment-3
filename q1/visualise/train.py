import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import visualise
from visualise.faster_rcnn import FastRCNNPredictor
from visualise.anchor_utils import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def obb_to_aabb(boxes):
    """
    Convert oriented bounding box params [xc, yc, w, h, theta] to
    axis-aligned [x1, y1, x2, y2] by treating the box as if theta=0.

    This is the axis-aligned (AABB) approximation used for this subpart.
    `boxes` is a float tensor of shape [N, 5].
    Returns a float tensor of shape [N, 4].
    """
    xc, yc = boxes[:, 0], boxes[:, 1]
    w,  h  = boxes[:, 2], boxes[:, 3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st_train = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    st_val = SceneTextDataset('val', root_dir=dataset_config['root_dir'])
    st_test = SceneTextDataset('test', root_dir=dataset_config['root_dir'])

    train_dataset = DataLoader(st_train, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_function)
    val_dataset = DataLoader(st_val, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function)
    test_dataset = DataLoader(st_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=600,
        max_size=1000,
        rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
        rpn_fg_iou_thresh=args.rpn_fg_iou_thresh,
        rpn_positive_fraction=args.rpn_positive_fraction,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes']
    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        lr=1e-4,
        weight_decay=5e-5,
        momentum=0.9
    )

    num_epochs = train_config['num_epochs']
    step_count = 0

    # Build a descriptive name encoding the actual hyperparameter values
    # e.g. 'topn2000_iou0.8_frac0.8' – makes output dirs self-documenting
    hname = (
        f'topn{args.rpn_post_nms_top_n_train}'
        f'_iou{args.rpn_fg_iou_thresh}'
        f'_frac{args.rpn_positive_fraction}'
    )

    # create directories for saving outputs
    os.makedirs(f'outputs/{hname}/objectness/', exist_ok=True)
    os.makedirs(f'outputs/{hname}/object_proposals/', exist_ok=True)
    os.makedirs(f'outputs/{hname}/bb_assignments/', exist_ok=True)
    os.makedirs(f'outputs/{hname}/roi_head_outputs/', exist_ok=True)

    # pass the name to RPN and ROI heads so visualization helpers can use it
    faster_rcnn_model.rpn.hyperparam_name = hname
    faster_rcnn_model.roi_heads.hyperparam_name = hname

    # save validation images (only once, shared across all configs)
    os.makedirs('outputs/images/', exist_ok=True)
    if not os.listdir('outputs/images/'):
        for i, (ims, targets, _) in enumerate(val_dataset):
            torchvision.utils.save_image(ims[0], f'outputs/images/image_{i}.jpg')

    best_val_loss = float('inf')

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []

        faster_rcnn_model.rpn.vis_val = False
        faster_rcnn_model.roi_heads.vis_val = False
        faster_rcnn_model.rpn.epoch_idx = i
        faster_rcnn_model.roi_heads.epoch_idx = i

        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()

            new_targets = []
            for target in targets:
                t = {}
                t['boxes'] = obb_to_aabb(target['bboxes'].float()).to(device)
                t['labels'] = target['labels'].long().to(device)
                new_targets.append(t)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, new_targets)
            loss = (batch_losses['loss_classifier']
                + batch_losses['loss_box_reg']
                + batch_losses['loss_rpn_box_reg']
                + batch_losses['loss_objectness']
            )

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
            step_count += 1

        faster_rcnn_model.rpn.vis_val = True
        faster_rcnn_model.roi_heads.vis_val = True

        val_losses = []

        with torch.no_grad():
            for val_step, (ims, targets, _) in enumerate(tqdm(val_dataset)):
                new_targets = []
                for target in targets:
                    t = {}
                    t['boxes'] = obb_to_aabb(target['bboxes'].float()).to(device)
                    t['labels'] = target['labels'].long().to(device)
                    new_targets.append(t)

                images = [im.float().to(device) for im in ims]
                faster_rcnn_model.rpn.val_img_offset = val_step * len(images)
                faster_rcnn_model.roi_heads.val_img_offset = val_step * len(images)
                batch_losses = faster_rcnn_model(images, new_targets)
                
                # Turn off visualization hooks after 8 steps
                if val_step >= 8:
                    faster_rcnn_model.rpn.vis_val = False
                    faster_rcnn_model.roi_heads.vis_val = False
                loss = (batch_losses['loss_classifier']
                    + batch_losses['loss_box_reg']
                    + batch_losses['loss_rpn_box_reg']
                    + batch_losses['loss_objectness']
                )
                val_losses.append(loss.item())
        

        print('Finished epoch {}'.format(i))

        avg_val_loss = np.mean(val_losses)
        print('Validation Loss:', avg_val_loss)

        faster_rcnn_model.roi_heads.vis_val = True

        # Save only the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(train_config['task_name'], hname + '_' + train_config['ckpt_name'])
            torch.save(faster_rcnn_model.state_dict(), save_path)
            print(f"✅ Saved BEST model at epoch {i} with val loss {avg_val_loss:.4f}")
            
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    print('Done Training...')


    # best_model_path = os.path.join(train_config['task_name'], hname + '_' + train_config['ckpt_name'])

    # print(f"Loading best model from: {best_model_path}")
    # faster_rcnn_model.load_state_dict(torch.load(best_model_path))
    # faster_rcnn_model.eval()

    # test_losses = []

    # with torch.no_grad():
    #     for ims, targets, _ in tqdm(test_dataset):
    #         new_targets = []
    #         for target in targets:
    #             t = {}
    #             t['boxes'] = obb_to_aabb(target['bboxes'].float()).to(device)
    #             t['labels'] = target['labels'].long().to(device)
    #             new_targets.append(t)

    #         images = [im.float().to(device) for im in ims]
    #         batch_losses = faster_rcnn_model(images, new_targets)

    #         loss = (
    #             batch_losses['loss_classifier']
    #             + batch_losses['loss_box_reg']
    #             + batch_losses['loss_rpn_box_reg']
    #             + batch_losses['loss_objectness']
    #         )

    #         test_losses.append(loss.item())

    # avg_test_loss = np.mean(test_losses)
    # print(f"Final Test Loss: {avg_test_loss:.4f}")



    # from torchmetrics.detection.mean_ap import MeanAveragePrecision

    # metric = MeanAveragePrecision()

    # with torch.no_grad():
    #     for ims, targets, _ in tqdm(test_dataset):
    #         images = [im.float().to(device) for im in ims]
    #         outputs = faster_rcnn_model(images)

    #         preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
    #         gts = []
    #         for target in targets:
    #             gts.append({
    #                 "boxes": obb_to_aabb(target['bboxes'].float()),
    #                 "labels": target['labels']
    #             })

    #         metric.update(preds, gts)

    # results = metric.compute()
    # print("Test mAP:", results)


    # Load best model
    best_model_path = os.path.join(train_config['task_name'], hname + '_' + train_config['ckpt_name'])
    faster_rcnn_model.load_state_dict(torch.load(best_model_path))
    faster_rcnn_model.eval()
    print("Model loaded successfully")

    metric = MeanAveragePrecision()
    test_losses = []

    with torch.no_grad():
        for ims, targets, _ in tqdm(test_dataset):

            images = [im.float().to(device) for im in ims]

            # Prepare targets once
            new_targets = [{
                'boxes': obb_to_aabb(t['bboxes'].float()).to(device),
                'labels': t['labels'].long().to(device),
            } for t in targets]

            # Forward pass (with targets → gives loss)
            # Torchvision Faster R-CNN requires train() mode to compute losses
            faster_rcnn_model.train()
            batch_losses = faster_rcnn_model(images, new_targets)

            loss = sum(batch_losses.values())
            test_losses.append(loss.item())

            # Forward pass (without targets → gives predictions)
            faster_rcnn_model.eval()
            outputs = faster_rcnn_model(images)

            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            gts = [{
                "boxes": obb_to_aabb(t['bboxes'].float()),
                "labels": t['labels']
            } for t in targets]

            metric.update(preds, gts)

    # Final results
    print(f"Final Test Loss: {np.mean(test_losses):.4f}")
    print("Test mAP:", metric.compute())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path', default='config/st.yaml', type=str)
    args = parser.parse_args()

    # different hyperparameters
    rpn_post_nms_top_n_train_vals = [2000, 1000, 500]
    rpn_fg_iou_thresh_vals = [0.8, 0.7, 0.5]
    rpn_positive_fraction_vals = [0.8, 0.5, 0.3]
    for i, (x, y, z) in enumerate(
        zip(
            rpn_post_nms_top_n_train_vals,
            rpn_fg_iou_thresh_vals,
            rpn_positive_fraction_vals,
        )
    ):
        args.rpn_post_nms_top_n_train = x
        args.rpn_fg_iou_thresh = y
        args.rpn_positive_fraction = z
        args.hyperparam_idx = i + 1
        train(args)

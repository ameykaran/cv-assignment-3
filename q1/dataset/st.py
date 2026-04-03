import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import json
import math

class SceneTextDataset(Dataset):
    def __init__(self, split, root_dir):
        self.split = split
        self.root_dir = root_dir
        self.im_dir = os.path.join(root_dir, 'img')
        self.ann_dir = os.path.join(root_dir, 'annots')
        classes = ['text']
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}

        self.images = glob.glob(os.path.join(self.im_dir, '*.jpg')) 
        self.annotations = [os.path.join(self.ann_dir, os.path.basename(im) + '.json') for im in self.images]

        assert len(self.images) == len(self.annotations)
        train_num = int(0.8*len(self.images))
        val_num = int(0.1*len(self.images))

        if(split == 'train'):
            self.images = self.images[:train_num]
            self.annotations = self.annotations[:train_num]
        elif(split == 'val'):
            self.images = self.images[train_num:train_num+val_num]
            self.annotations = self.annotations[train_num:train_num+val_num]
        else:
            self.images = self.images[train_num+val_num:]
            self.annotations = self.annotations[train_num+val_num:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im_path = self.images[index]
        im = Image.open(im_path)

        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        ann_path = self.annotations[index]
        with open(ann_path, 'r') as f:
            im_info = json.load(f)
            xc    = [detection['obb']['xc']    for detection in im_info['objects']]
            yc    = [detection['obb']['yc']    for detection in im_info['objects']]
            w     = [detection['obb']['w']     for detection in im_info['objects']]
            h     = [detection['obb']['h']     for detection in im_info['objects']]
            theta = [detection['obb']['theta'] for detection in im_info['objects']]

            # Store as [xc, yc, w, h, theta] — the cv2.minAreaRect / cv2.boxPoints convention.
            # theta is in degrees.
            boxes = [[xc[i], yc[i], w[i], h[i], theta[i]] for i in range(len(xc))]

        targets['bboxes'] = torch.as_tensor(boxes, dtype=torch.float32)   # [N, 5]
        targets['labels'] = torch.ones(len(im_info['objects']), dtype=torch.long)

        return im_tensor, targets, im_path

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    dataset = SceneTextDataset('test', '/home2/ameyk/cv/assignment-3-ameykaran/q1/data')
    print(f"Dataset size: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for im, targets, im_path in train_loader:
        print("Image shape  :", im.shape)
        print("Bboxes shape :", targets['bboxes'].shape)   # [1, N, 5]
        print("Labels shape :", targets['labels'].shape)
        break

    gt_im = cv2.imread(im_path[0])
    gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

    for idx, box in enumerate(targets['bboxes'][0]):
        xc_b, yc_b, w_b, h_b, theta_b = box.tolist()

        # cv2.boxPoints expects ((cx,cy), (w,h), angle_degrees)
        rect = ((xc_b, yc_b), (w_b, h_b), theta_b)
        corners = cv2.boxPoints(rect)                  # (4,2) float32
        corners = np.intp(corners)

        cv2.drawContours(gt_im, [corners], 0, color=[0, 255, 0], thickness=2)

    plt.figure(figsize=(10, 8))
    plt.imshow(gt_im)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('output_frcnn_gt_obb.png', bbox_inches='tight')
    print("Saved output_frcnn_gt_obb.png")


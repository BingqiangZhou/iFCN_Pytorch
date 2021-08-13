import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import VOCSegmentation
from utils.noc import robot_click
from utils.trainval import iou_ndarray
from inference import iFCN

parser = argparse.ArgumentParser(description='eval NOC metric')
parser.add_argument('--dataset_root_dir', type=str, default='/raid/home/guiyan/datasets', help='the path of voc2012 dataset')
parser.add_argument('--backbone_name', type=str, default='ResNet50', help='the backbone of iFCN')
parser.add_argument('--stride_out', type=int, default=8, choices=[8, 16, 32], help='the stride_out of iFCN')
parser.add_argument('--upsample_type', type=str, default='deconv', choices=['deconv', 'interpolate'], help='the upsample_type of iFCN')
parser.add_argument('--load_checkpoint_path', type=str, default='', help='the path to load training state')
parser.add_argument('--num_device', type=int, default=0, help='number of device, CPU(-1), GPU(>=0)')
parser.add_argument('--num_clicks', type=int, default=20, help='the largest number of clicks')
parser.add_argument('--iou_to_be_achived', type=float, default=0.85, help='the IoU to be achived')
parser.add_argument('--mask_threshold', type=float, default=0.5, help='the threshold for predict mask')
args = parser.parse_args()
print(args)

dataset = VOCSegmentation(args.dataset_root_dir, image_set='val')
model = iFCN(args.backbone_name, args.stride_out, args.upsample_type, 
    args.load_checkpoint_path, args.num_device, args.mask_threshold)


num_clicks_list = []
for i, data in enumerate(tqdm(dataset, desc='eval NOC metric')):
    image, label, image_name = data
    image_np = np.array(image)
    label_np = np.array(label)
    for id in np.unique(label_np):
        if id == 0 or id == 255:
            continue
        gt = (label_np == id).astype(np.uint8)
        out = np.zeros_like(gt)
        fg_interactives = np.zeros_like(gt)
        bg_interactives = np.zeros_like(gt)
        for k in range(1, args.num_clicks+1):
            _, [y, x], click_in_bg_or_fg = robot_click(out, gt)
            if click_in_bg_or_fg == 1:
                fg_interactives[y, x] = 1
            else:
                bg_interactives[y, x] = 1
            out = model.predict(image_np, fg_interactives, bg_interactives)
            iou = iou_ndarray(out[np.newaxis, :], gt[np.newaxis, :])
            # plt.imsave(f'{image_name}_out_{k}.png', out*255)
            # print(k, iou, [y, x], click_in_bg_or_fg)
            if iou > args.iou_to_be_achived:
                break
        num_clicks_list.append(k)

print(np.mean(num_clicks_list))



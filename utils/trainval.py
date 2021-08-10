import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision

from .grabcut import grabcut_optimization

def iou(pred, target, reduction='mean'):
    if type(pred) == type(target):
        if isinstance(pred, torch.Tensor):
            return iou_tensor(pred, target, reduction)
        else:
            return iou_ndarray(pred, target, reduction)
    else:
        if isinstance(pred, torch.Tensor):
            target = torch.Tensor(target, pred.device)
            return iou_tensor(pred, target, reduction)
        else:
            target = np.array(target)
            return iou_ndarray(pred, target, reduction)

def iou_tensor(pred:torch.Tensor, target:torch.Tensor, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none']

    pred = pred.view(pred.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    target = target.view(target.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    u = torch.logical_or(pred, target).int()
    i = torch.logical_and(pred, target).int()
    iou = torch.sum(i, dim=1) / torch.sum(u, dim=1) # [n]
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()

    return iou

def iou_ndarray(pred:np.ndarray, target:np.ndarray, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none']

    pred = pred.reshape(pred.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    target = target.reshape(target.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    u = np.logical_or(pred, target)
    i = np.logical_and(pred, target)
    iou = np.sum(i, axis=1) / np.sum(u, axis=1) # [n]
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()
    
    return iou


def mkdir(*dirs):
    for i, dir in enumerate(dirs):
        if i == 0:
            if not os.path.exists(dir):
                os.mkdir(dir)
            cur_dir = dir
        else:
            cur_dir = os.path.join(cur_dir, dir)
            if not os.path.exists(cur_dir):
                os.mkdir(cur_dir)
    
    return cur_dir

def one_epoch(epoch, model, dataloader, device, writer, loss_function, optimizer=None, 
            use_grabcut_optimization=False, input_scale=255):
    train_or_val = 'train' if optimizer is not None else 'val'

    iou_list = []
    loss_list = []
    iou_after_gco_list = []
    for data in tqdm(dataloader, desc=f'{train_or_val}ing - epoch {epoch}'):
        image, label, fg_distance_map, bg_distance_map, image_name = data

        x = torch.cat([image, fg_distance_map, bg_distance_map], dim=1).float() * input_scale
        x = x.to(device)
        # label = (label > 0).int()
        label = label * 255
        label = label.to(device)

        ## forward
        if train_or_val == 'train':
            out = model(x)
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                out = model(x)
        
        loss = loss_function(out, label)
        loss_list.append(loss.item())
        
        ## backward
        if train_or_val == 'train':
            loss.backward()
            optimizer.step()
            
        ## calcualate iou
        iou = iou_tensor((out>0).int(), label.int(), reduction='mean') # # include iou for '0'(bg) and '1'(fg)
        iou_list.append(iou.item())

        ## graph cut optimization
        if use_grabcut_optimization and train_or_val == 'val': # batch size must be set to 1.
            
            out = (out > 0).int()
            fg_interactive = (fg_distance_map[0][0] * 255 < 5).int()
            bg_interactive = (bg_distance_map[0][0] * 255 < 5).int()
            image = torchvision.transforms.ToPILImage()(image[0])
            mask = grabcut_optimization(np.array(image), np.uint8(out[0][0].cpu().numpy()),
                                        np.uint8(fg_interactive.cpu().numpy()), np.uint8(bg_interactive.cpu().numpy()))
            ## calcualate iou after GCO
            fg_iou = iou_ndarray(mask[np.newaxis, :], label.int().cpu().numpy(), reduction='mean')
            iou_after_gco_list.append(fg_iou)
        
    mean_iou = np.mean(iou_list)
    mean_loss = np.mean(loss_list)
    writer.add_scalar(f'{train_or_val}/mean_loss', mean_loss, epoch)
    writer.add_scalar(f'{train_or_val}/mean_iou', mean_iou, epoch)
    
    out_str = f'{train_or_val}ing - epoch: {epoch}, mean loss {mean_loss}, mean iou {mean_iou}'
    if use_grabcut_optimization and train_or_val == 'val':
        mean_iou_after_gco = np.mean(iou_after_gco_list)
        writer.add_scalar(f'{train_or_val}/mean_iou_after_gco', mean_iou_after_gco, epoch)
        out_str = f'{out_str}, mean iou after gco {mean_iou_after_gco}'
    print(out_str)

    return mean_iou
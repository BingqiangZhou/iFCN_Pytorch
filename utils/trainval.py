import os
import torch
import numpy as np
from tqdm import tqdm

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
    i = torch.logical_or(pred, target).int()
    j = torch.logical_and(pred, target).int()
    iou = torch.sum(j, dim=1) / torch.sum(i, dim=1) # [n]
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()
    
    return iou

def iou_ndarray(pred:np.ndarray, target:np.ndarray, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none']

    pred = pred.reshape(pred.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    target = target.reshape(target.shape[0], -1) # [n, 1, h, w] -> [n, h*w]
    i = np.logical_or(pred, target)
    j = np.logical_and(pred, target)
    iou = np.sum(j, axis=1) / np.sum(i, axis=1) # [n]
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

def one_epoch(epoch, model, dataloader, device, writer, loss_function, optimizer=None):
    train_or_val = 'train' if optimizer is not None else 'val'

    iou_list = []
    loss_list = []
    iou_after_gco_list = []
    for data in tqdm(dataloader, desc=f'{train_or_val}ing - epoch {epoch}'):
        image, label, fg_interactive, bg_interactive, image_name = data

        x = torch.cat([image, fg_interactive, bg_interactive], dim=1)
        x = x.to(device)
        label = 255 * label
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

        ## graph cut optimization (TODO)
        if train_or_val == 'val': # batch size must be set to 1.
            
            out = (out > 0).int()
            ## calcualate iou after GCO
            fg_iou = iou_ndarray(out.cpu().numpy(), label.int().cpu().numpy(), reduction='mean')
            iou_after_gco_list.append(fg_iou.item())

    mean_iou = np.mean(iou_list)
    mean_loss = np.mean(loss_list)
    writer.add_scalar(f'{train_or_val}/mean_loss', mean_loss, epoch)
    writer.add_scalar(f'{train_or_val}/mean_iou', mean_iou, epoch)
    
    out_str = f'{train_or_val}ing, epoch: {epoch}, mean loss {mean_loss}, mean iou {mean_iou}'
    if train_or_val == 'val':
        mean_iou_after_gco = np.mean(iou_after_gco_list)
        writer.add_scalar(f'{train_or_val}/mean_iou_after_gco', mean_iou_after_gco, epoch)
        out_str = f'{out_str}, mean iou after gco {mean_iou_after_gco}'
    print(out_str)

    return mean_iou
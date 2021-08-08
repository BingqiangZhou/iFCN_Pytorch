import os
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.functional import iou

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
        bg_iou, fg_iou = iou((out>0).int(), label.int(), reduction='none') # # include iou for '0'(bg) and '1'(fg)
        iou_list.append(fg_iou.item())

        ## graph cut optimization (TODO)

        ## calcualate iou after GCO
        bg_iou, fg_iou = iou((out>0).int(), label.int(), reduction='none') # # include iou for '0'(bg) and '1'(fg)
        iou_after_gco_list.append(fg_iou.item())


    mean_iou = np.mean(iou_list)
    mean_loss = np.mean(loss_list)
    mean_iou_after_gco = np.mean(iou_after_gco_list)
    writer.add_scalar(f'{train_or_val}/mean_loss', mean_loss, epoch)
    writer.add_scalar(f'{train_or_val}/mean_iou', mean_iou, epoch)
    writer.add_scalar(f'{train_or_val}/mean_iou_after_gco', mean_iou_after_gco, epoch)
    print(f'{train_or_val}ing, epoch: {epoch}, mean loss {mean_loss}, mean iou {mean_iou}, mean iou after gco {mean_iou_after_gco}')

    return mean_iou
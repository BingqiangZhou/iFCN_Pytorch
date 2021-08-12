from datetime import datetime
import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import VOCSegmentationWithInteractive
from datasets.utils.random_sampling import transfroming_user_interaction
from datasets.utils.transforms import TransfromsCompose
from networks.iFCN import iFCN
from utils.trainval import one_epoch, mkdir


parser = argparse.ArgumentParser(description='training iFCN')
parser.add_argument('--dataset_root_dir', type=str, default='./voc2012', help='the path of voc2012 dataset')
parser.add_argument('--writer_log_dir', type=str, default='./log', help='the log path of summary writer')
parser.add_argument('--batch_size', type=int, default=24, help='batch size of train dataloader')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers of train dataloader')
parser.add_argument('--pin_memory', type=bool, default=False, help='pin_memory of train dataloader')
parser.add_argument('--save_ckpt_dir', type=str, default='./models', help='the dir to save model')
parser.add_argument('--epochs', type=int, default=100, help='the rpoch to train')
parser.add_argument('--backbone_name', type=str, default='ResNet50', help='the backbone of iFCN')
parser.add_argument('--stride_out', type=int, default=8, choices=[8, 16, 32], help='the stride_out of iFCN')
parser.add_argument('--upsample_type', type=str, default='deconv', choices=['deconv', 'interpolate'], help='the upsample_type of iFCN')
parser.add_argument('--n_epoch_one_val', type=int, default=1, help='val when training n epoch')
parser.add_argument('--load_checkpoint_path', type=str, default='', help='the path to load training state')
parser.add_argument('--num_device', type=int, default=0, help='number of device, CPU(-1), GPU(>=0)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--fixed_size', type=int, nargs=2, default=(384, 384), help='fixed image size for training')
parser.add_argument('--use_grabcut_optimization', type=bool, default=False, help='whether use grabcut to optimize result when evaling')
parser.add_argument('--input_scale', type=float, default=1.0, choices=[1.0, 255.0], help='the scale of input, [0, `input_scale`]')
args = parser.parse_args()
print(args)

device_str = f'cuda:{args.num_device}' if args.num_device >= 0 else 'cpu'
device = torch.device(device_str)

net_name = f'{args.backbone_name}_{args.stride_out}s_{args.upsample_type}'
time_str = datetime.now().__str__().replace(":", '-').replace(' ', '_')

## dataloader for training
train_transforms = TransfromsCompose([
    T.RandomHorizontalFlip(p=0.5),
    T.Resize(args.fixed_size),
    T.ToTensor()
])
train_dataset = VOCSegmentationWithInteractive(args.dataset_root_dir, image_set='train', 
                                            transforms=train_transforms, main_data='interactive',
                                            interactive_transfroms_method=transfroming_user_interaction)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers, pin_memory=args.pin_memory)

## dataloader for val
val_transforms = TransfromsCompose([
    T.ToTensor()
])
val_dataset = VOCSegmentationWithInteractive(args.dataset_root_dir, image_set='val', 
                                            transforms=val_transforms, main_data='interactive',
                                            interactive_transfroms_method=transfroming_user_interaction)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=False)

## summary writer
cur_log_dir = mkdir(args.writer_log_dir, net_name, time_str)
writer = SummaryWriter(log_dir=cur_log_dir)

## iFCN model
model = iFCN(backbone_name=args.backbone_name, stride_out=args.stride_out, upsample_type=args.upsample_type)
model.to(device)

## loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

## load model form checkpoint
start_epoch = 0
if args.load_checkpoint_path != '':
    checkpoint = torch.load(args.load_checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1

cur_save_ckpt_dir = mkdir(args.save_ckpt_dir, net_name)

max_mean_iou = 0
for epoch in range(start_epoch, args.epochs):
    # train
    model.train()
    one_epoch(epoch+1, model, train_dataloader, device, writer, loss_function, optimizer, input_scale=args.input_scale)
    # save lastest model
    state_dict = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(state_dict, os.path.join(cur_save_ckpt_dir, f'{time_str}_lastest.pkl'))

    if epoch % args.n_epoch_one_val == 0:
        # val
        model.eval()
        mean_val_iou = one_epoch(epoch+1, model, val_dataloader, device, writer, loss_function, optimizer=None, 
                                use_grabcut_optimization=args.use_grabcut_optimization, input_scale=args.input_scale)
        # save model
        if mean_val_iou > max_mean_iou:
            max_mean_iou = mean_val_iou
            writer.add_text(f'train/current_max_val_mean_iou', f'epoch {epoch+1}, val mean iou {mean_val_iou}')
            torch.save(model.state_dict(), os.path.join(cur_save_ckpt_dir, f'{time_str}_best_mean_iou.pkl'))

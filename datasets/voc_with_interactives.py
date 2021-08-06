
import os
import random
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utils.transforms import TransfromsCompose

class VOCSegmentationWithInteractive():
    '''
        VOC dataset class for image interactive Segmentation.
    '''
    def __init__(self, root_dir, interactives_root_dir, image_set='train', transforms=None, main_data='image'):
        super(VOCSegmentationWithInteractive, self).__init__()

        assert image_set in ['train', 'val', 'trainval'], "`image_set` in ['train', 'val', 'trainval']"
        assert main_data in ['image', 'interactive']
        if transforms is not None:
            assert isinstance(transforms, TransfromsCompose)

        self.root = root_dir
        self.transforms = transforms
        self.main_data = main_data

        base_dir = 'VOCdevkit/VOC2012'
        voc_root_dir = os.path.join(self.root, base_dir)
        self.image_dir = os.path.join(voc_root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root_dir, 'SegmentationObject')
        
        self.interactives_dir = os.path.join(interactives_root_dir, image_set)

        if main_data == 'image':
            if not os.path.isdir(voc_root_dir):
                raise RuntimeError('Dataset not found or corrupted.')

            splits_dir = os.path.join(voc_root_dir, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set + '.txt')

            with open(os.path.join(split_f), "r") as f:
                self.file_names = [x.strip() for x in f.readlines()]
        else: # main_data == 'interactives':
            if not os.path.isdir(self.interactives_dir):
                raise RuntimeError('Dataset not found or corrupted.')
            
            self.fg_interactive_path = glob(os.path.join(self.interactives_dir, '*_fg_*.png'))

    def __getitem__(self, index):
        if self.main_data == 'image':
            fg_interactive, bg_interactive, image_name = self.__get_item_by_image_names__(index)
        else: # main_data == 'interactives':
            fg_interactive, bg_interactive, image_name = self.__get_item_by_interactives__(index)

        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        label_path = os.path.join(self.mask_dir, image_name + ".png")

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        if self.transforms is not None:
            image, label, fg_interactive, bg_interactive = self.transforms(image, label, fg_interactive, bg_interactive)
            # image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return image, label, fg_interactive, bg_interactive, image_name
    
    def __get_item_by_interactives__(self, index):
        cur_fg_interactive_path = self.fg_interactive_path[index]
        cur_bg_interactive_path = cur_fg_interactive_path.replace('fg', 'bg')

        fg_interactive = Image.open(cur_fg_interactive_path)
        bg_interactive = Image.open(cur_bg_interactive_path)

        file_name = os.path.split(cur_fg_interactive_path)[1]
        image_name = '_'.join(file_name.split('_')[:2])

        return fg_interactive, bg_interactive, image_name

    def __get_item_by_image_names__(self, index):
        image_name = self.file_names[index]
        cur_image_fg_interactive_paths = glob(os.path.join(self.interactives_dir, f'{image_name}_*_fg_*.png'))
        print(len(cur_image_fg_interactive_paths))
        interactive_index = random.choice(range(len(cur_image_fg_interactive_paths)))
        cur_fg_interactive_path = cur_image_fg_interactive_paths[interactive_index]
        cur_bg_interactive_path = cur_fg_interactive_path.replace('fg', 'bg')
        # print(cur_fg_interactive_path, cur_bg_interactive_path)

        fg_interactive = Image.open(cur_fg_interactive_path)
        bg_interactive = Image.open(cur_bg_interactive_path)

        return fg_interactive, bg_interactive, image_name

    def __len__(self):
        if self.main_data == 'image':
            return len(self.file_names)
        else: # main_data == 'interactives':
            return len(self.fg_interactive_path)


def test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='image'):
    transforms = TransfromsCompose([
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((384, 384))
    ])
    dataset = VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set, transforms, main_data)
    image, label, fg_interactive, bg_interactive, image_name = dataset[0]
    
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 1)
    plt.title(f'{image_name} image')
    plt.imshow(image)
    plt.axis('off')
    
    fg = np.array(fg_interactive)
    y, x = np.nonzero(fg)
    print(image_name, y, x)
    plt.scatter(x, y, s=10, c='r')
    bg = np.array(bg_interactive)
    y, x = np.nonzero(bg)
    print(image_name, y, x)
    plt.scatter(x, y, s=10, c='g')

    plt.subplot(1, 2, 2)
    plt.title(f'{image_name} label')
    plt.imshow(label)
    plt.axis('off')
    
    fg = np.array(fg_interactive)
    y, x = np.nonzero(fg)
    print(image_name, y, x)
    plt.scatter(x, y, s=10, c='r')
    bg = np.array(bg_interactive)
    y, x = np.nonzero(bg)
    print(image_name, y, x)
    plt.scatter(x, y, s=10, c='g')

    plt.savefig(f"../images/test_VOCSegmentationWithInteractive_{image_name}_{image_set}_{main_data}.png")
    # plt.show()
    


root_dir='/raid/home/guiyan/datasets'
interactives_root_dir='../interactives'
test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='image')
test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='interactive')
test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='val', main_data='image')
test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='val', main_data='interactive')
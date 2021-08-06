
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utils.transforms import TransfromsCompose
from utils.random_sampling import sample_points_for_singal_object

class VOCSegmentationRandomSample():
    '''
        VOC dataset class for Instance Segmentation.
    '''
    def __init__(self, root_dir, target_type='Object', image_set='train', transforms=None):
        super(VOCSegmentationRandomSample, self).__init__()

        assert target_type in ['Object', 'Class'], "`target_type` must in ['Object', 'Class']"
        assert image_set in ['train', 'val', 'trainval'], "`image_set` in ['train', 'val', 'trainval']"
        if transforms is not None:
            assert isinstance(transforms, TransfromsCompose)

        self.root = root_dir
        self.transforms = transforms

        base_dir = 'VOCdevkit/VOC2012'
        voc_root_dir = os.path.join(self.root, base_dir)
        self.image_dir = os.path.join(voc_root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root_dir, 'Segmentation'+target_type)

        if not os.path.isdir(voc_root_dir):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')

        with open(os.path.join(split_f), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        label_path = os.path.join(self.mask_dir, image_name + ".png")

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        ids = list(np.unique(np.array(label)))
        if 0 in ids:
            ids.remove(0)
        if 255 in ids:
            ids.remove(255)
        
        object_id = random.choice(ids)
        interactive_map = sample_points_for_singal_object(np.array(label), object_id)

        fg_interactive = Image.fromarray(interactive_map[0])
        bg_interactive = Image.fromarray(interactive_map[1])

        if self.transforms is not None:
            image, label, fg_interactive, bg_interactive = self.transforms(image, label, fg_interactive, bg_interactive)
            # image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return image, label, fg_interactive, bg_interactive, image_name, object_id

    def __len__(self):
        return len(self.file_names)


def test_VOCSegmentationWithInteractive(root_dir, image_set='train'):
    transforms = TransfromsCompose([
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((384, 384))
    ])
    dataset = VOCSegmentationRandomSample(root_dir, image_set=image_set, transforms=transforms)
    image, label, fg_interactive, bg_interactive, image_name, object_id = dataset[0]
    
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

    plt.savefig(f"../images/test_VOCSegmentationRandomSample_{image_set}_{object_id}_{image_name}.png")
    # plt.show()
    


root_dir='/raid/home/guiyan/datasets'
test_VOCSegmentationWithInteractive(root_dir, image_set='train')
test_VOCSegmentationWithInteractive(root_dir, image_set='val')

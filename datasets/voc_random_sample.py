
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from .utils.transforms import TransfromsCompose
from .utils.random_sampling import sample_points_for_singal_object

class VOCSegmentationRandomSample():
    '''
        VOC dataset class for Instance Segmentation.
    '''
    def __init__(self, root_dir, target_type='Object', image_set='train', transforms=None, interactive_transfroms_method=None):
        super(VOCSegmentationRandomSample, self).__init__()

        assert target_type in ['Object', 'Class'], "`target_type` must in ['Object', 'Class']"
        assert image_set in ['train', 'val', 'trainval'], "`image_set` in ['train', 'val', 'trainval']"
        if transforms is not None:
            assert isinstance(transforms, TransfromsCompose)

        self.root = root_dir
        self.transforms = transforms
        self.interactive_transfroms_method = interactive_transfroms_method

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
        label_np = np.array(Image.open(label_path))

        ids = list(np.unique(label_np))
        if 0 in ids:
            ids.remove(0)
        if 255 in ids:
            ids.remove(255)
        
        object_id = random.choice(ids)
        interactive_map = sample_points_for_singal_object(label_np, object_id)
        
        label = np.uint8(label_np == object_id)
        label = Image.fromarray(label)
        if self.interactive_transfroms_method is not None:
            fg_interactive = self.interactive_transfroms_method(np.array(interactive_map[0]))
            fg_interactive = Image.fromarray(np.uint8(fg_interactive))
            bg_interactive = self.interactive_transfroms_method(np.array(interactive_map[1]))
            bg_interactive = Image.fromarray(np.uint8(bg_interactive))
        else:
            fg_interactive = Image.fromarray(interactive_map[0])
            bg_interactive = Image.fromarray(interactive_map[1])

        if self.transforms is not None:
            image, label, fg_interactive, bg_interactive = self.transforms(image, label, fg_interactive, bg_interactive)
            # image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return image, label, fg_interactive, bg_interactive, image_name

    def __len__(self):
        return len(self.file_names)
        

import os
from PIL import Image

from .utils.transforms import TransfromsCompose

class VOCSegmentation():
    '''
        VOC dataset class for Instance Segmentation.
    '''
    def __init__(self, root_dir, target_type='Object', image_set='train', transforms=None):
        super(VOCSegmentation, self).__init__()

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

        if self.transforms is not None:
            image, label = self.transforms(image, label)
            # image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return image, label, image_name

    def __len__(self):
        return len(self.file_names)
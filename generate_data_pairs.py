
import os
import glob
import cv2 as cv
from PIL import Image
import numpy as np
from tqdm import tqdm

from datasets.voc import VOCSegmentation
from datasets.utils.random_sampling import sample_points_for_singal_object

class GenerateDataPairs():
    def __init__(self, root_dir, save_dir='./voc2012', image_set='train', n_pairs:int=1, minimum_object_area:int=100) -> None:
        
        assert image_set in ['train', 'val', 'trainval']
        assert isinstance(n_pairs, int)
        self.n_pairs = n_pairs if n_pairs > 0 else 1
        self.minimum_object_area = minimum_object_area if minimum_object_area > 0 else 100
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, image_set) # save_dir/image_set
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        image_dir = os.path.join(save_dir, 'images') # image_dir
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        label_dir = os.path.join(save_dir, 'labels') # label_dir
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        interactive_dir = os.path.join(save_dir, 'interactives') # interactive_dir
        if not os.path.exists(interactive_dir):
            os.mkdir(interactive_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.interactive_dir = interactive_dir

        self.dataset = VOCSegmentation(root_dir, image_set=image_set)
        
        self.generate()
    
    def check_object_area(self, object_binary_mask, minimum_object_area=100):
        contours, hierarchy = cv.findContours(object_binary_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for c in contours:
            cur_area = cv.contourArea(c)
            if cur_area > max_area:
                max_area = cur_area
        return (max_area > minimum_object_area)

    def generate(self):
        for i, data in enumerate(tqdm(self.dataset)):
            # if i > 0:
            #     break
            image, label, name = data
            label_np = np.array(label)
            have_object_flag = False
            for id in np.unique(label_np):
                if id == 0 or id == 255: # skip background and object's contour
                    continue
                for k in range(self.n_pairs):
                    if self.check_object_area(np.uint8((label_np == id) * 255), 400):
                        object_interative_map = sample_points_for_singal_object(label_np, id) # [2, h, w]
                        if (not np.any(object_interative_map[0])) or (not np.any(object_interative_map[1])):
                            break
                        have_object_flag = True
                        fg_save_path = os.path.join(self.interactive_dir, f'{name}-{id}-fg-{k+1}.png')
                        Image.fromarray(object_interative_map[0]).save(fg_save_path)
                        bg_save_path = os.path.join(self.interactive_dir, f'{name}-{id}-bg-{k+1}.png')
                        Image.fromarray(object_interative_map[1]).save(bg_save_path)

            if have_object_flag:
                image_save_path = os.path.join(self.image_dir, f'{name}.jpg')
                image.save(image_save_path)
                label_save_path = os.path.join(self.label_dir, f'{name}.png')
                label.save(label_save_path)


# GenerateDataPairs(root_dir='/raid/home/guiyan/datasets', save_dir='./voc2012', image_set='val', n_pairs=1)
# GenerateDataPairs(root_dir='/raid/home/guiyan/datasets', save_dir='./voc2012', image_set='train', n_pairs=15)


import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.voc import VOCSegmentation
from utils.random_sampling import sample_points_for_singal_object

class GenerateDataPairs():
    def __init__(self, root_dir, save_dir='./interactives', image_set='train', n_pairs:int=1) -> None:
        
        assert image_set in ['train', 'val', 'trainval']
        assert isinstance(n_pairs, int)
        self.n_paris = n_pairs if n_pairs > 0 else 1
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, image_set)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir

        self.dataset = VOCSegmentation(root_dir, image_set=image_set)
        
        self.generate()
    
    def generate(self):
        for i, data in enumerate(tqdm(self.dataset)):
            # if i > 0:
            #     break
            image, label, name = data
            label_np = np.array(label)
            interative_maps = self.generate_from_label(label_np, self.n_paris) # [nums_object, n_pairs, 2, h, w]
            # print(name, interative_maps.shape)
            self.save_to_image(interative_maps, name)
    
    def generate_from_label(self, label, n_pairs):
        interative_maps = []
        for id in np.unique(label):
            if id == 0 or id == 255: # skip background and object's contour
                continue
            object_interative_maps = []
            for k in range(n_pairs):
                object_interative_map = sample_points_for_singal_object(label, id) # [2, h, w]
                object_interative_maps.append(object_interative_map) 
            interative_maps.append(np.stack(object_interative_maps)) # [n_pairs, 2, h, w]
        interative_maps = np.stack(interative_maps) # [nums_object, n_pairs, 2, h, w]
        return interative_maps

    def save_to_image(self, interative_maps, name):
        nums_object, n_pair = interative_maps.shape[:2]
        for id in range(nums_object):
            for k in range(n_pair):
                save_path = os.path.join(self.save_dir, f'{name}_{id+1}_fg_{k+1}.png')
                Image.fromarray(interative_maps[id, k, 0]).save(save_path)
                save_path = os.path.join(self.save_dir, f'{name}_{id+1}_bg_{k+1}.png')
                Image.fromarray(interative_maps[id, k, 1]).save(save_path)

# GenerateDataPairs(root_dir='/raid/home/guiyan/datasets', save_dir='./interactives', image_set='train', n_pairs=15)
# GenerateDataPairs(root_dir='/raid/home/guiyan/datasets', save_dir='./interactives', image_set='val', n_pairs=1)

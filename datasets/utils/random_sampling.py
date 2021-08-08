
'''
references url:
    https://github.com/isl-org/Intseg/blob/master/genIntSegPairs.m
'''

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# 0值到1的最小距离
def bwdist(binary_mask):
    distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)   
    return distance_map

# 在采点区域随机采一个点
def random_sample_a_points(sample_region:np.ndarray):
    sample_map = np.random.rand(*(sample_region.shape)) * sample_region
    max_value = np.max(sample_map)
    if max_value == 0:
        return None
    else:
        index = np.argmax(sample_map)
        y, x = np.unravel_index(index, shape=sample_map.shape, order='C')
        # y = index / ncols , x = index % ncols
        # y, x = index // sample_map.shape[1], index % sample_map.shape[1]
    
    return int(y), int(x)

# 在随机采点区域内随机采点
def sample_fg_points(fg_binary_mask:np.ndarray, num_points=5, d_margin=5, d_step=10):
    sample_region = np.int8(bwdist(1-fg_binary_mask) > d_margin) # 前景采点区域
    
    pc = np.zeros_like(fg_binary_mask) # positive channel
    for i in range(num_points):
        if i > 0:
            sample_region = np.int8(sample_region + (bwdist(pc) > d_step) == 2)
        index = random_sample_a_points(sample_region)
        if index is None:
            break
        pc[index] = 1
    return pc

def sample_bg_points_strategy_1(bg_binary_mask:np.ndarray, num_points=10, d_margin=5, d_step=10, d=40):
    fg = 1 - bg_binary_mask
    sample_region = np.int8((bwdist(fg) < d) - fg) # 背景采点区域
    nc = sample_fg_points(sample_region, num_points, d_margin, d_step)
    return nc

def sample_bg_points_strategy_2(bg_object_label:np.ndarray, num_points=5, d_margin=5, d_step=10):
    nc = np.zeros_like(bg_object_label)
    for id in np.unique(bg_object_label):
        if id == 0 or id == 255: # except bg and edge
            continue
        bg_object_mask = np.int8(bg_object_label == id)
        channel = sample_fg_points(bg_object_mask, num_points, d_margin, d_step)
        nc[channel == 1] = 1
    return nc

def sample_bg_points_strategy_3(bg_binary_mask:np.ndarray, num_points=10, d=40):
    nc = np.zeros_like(bg_binary_mask)
    fg = 1 - bg_binary_mask
    bg_sample_region = np.int8((bwdist(fg) < d) - fg) # 背景采点区域
    index = random_sample_a_points(bg_sample_region) 
    nc[index] = 1
    
    sample_region = 1 - bg_sample_region
    sample_region[index] = 1

    for i in range(num_points - 1):
        index = np.argmax(bwdist(sample_region) * bg_sample_region)
        y, x = np.unravel_index(index, shape=bg_sample_region.shape, order='C')
        sample_region[y, x] = 1
        nc[y, x] = 1
    return nc

def sample_points_for_singal_object(label, object_id):
    temp_label = label.copy()
    temp_label[label == 255] = 0
    
    id_list = np.unique(temp_label)
    strategy_index_list = [1, 3] if len(id_list) <= 2 else [1, 2, 3]
    strategy_index = np.random.choice(strategy_index_list) # random choice a sample strategy

    object_mask = np.uint8(label == object_id)
    if object_mask.any() == False:
        return None
    
    pc = sample_fg_points(object_mask)
    if strategy_index == 1:
        nc = sample_bg_points_strategy_1(1 - object_mask)
    elif strategy_index == 3:
        nc = sample_bg_points_strategy_3(1 - object_mask)
    else: # strategy_index == 2
        bg_object_label = label.copy()
        bg_object_label[label == object_id] = 0
        nc = sample_bg_points_strategy_2(bg_object_label)
    return np.stack([pc, nc]) # [2, h, w]

def sample_points_for_region(sample_region, num_points=10, d_step=10):
    pc = np.zeros_like(sample_region) # point channel
    for i in range(num_points):
        if i > 0:
            sample_region = np.int8(sample_region + (bwdist(pc) > d_step) == 2)
        index = random_sample_a_points(sample_region)
        if index is None:
            break
        pc[index] = 1
    return pc

def transfroming_user_interaction(interactive_map):
    interactive_map[interactive_map > 0] = 1
    distance_map = bwdist(interactive_map)
    distance_map[distance_map > 255] = 255
    return distance_map

def test_random_sample():
    
    image = np.array(Image.open('../../images/2007_000033.jpg'))
    label = np.array(Image.open('../../images/2007_000033.png'))
    label[label == 255] = 0

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.title("image")
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(2, 3, 2)
    plt.title("label")
    plt.axis('off')
    plt.imshow(label)
    
    fg_binary_mask = np.uint8(label == 1)
    bg_object_label = label.copy()
    bg_object_label[fg_binary_mask == 1] = 0

    plt.subplot(2, 3, 3)
    plt.title("fg points")
    plt.axis('off')
    pc = sample_fg_points(fg_binary_mask)
    temp = label.copy()
    y, x = np.nonzero(pc)
    plt.imshow(temp)
    plt.scatter(x, y, s=5, c='r')

    plt.subplot(2, 3, 4)
    plt.title("bg points strategy 1")
    plt.axis('off')
    pc = sample_bg_points_strategy_1(1 - fg_binary_mask)
    temp = label.copy()
    y, x = np.nonzero(pc)
    plt.imshow(temp)
    plt.scatter(x, y, s=5, c='r')
    
    plt.subplot(2, 3, 5)
    plt.title("bg points strategy 2")
    plt.axis('off')
    pc = sample_bg_points_strategy_2(bg_object_label)
    temp = label.copy()
    y, x = np.nonzero(pc)
    plt.imshow(temp)
    plt.scatter(x, y, s=5, c='r')

    plt.subplot(2, 3, 6)
    plt.title("bg points strategy 3")
    plt.axis('off')
    pc = sample_bg_points_strategy_3(1 - fg_binary_mask)
    temp = label.copy()
    y, x = np.nonzero(pc)
    plt.imshow(temp)
    plt.scatter(x, y, s=5, c='r')

    plt.savefig("../../images/test_random_sampling.png")
    # plt.show()

# test_random_sample()
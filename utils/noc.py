'''
    references url: https://github.com/MarcoForte/DeepInteractiveSegmentation/blob/master/interaction.py
'''

import cv2 as cv
import numpy as np

def dt(a):
    return cv.distanceTransform((a * 255).astype(np.uint8), cv.DIST_L2, 0)


def get_largest_incorrect_region(alpha, gt):

    largest_incorrect_BF = []
    for val in [0, 1]:

        incorrect = (gt == val) * (alpha != val)
        ret, labels_con = cv.connectedComponents(incorrect.astype(np.uint8) * 255)
        label_unique, counts = np.unique(labels_con[labels_con != 0], return_counts=True)
        if(len(counts) > 0):
            largest_incorrect = labels_con == label_unique[np.argmax(counts)]
            largest_incorrect_BF.append(largest_incorrect)
        else:
            largest_incorrect_BF.append(np.zeros_like(incorrect))

    largest_incorrect_cat = np.argmax([np.count_nonzero(x) for x in largest_incorrect_BF])
    largest_incorrect = largest_incorrect_BF[largest_incorrect_cat]
    return largest_incorrect, largest_incorrect_cat


def robot_click(alpha, gt):
    largest_incorrect, click_cat = get_largest_incorrect_region(alpha, gt)
    dist = dt(largest_incorrect)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return largest_incorrect, [y, x], click_cat

# from PIL import Image

# label = np.array(Image.open('2007_000033.png'))
# gt = (label == 1).astype(np.uint8)
# out = cv.imread('out.png', cv.IMREAD_GRAYSCALE)
# out = (out == 255).astype(np.uint8)
# # out = np.zeros_like(label)
# largest_incorrect, [y, x], click_cat = robot_click(out, gt)
# print([y, x], click_cat, largest_incorrect.shape)
# cv.imwrite('largest_incorrect.png', (largest_incorrect*255).astype(np.uint8))
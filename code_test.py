import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from datasets import VOCSegmentation, VOCSegmentationWithInteractive, VOCSegmentationRandomSample
from datasets.utils.transforms import TransfromsCompose


def test_GenerateDataPairs_result(name='2007_000032', dir='./interactives', image_set='train'):
    interactive_images_list = glob.glob(os.path.join(dir, image_set, f'{name}*_fg_1.png'))
    print(len(interactive_images_list), interactive_images_list[0])

    label = np.array(Image.open(f'./images/{name}.png'))
    plt.figure(figsize=(20, 8))
    for i in range(len(interactive_images_list)):
        plt.subplot(1, len(interactive_images_list), i+1)
        plt.title(f"object {i+1}")
        plt.axis('off')
        temp = label.copy()
        plt.imshow(temp)
        fg_image_path = os.path.join(dir, image_set, f'{name}_{i+1}_fg_1.png')
        fg = np.array(Image.open(fg_image_path))
        y, x = np.nonzero(fg)
        print(i+1, y, x)
        plt.scatter(x, y, s=5, c='r')
        bg_image_path = fg_image_path.replace('fg', 'bg')
        print(bg_image_path)
        bg = np.array(Image.open(bg_image_path))
        y, x = np.nonzero(bg)
        print(y, x)
        plt.scatter(x, y, s=5, c='g')
    
    plt.savefig(f"./images/test_generate_data_pairs_{name}.png")
    # plt.show()

# test_GenerateDataPairs_result(name='2007_000032', dir='./interactives', image_set='train')
# test_GenerateDataPairs_result(name='2007_000033', dir='./interactives', image_set='val')

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
    


# root_dir='/raid/home/guiyan/datasets'
# interactives_root_dir='./interactives'
# test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='image')
# test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='interactive')
# test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='val', main_data='image')
# test_VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='val', main_data='interactive')


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

    plt.savefig(f"./images/test_VOCSegmentationRandomSample_{image_set}_{object_id}_{image_name}.png")
    # plt.show()
    
# root_dir='/raid/home/guiyan/datasets'
# test_VOCSegmentationWithInteractive(root_dir, image_set='train')
# test_VOCSegmentationWithInteractive(root_dir, image_set='val')


root_dir='/raid/home/guiyan/datasets'
interactives_root_dir='./interactives'

dataset = VOCSegmentation(root_dir)
image, label, image_name = dataset[0]

dataset = VOCSegmentationWithInteractive(root_dir, interactives_root_dir, image_set='train', main_data='image')
image, label, fg_interactive, bg_interactive, image_name = dataset[0]

dataset = VOCSegmentationRandomSample(root_dir, image_set='train')
image, label, fg_interactive, bg_interactive, image_name, object_id = dataset[0]

import cv2 as cv
import numpy as np

def grabcut_optimization(image, predict_mask, fg_interactive_mask, bg_interactive_mask, iterCount=5):
    '''
        reference urls:
            https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html
            https://docs.opencv.org/master/d3/d47/group__imgproc__segmentation.html#ga909c1dda50efcbeaa3ce126be862b37f
    '''

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    predict_mask = predict_mask + 2 # possible background (2) and foreground(3) pixel
    predict_mask[fg_interactive_mask == 1] = 1 # obvious foreground (object) pixel (1)
    predict_mask[bg_interactive_mask == 1] = 0 # obvious background pixel (0)

    predict_mask, bgdModel, fgdModel = cv.grabCut(image, predict_mask, None, bgdModel,fgdModel, iterCount,cv.GC_INIT_WITH_MASK)
    predict_mask = np.where((predict_mask==2)|(predict_mask==0),0,1).astype('uint8')
    return predict_mask
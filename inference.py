
import cv2 as cv
import torch
import numpy as np

from networks.iFCN import iFCN as Networks

def dt(a):
    x = cv.distanceTransform((1- a).astype(np.uint8), cv.DIST_L2, 0)
    x[x > 255] = 255
    return x

class iFCN():
    def __init__(self, backbone_name, stride_out, upsample_type='deconv',
                weights_path=None, num_device=0, threshold=0.5):
        device_str = "cpu" if num_device < 0 else f"cuda:{num_device}"
        self.device = torch.device(device_str)
        self.threshold = threshold

        self.model = Networks(backbone_name, stride_out=stride_out, upsample_type=upsample_type)
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_np, fg_interactive_map, bg_interactive_map=None):

        fg_distance_map = dt(fg_interactive_map)
        if bg_interactive_map is None:
            bg_distance_map = np.zeros_like(fg_interactive_map)
        else:
            bg_distance_map = dt(bg_interactive_map)

        with torch.no_grad():
            x_np = np.concatenate([image_np, fg_distance_map[:, :, None], bg_distance_map[:, :, None]], axis=-1) / 255.0
            x = torch.from_numpy(x_np).permute(2, 0, 1)[None, :, :, :].float().to(self.device)
            alpha = torch.sigmoid(self.model(x))

        
        alpha = (alpha[0][0].cpu().numpy() > self.threshold).astype(np.uint8)

        return alpha


# image = cv.imread("./images/2007_000033.jpg")
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# fg = np.zeros(image.shape[:2])
# bg = np.zeros(image.shape[:2])
# net = iFCN('ResNet50', 8, weights_path='./models/ResNet50_8s_deconv/2021-08-11 01-41-10.746391_best_mean_iou.pkl', num_device=-1)

# fg[160, 220] = 1
# bg[300, 240] = 1

# # out = net.predict(image, fg, bg_interactive_map=None)
# out = net.predict(image, fg, bg)

# cv.imwrite('out.png', np.array(out*255, dtype=np.uint8))

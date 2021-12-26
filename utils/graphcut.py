
### references：
# - http://pmneila.github.io/PyMaxflow/tutorial.html
# - http://pmneila.github.io/PyMaxflow/maxflow.html
# - https://github.com/XunDiYang/Graph_Cut/blob/8fccf1016241ee45781486acec96df38b069181c/code/graph_cut.py

import maxflow
import cv2 as cv
import numpy as np

class GraphCutOptimizer:
    def __init__(self):
        # 按列分与按列分，边的结构，参考：http://pmneila.github.io/PyMaxflow/maxflow.html
        self.structure_col = np.array([[0, 0, 0],[0, 0, 0],[0, 1, 0]])
        self.structure_row = np.array([[0, 0, 0],[0, 0, 1],[0, 0, 0]])

        # 边的结构减去中心核的结构作为新的卷积核，用于计算边的权值
        self.center_kernel = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])

    def _get_region_term(self):
        '''
        数据/区域项（region properties term）：希望节点标签值相对应的前背景的概率值越大越好。
        '''
        self.region_term_bg = self.lambda_ * -np.log(1-self.prob)
        self.region_term_fg = self.lambda_ * -np.log(self.prob)
    
    def _get_boundary_term(self):
        '''
        （边界）平滑项（“boundary” properties term）：希望邻域内颜色相近的格子的标签相同。
        '''
        row_delta = cv.filter2D(self.image.astype(np.int16), cv.CV_32F, (self.structure_row - self.center_kernel).astype(np.int16))
        col_delta = cv.filter2D(self.image.astype(np.int16), cv.CV_32F, (self.structure_col - self.center_kernel).astype(np.int16))
        self.row_edge_weights = np.exp(-row_delta/(2 * self.sigma**2))
        self.col_edge_weights = np.exp(-col_delta/(2 * self.sigma**2))
        # 1 / (1 + np.sum(np.power(delta, 2)))

    def _build_graph_and_cut(self):
        '''
        构图并切割
        '''
        # 初始化图
        g = maxflow.GraphFloat()
        node_ids = g.add_grid_nodes(self.image.shape)

        # 构建边界项
        self._get_boundary_term()
        g.add_grid_edges(node_ids, weights=self.row_edge_weights, structure=self.structure_row, symmetric=True)
        g.add_grid_edges(node_ids, weights=self.col_edge_weights, structure=self.structure_col, symmetric=True)

        # 构建区域项
        self._get_region_term()
        g.add_grid_tedges(node_ids, self.region_term_fg, self.region_term_bg)

        # 找到最大流
        g.maxflow()

        # 获取分割结果
        return np.uint8(g.get_grid_segments(node_ids))

    def get_segment_result(self, image, prob, lambda_=1, sigma=50.):
        '''
        image: 灰度原图像
        prob: 概率图, [0, 1]
        lambda_: 区域项R前面的系数，默认为1，认为区域项R与边界项B一样重要；lambda_ * R + B
        sigma: 刻画相似度的标准差，exp(- (Ia - Ib)**2 / (2 * sigma**2))
        '''
        assert image.ndim == 2, "please use gray image."
        assert prob.min() >= 0 and prob.max() <= 1, "please keep `prob` in [0, 1]."
        assert image.shape == prob.shape, "image'shape and prob's shape is not the same."
        
        self.image = image
        self.prob = prob
        self.lambda_ = lambda_
        self.sigma = sigma

        return self._build_graph_and_cut()

if __name__ == "__main__":
    import torchvision
    import torch

    image = cv.imread("../images/cat.jpg")
    image_rgb = image[:, :, ::-1] # bgr -> rgb
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress= True)
    model.eval()
    image_tensor = torchvision.transforms.ToTensor()(image_rgb.copy())
    out = model(image_tensor[None,])
    cat_prob = torch.softmax(out["out"], dim=1)[0][8].detach().cpu().numpy() # 取出 猫 对应的概率

    gary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("cat_before_gco.png", (cat_prob>0.5).astype(np.uint8) * 255)
    r = GraphCutOptimizer().get_segment_result(gary, cat_prob)
    cv.imwrite("cat_after_gco.png", r * 255)
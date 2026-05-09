'''
Optimized with Numpy Vectorization
'''
from os import listdir
from os.path import join
import argparse
import numpy as np
import sys
import os
import random 

def compute_iou_matrix(X, centroids):
    """
    向量化计算 X 中所有框与 centroids 的 IoU 矩阵
    X.shape = (N, 2)
    centroids.shape = (k, 2)
    返回: iou_matrix.shape = (N, k)
    """
    # 增加维度以便使用 NumPy 的广播机制 (Broadcasting)
    # X_exp.shape = (N, 1, 2), C_exp.shape = (1, k, 2)
    X_exp = X[:, np.newaxis, :]
    C_exp = centroids[np.newaxis, :, :]

    # 计算交集宽高 (取两者最小值)
    inter_w = np.minimum(X_exp[..., 0], C_exp[..., 0])
    inter_h = np.minimum(X_exp[..., 1], C_exp[..., 1])
    
    # 过滤掉不相交的情况（虽然 anchor 聚类通常都从同一原点出发，一定相交，但安全起见）
    inter_area = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)

    # 计算各自的面积
    X_area = X[:, 0] * X[:, 1]
    C_area = centroids[:, 0] * centroids[:, 1]

    # 计算并集面积： A + B - 交集
    union_area = X_area[:, np.newaxis] + C_area[np.newaxis, :] - inter_area

    # 计算 IoU，加上 1e-10 防止除以 0
    return inter_area / np.maximum(union_area, 1e-10)

def avg_IOU(X, centroids):
    iou_matrix = compute_iou_matrix(X, centroids)
    # 找到每个样本的最大 IoU 并求平均
    return np.mean(np.max(iou_matrix, axis=1))

def write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file):
    anchors = centroids.copy()
    print("Anchors Shape:", anchors.shape)

    # 还原到配置文件中的像素尺寸
    anchors[:, 0] *= width_in_cfg_file
    anchors[:, 1] *= height_in_cfg_file

    # 按宽度对 anchor 进行排序
    sorted_indices = np.argsort(anchors[:, 0])
    anchors = anchors[sorted_indices]

    print('Anchors = \n', anchors)
        
    with open(anchor_file, 'w') as f:
        # 格式化写入
        anchor_strs = ['%0.2f,%0.2f' % (a[0], a[1]) for a in anchors]
        f.write(', '.join(anchor_strs) + '\n')
        f.write('%f\n' % (avg_IOU(X, centroids)))
    print()

def kmeans(X, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file):
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)    
    iteration = 0

    while True:
        iteration += 1          
        
        # 1. 向量化计算距离矩阵 D (Shape: N x k)
        iou_matrix = compute_iou_matrix(X, centroids)
        D = 1 - iou_matrix
        
        # 计算距离变化 (仅用于打印观察收敛情况)
        if iteration > 1:
            diff = np.sum(np.abs(old_D - D))
            print(f"iter {iteration}: dists diff = {diff:.6f}")
        else:
            print(f"iter {iteration}...")
            
        old_D = D.copy()
            
        # 2. 为每个样本分配最近的聚类中心
        assignments = np.argmin(D, axis=1)
        
        # 3. 检查是否收敛 (分配不再改变)
        if (assignments == prev_assignments).all():
            print("K-means 聚类收敛！")
            print("Centroids (Normalized) = \n", centroids)
            write_anchors_to_file(centroids, X, anchor_file, width_in_cfg_file, height_in_cfg_file)
            return

        # 4. 计算新的聚类中心 (向量化求平均)
        for j in range(k):
            mask = (assignments == j)
            # 如果某个中心没有任何样本，保留原中心或重新随机分配
            if np.any(mask):
                centroids[j] = np.mean(X[mask], axis=0)
        
        prev_assignments = assignments.copy()     

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'H:\dataset\fac_maks\train', 
                        help='数据集根目录，应包含 images 和 labels 文件夹\n')
    parser.add_argument('--output_dir', default='./', type=str, 
                        help='Output anchor directory\n')  
    parser.add_argument('--num_clusters', default=6, type=int, 
                        help='number of clusters\n')  
    parser.add_argument('--input_width', default=352, type=int, 
                        help='model input width\n')  
    parser.add_argument('--input_height', default=352, type=int, 
                        help='model input height\n')  
   
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    label_dir = join(args.data_dir, 'labels')

    if not os.path.exists(label_dir):
        print(f"错误: 找不到标签目录 {label_dir}")
        return

    annotation_dims = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"正在从 {label_dir} 读取 {len(label_files)} 个标注文件...")

    for label_file in label_files:
        label_path = join(label_dir, label_file)
        with open(label_path, 'r', encoding='utf-8') as f2:
            for line in f2.readlines():
                line = line.strip()
                if not line:
                    continue
                split_line = line.split()
                # YOLO 格式: class_id x_center y_center width height
                if len(split_line) >= 5:
                    w, h = float(split_line[3]), float(split_line[4])
                    # 过滤掉异常的 0 宽高数据
                    if w > 0 and h > 0:
                        annotation_dims.append([w, h])

    if len(annotation_dims) == 0:
        print("未发现有效的标注数据，请检查标签文件格式。")
        return

    annotation_dims = np.array(annotation_dims)
    print(f"成功加载 {annotation_dims.shape[0]} 个有效 bounding boxes.")

    eps = 0.005
    width_in_cfg_file = args.input_width
    height_in_cfg_file = args.input_height

    clusters_list = range(1, 11) if args.num_clusters == 0 else [args.num_clusters]

    for num_clusters in clusters_list:
        print(f"\n>>> 正在计算 {num_clusters} 个 anchors...")
        anchor_file = join(args.output_dir, f'anchors{num_clusters}.txt')
        
        # 使用 np.random.choice 更高效且避免重复选中同一个初始点
        indices = np.random.choice(annotation_dims.shape[0], num_clusters, replace=False)
        centroids = annotation_dims[indices].copy()
        
        kmeans(annotation_dims, centroids, eps, anchor_file, width_in_cfg_file, height_in_cfg_file)

if __name__ == "__main__":
    main(sys.argv)
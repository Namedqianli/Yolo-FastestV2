import os
import cv2
import random
import lmdb
import pickle
import tqdm
import io
import numpy as np

import torch
import albumentations as A
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

def aug(img):
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.ToSepia(p=0.5),
        A.AdvancedBlur((5, 5), always_apply=True, p=0.5),
        A.ImageCompression(quality_lower=30, quality_upper=95, p=0.5),
        A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.Defocus(radius=(3, 6), p=0.5),
    ])
    transformed = transform(image=img)
    transformed_img = transformed['image']
    
    return transformed_img

def contrast_and_brightness(img):
    alpha = random.uniform(0.25, 1.75)
    beta = random.uniform(0.25, 1.75)
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def motion_blur(image):
    if random.randint(1,2) == 1:
        degree = random.randint(2,3)
        angle = random.uniform(-360, 360)
        image = np.array(image)
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    else:
        return image

def augment_hsv(img, hgain = 0.0138, sgain = 0.678, vgain = 0.36):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


def random_resize(img):
    h, w, _ = img.shape
    rw = int(w * random.uniform(0.8, 1))
    rh = int(h * random.uniform(0.8, 1))

    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_LINEAR) 
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR) 
    return img

def img_aug(img):
    img = contrast_and_brightness(img)
    #img = motion_blur(img)
    #img = random_resize(img)
    #img = augment_hsv(img)
    return img

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    def __init__(self, root_path, img_size_width=352, img_size_height=352, imgaug=False):
        assert os.path.exists(root_path), "%s 路径不存在" % root_path

        self.root_path = root_path
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.imgaug = imgaug
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        
        self.data_list = []
        
        # 1. 定义 images 和 labels 的文件夹路径
        image_dir = os.path.join(root_path, 'images')
        label_dir = os.path.join(root_path, 'labels')

        if not os.path.exists(image_dir):
            raise Exception(f"未在 {root_path} 下找到 images 文件夹")

        # 2. 扫描所有图片文件
        print(f"正在扫描数据集: {image_dir} ...")
        for root, _, files in os.walk(image_dir):
            for file in files:
                extension = file.split(".")[-1].lower()
                if extension in self.img_formats:
                    img_path = os.path.join(root, file)
                    
                    # 3. 根据图片路径推导标签路径
                    # 将路径中的 /images/ 替换为 /labels/，将后缀换成 .txt
                    # 这种方式支持 images 下有子目录的情况
                    rel_path = os.path.relpath(img_path, image_dir)
                    label_rel_path = os.path.splitext(rel_path)[0] + ".txt"
                    label_path = os.path.join(label_dir, label_rel_path)

                    # 4. 检查对应的标签文件是否存在
                    if os.path.exists(label_path):
                        self.data_list.append((img_path, label_path))
                    else:
                        # 可选：如果有些图片没有标签，你可以选择跳过或者报错
                        print(f"警告: 找不到对应的标签文件，已跳过: {label_path}")

        print(f"数据集加载完成，共计 {len(self.data_list)} 组有效样本。")

    def __getitem__(self, index):
        # 5. 从预存的元组中直接获取路径
        img_path, label_path = self.data_list[index]

        # 图像读取与预处理
        img = Image.open(img_path).convert('RGB') # 确保是RGB
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.img_size_width, self.img_size_height), interpolation=cv2.INTER_LINEAR) 
        
        # 数据增强
        if self.imgaug:
            img = img_aug(img)
            
        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # 加载 label 文件
        label = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    l = line.split()
                    # YOLO 格式通常是: class x y w h
                    # 这里保持你原来的逻辑：[0, class, x, y, w, h]
                    if len(l) == 5:
                        label.append([0, l[0], l[1], l[2], l[3], l[4]])
        
        # 如果该图片没有标注框，创建一个空的 array，避免程序崩溃
        if len(label) == 0:
            label = np.zeros((0, 6), dtype=np.float32)
        else:
            label = np.array(label, dtype=np.float32)
        
        return torch.from_numpy(img).float(), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

class FastLmdbDataset(data.Dataset):
    def __init__(self, db_path, imgaug=False):
        self.db_path = db_path
        self.imgaug = imgaug
        self.env = None
        
        # 预先获取长度
        tmp_env = lmdb.open(db_path, readonly=True, lock=False)
        self.length = int(tmp_env.begin().get('length'.encode()).decode())
        tmp_env.close()

    def _init_db(self):
        self.env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
            
        with self.env.begin(write=False) as txn:
            byte_data = txn.get(f'{index:08}'.encode())
        
        # 反序列化
        data_dict = pickle.loads(byte_data)
        
        # 图像解码 (从 JPEG 字节流转回 BGR)
        img_array = np.frombuffer(data_dict['img'], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # 获取标签
        label = data_dict['label']
        
        # 只有动态数据增强保留在这里（每次 epoch 都不一样）
        if self.imgaug:
            img = img_aug(img)
            
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        return torch.from_numpy(img).float(), torch.from_numpy(label)

    def __len__(self):
        return self.length

def create_preprocessed_lmdb(dataset, save_path, map_size=1e12):
    """
    dataset: 现有的 TensorDataset 实例
    save_path: lmdb 文件夹路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    env = lmdb.open(save_path, map_size=int(map_size))
    
    with env.begin(write=True) as txn:
        for i in tqdm.tqdm(range(len(dataset)), desc="Packing Data"):
            img_path, label_path = dataset.data_list[i]
            
            # 1. 预处理图像：读取并 Resize 到统一尺寸
            img = cv2.imread(img_path)
            img = cv2.resize(img, (dataset.img_size_width, dataset.img_size_height))
            
            # 2. 预处理标签：解析成 numpy 数组
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split()
                    if len(l) == 5:
                        label.append([0, l[0], l[1], l[2], l[3], l[4]])
            
            label_np = np.array(label, dtype=np.float32) if label else np.zeros((0, 6), dtype=np.float32)

            # 3. 封装成字典并序列化
            # 为了节省空间，图像可以转为 jpeg 编码的字节流，或者直接存 raw numpy
            # 这里推荐用 cv2.imencode 压缩一下，能省下 5-10 倍磁盘空间且解码极快
            _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            data_struct = {
                'img': img_encoded.tobytes(),
                'label': label_np
            }
            
            # 存储二进制序列化数据
            txn.put(f'{i:08}'.encode(), pickle.dumps(data_struct))
            
        txn.put('length'.encode(), str(len(dataset)).encode())
    
    env.close()
    print(f"预处理数据集已就绪: {save_path}")

def process_data_worker(idx, img_path, label_path, width, height):
    """
    负责具体的 IO 和 CPU 密集型计算
    """
    try:
        # 1. 图像处理
        img = cv2.imread(img_path)
        if img is None: return None
        img = cv2.resize(img, (width, height))
        _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 2. 标签处理
        label = []
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    l = line.strip().split()
                    if len(l) == 5:
                        label.append([0, l[0], l[1], l[2], l[3], l[4]])
        
        label_np = np.array(label, dtype=np.float32) if label else np.zeros((0, 6), dtype=np.float32)
        
        # 3. 序列化
        data_struct = {'img': img_encoded.tobytes(), 'label': label_np}
        return idx, pickle.dumps(data_struct)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def create_lmdb_streaming(dataset, save_path, map_size=1e12, num_workers=32, batch_size=6000):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    env = lmdb.open(save_path, map_size=int(map_size))
    total_samples = len(dataset)
    
    # 使用 tqdm 显示总体进度
    pbar = tqdm.tqdm(total=total_samples, desc="Overall Progress")
    
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = range(batch_start, batch_end)
        
        batch_data = []
        # 使用 ProcessPoolExecutor 真正利用多核处理图像 Resize
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in batch_indices:
                img_path, label_path = dataset.data_list[i]
                futures.append(executor.submit(
                    process_data_worker, i, img_path, label_path, 
                    dataset.img_size_width, dataset.img_size_height
                ))
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    batch_data.append(res)
        
        # 写入当前 Batch
        with env.begin(write=True) as txn:
            for idx, binary_data in batch_data:
                # 使用 8 位填充索引作为 key，方便排序
                txn.put(f'{idx:08}'.encode(), binary_data)
        
        pbar.update(batch_end - batch_start)
    
    # 写入长度元数据
    with env.begin(write=True) as txn:
        txn.put('length'.encode(), str(total_samples).encode())
    
    pbar.close()
    env.close()
    print(f"\n[Done] LMDB 数据库已存至: {save_path}")

if __name__ == "__main__":
    # 使用方式：传入包含 images 和 labels 文件夹的根目录
    # 假设目录结构为：
    # /data/widerface/
    # ├── images/
    # └── labels/
    path = r"C:\dataset\fac_maks\train"
    dataset = TensorDataset(path)
    if len(dataset) > 0:
        img, label = dataset.__getitem__(0)
        print("Image shape:", img.shape)
        print("Label shape:", label.shape)
    else:
        print("数据集为空，请检查路径。")

    create_lmdb_streaming(dataset, path)

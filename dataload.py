import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils,datasets
from PIL import Image
import pandas as pd
import numpy as np
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")

import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TwoCamerasDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path #文件路径
        self.transform = transform #对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.root_path) #把路径下的所有文件放在一个列表中
        # 获取所有图片文件
        # all_images = [f for f in os.listdir(root_path) if f.endswith('.png')]
        all_images=self.images
        # 按照身份进行分组
        identity_groups = {}
        for image in all_images:
            # identity = image.split('_')[0]  # 假设文件名格式为 identity_index.png
            identity=int(image.split('\\')[-1].split('.')[0].split('_')[0][3:])
            if identity not in identity_groups:
                identity_groups[identity] = []
            identity_groups[identity].append(image)

        # 分成两组，模拟两个摄像头
        self.camera1_images = []
        self.camera2_images = []
        self.labels = []
        for object_name, images in identity_groups.items():
            # images = sorted(images)  # 确保图像按照角度排序
            random.shuffle(images)
            split_index = len(images) // 2
            self.camera1_images.extend(images[:split_index]) #把符合要求的文件放在同一个列表中
            self.camera2_images.extend(images[split_index:])
            self.labels.extend([object_name] * split_index)  # 添加身份标签

    def __len__(self):
        return min(len(self.camera1_images), len(self.camera2_images))

    def __getitem__(self, idx):
        img1_filename = self.camera1_images[idx]
        img2_filename = self.camera2_images[idx]
        label = self.labels[idx]

        img1_path = os.path.join(self.root_path, img1_filename)
        img2_path = os.path.join(self.root_path, img2_filename)

        img1 = Image.open(img1_path).convert('RGB')#读取图像
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2,label

def load_data(extracted_folder,batch_size):
    #pytorch dataset

    train_root_path = extracted_folder+"/train"
    eval_root_path = extracted_folder+"/eval"
    train_data=TwoCamerasDataset(train_root_path,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))
    eval_data=TwoCamerasDataset(eval_root_path,transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

    # Wraps iterables around the Datasets to enable easy access to the instances.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=True,drop_last=True)

    return train_dataloader, eval_dataloader
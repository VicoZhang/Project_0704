"""
encoding = 'utf-8'
author: Vico Zhang
此文件生成数据集，返回 train_dataset 与 test_dataset
More information: https://github.com/VicoZhang/Project_0704.git
"""


import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class ReadData(Dataset):
    def __init__(self, root, label):
        self.root = root
        self.label = label
        self.path = os.path.join(self.root, self.label)
        self.img_path = os.listdir(self.path)
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        img_name = self.path[index]
        img_item_path = os.path.join(self.root, self.label, img_name)
        img = Image.open(img_item_path)
        img = self.transforms(img)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.img_path)


data_dir = '../Gray_scale/Gray_image'
type_1 = '01'
type_2 = '04'
type_3 = '07'
type_4 = '08'

type_1_dataset = ReadData(data_dir, type_1)
type_2_dataset = ReadData(data_dir, type_2)
type_3_dataset = ReadData(data_dir, type_3)
type_4_dataset = ReadData(data_dir, type_4)


# 训练集与验证集的长度
train_len = 150
test_len = 48

type_1_dataset_train_set, type_1_dataset_test_set = random_split(
    dataset=type_1_dataset,
    lengths=[train_len, test_len],
    generator=torch.Generator().manual_seed(704)
)
type_2_dataset_train_set, type_2_dataset_test_set = random_split(
    dataset=type_2_dataset,
    lengths=[train_len, test_len],
    generator=torch.Generator().manual_seed(704)
)
type_3_dataset_train_set, type_3_dataset_test_set = random_split(
    dataset=type_3_dataset,
    lengths=[train_len, test_len],
    generator=torch.Generator().manual_seed(704)
)
type_4_dataset_train_set, type_4_dataset_test_set = random_split(
    dataset=type_4_dataset,
    lengths=[train_len, test_len],
    generator=torch.Generator().manual_seed(704)
)


train_dataset = type_1_dataset_train_set + type_2_dataset_train_set\
                + type_3_dataset_train_set + type_4_dataset_train_set
test_dataset = type_1_dataset_test_set + type_2_dataset_test_set\
                + type_3_dataset_test_set + type_4_dataset_test_set


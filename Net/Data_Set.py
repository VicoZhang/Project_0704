"""
encoding = 'utf-8'
author: Vico Zhang
此文件生成数据集，返回 train_dataset 与 test_dataset
More information: https://github.com/VicoZhang/Project_0704.git
"""

import os
import torch
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import tensorboard


class ReadData(Dataset):
    def __init__(self, root, type):
        self.root = root
        self.type = type
        self.encode = self._take_encode()
        self.label = self._decode()
        self.img_path = os.listdir(os.path.join(self.root, self.type))
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root, self.type, img_name)
        img = Image.open(img_item_path)
        img = self.transforms(img)
        return img, self.label

    def __len__(self):
        return len(self.img_path)

    def _take_encode(self):
        label_list = os.listdir(self.root)
        coding_list = one_hot(torch.arange(len(label_list)))
        return coding_list[label_list.index(self.type)]

    def _decode(self):
        return self.encode.max(dim=0)[1]


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

train_dataset = type_1_dataset_train_set + type_2_dataset_train_set \
                + type_3_dataset_train_set + type_4_dataset_train_set
test_dataset = type_1_dataset_test_set + type_2_dataset_test_set \
               + type_3_dataset_test_set + type_4_dataset_test_set

if __name__ == '__main__':
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # writer = tensorboard.SummaryWriter(log_dir='dataset_log')
    step = 0
    for item in data_loader:
        step += 1
        test_img, test_label = item
        print(test_label)
    #     writer.add_images('train_data', test_img, step)
    # writer.close()
    print("测试通过")

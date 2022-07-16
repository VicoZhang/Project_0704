"""
encoding = 'utf-8'
author: Vico Zhang
此文件为底层文件，请勿修改
More information: https://github.com/VicoZhang/Project_0704.git
"""

import os
import numpy as np
import re


class ReadValue:
    """
    底层，读取.out文件。

    调用：value_reader = ReadValue(path), path为.out文件路径,
    【此路径将以列表的形式给出，因为同一次仿真的.out文件可能不止一个】

    方法：ReadValue.read_value，返回值为tuple (仿真数据(ndarray, 项目*采样点)，仿真时间(ndarray, 1*采样点))

    """

    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8') as f:
            self.temp = f.readlines()[1:]
        self.TIME_NUM = len(self.temp)
        self.ITEM_NUM = len(re.split(r"[ ]+", self.temp[1].replace("\n", " "))[1:-1])

    def read_value(self) -> tuple:
        value_temp = np.zeros((self.TIME_NUM, self.ITEM_NUM))
        for i in range(self.TIME_NUM):
            value_temp[i] = np.array(re.split(r"[ ]+", self.temp[i].replace("\n", " "))[1:-1]).astype(np.float64)
        time_temp = value_temp[:, 0]
        value_temp = value_temp.T[1:]
        return value_temp, time_temp


class ReadIndex:
    """
    底层，读取.inf文件

    调用:index_reader = ReadIndex(root_path, file_name) root_path为数据根目录，file_name为.inf文件名

    方法:index_reader.read_index(), 返回一个字典，字典的键值为 inf 中的项目名，值为项目对应在数据中的序列
    """

    def __init__(self, root_path, file_name):
        self.root_path = root_path
        self.file_name = file_name
        path = os.path.join(self.root_path, self.file_name)
        with open(path, encoding='utf-8') as f:
            self.temp = f.readlines()
        self.INDEX_NUM = len(self.temp)

    def read_index(self) -> dict:
        index_temp = ['0'] * self.INDEX_NUM
        reformat = re.compile(r"PGB\((?P<PGB>.*?)\)(.*?)Desc=\"(?P<Desc>.*?)\"", re.S)
        for line in self.temp:
            idx = reformat.finditer(line)
            for item in idx:
                index_temp[eval(item.group("PGB")) - 1] = item.group("Desc").replace(":", "_")
        return dict(zip(index_temp, np.arange(self.INDEX_NUM)))


class ReadInformation:
    """
    底层，读取 mrunout.out 信息

    调用: information_reader = ReadInformation (root_path, file_name), root_path为数据根目录，file_name为文件名

    方法:

    information_reader.read_information(), 返回不包含表头的 information 数据矩阵，行为仿真次数，列为对应项目

    information_reader.read_name(), 返回一个字典，字典的键值为 information 项目名，
    值为对应在 information_reader.read_information() 输出数据的列数
    """

    def __init__(self, root_path, file_name):
        self.root_path = root_path
        self.filename = file_name
        self.path = os.path.join(self.root_path, self.filename)
        with open(self.path) as f1:
            self.inf_temp = f1.readlines()
        self.RUN = len(self.inf_temp)
        self.NUM = len(re.split(r"[ ]+", self.inf_temp[2].replace("\n", " "))[1:-1])
        self.DELETE = 0

    def read_information(self):
        information_temp = np.zeros((self.RUN, self.NUM))
        for i in range(2, self.RUN):
            temp = re.split(r"[ ]+", self.inf_temp[i].replace("\n", " "))[1:-1]
            if len(temp) == 0:
                self.DELETE = i
                break
            else:
                information_temp[i - 2] = re.split(r"[ ]+", self.inf_temp[i].replace("\n", " "))[1:-1]
        return information_temp[:self.DELETE - 2]

    def read_name(self) -> dict:
        name_temp = np.array(re.split(r"[ ]+", self.inf_temp[1].replace("\n", " "))[2:-1])
        index_temp = np.arange(self.NUM)
        return dict(zip(name_temp, index_temp))


class FilesScanning:
    """
    获取.out和.inf文件路径
    """

    def __init__(self, root):
        self.root = root
        self.out_files_temp = []
        self.inf_files_temp = []
        for item in os.listdir(self.root):
            if item.split(".")[-1] == 'out':
                self.out_files_temp.append(item)
            elif item.split(".")[-1] == 'inf':
                self.inf_files_temp.append(item)
        self.out_files_temp = np.array(self.out_files_temp)
        self.inf_files_temp = np.array(self.inf_files_temp)

    def out(self):
        return self.out_files_temp

    def inf(self):
        return self.inf_files_temp


if __name__ == '__main__':
    # 以下为测试程序
    value_reader = ReadValue
    index_reader = ReadIndex
    information_reader = ReadInformation
    file_reader = FilesScanning
    data = value_reader('../Simulation/Project_220704.gf42/T1_r00001_02.out').read_value()
    index = index_reader('../Simulation/Project_220704.gf42', 'T1_r00001.inf').read_index()
    information = information_reader('../Simulation/Project_220704.gf42', 'mrunout_01.out')
    print(np.array([information.read_information(), information.read_name()]))
    print(data[0].shape, data[1].shape)
    print(index)
    print("测试正常")

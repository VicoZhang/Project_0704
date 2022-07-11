import os
import numpy as np
import re


class ReadValue:
    """
    读取.out文件，包括数据文件和多重运行模块控制的变量
    """

    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8') as f:
            self.temp = f.readlines()[1:]
        self.TIME_NUM = len(self.temp)
        self.ITEM_NUM = len(re.split(r"[ ]+", self.temp[1].replace("\n", " "))[1:-1])

    def read_value(self):
        value_temp = np.zeros((self.TIME_NUM, self.ITEM_NUM))
        for i in range(self.TIME_NUM):
            value_temp[i] = np.array(re.split(r"[ ]+", self.temp[i].replace("\n", " "))[1:-1]).astype(np.float64)
        time_temp = value_temp[:, 0]
        value_temp = value_temp.T[1:]
        return value_temp, time_temp


class ReadIndex:
    """
    读取.inf文件，匹配数据变量
    """

    def __init__(self, root_path, file_name):
        self.root_path = root_path
        self.file_name = file_name
        path = os.path.join(self.root_path, self.file_name)
        with open(path, encoding='utf-8') as f:
            self.temp = f.readlines()
        self.INDEX_NUM = len(self.temp)

    def read_index(self):
        index_temp = ['0'] * self.INDEX_NUM
        reformat = re.compile(r"PGB\((?P<PGB>.*?)\)(.*?)Desc=\"(?P<Desc>.*?)\"", re.S)
        for line in self.temp:
            idx = reformat.finditer(line)
            for item in idx:
                index_temp[eval(item.group("PGB")) - 1] = item.group("Desc").replace(":", "_")
        return dict(zip(index_temp, np.arange(self.INDEX_NUM)))


class ReadInformation:
    """
    读取 mrunout 信息
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
                information_temp[i-2] = re.split(r"[ ]+", self.inf_temp[i].replace("\n", " "))[1:-1]
        return information_temp[:self.DELETE-2]

    def read_name(self):
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

"""
encoding = 'utf-8'
author: Vico Zhang
此文件创建 Data类, 其调用 Read_from_out.py, 生成类，类的属性包括了仿真信息，方法可以用于数据转换，这里以生成灰度图为例。
More information: https://github.com/VicoZhang/Project_0704.git
"""

import Read_from_out

import random
import imageio
import numpy as np
import os


class Data:
    def __init__(self, root, i):  # root 为仿真数据文件夹, i 为仿真次数
        # 数据属性设置
        self.NO = i  # 仿真次数，用于数据索引
        self.name = 'r{:05}'.format(self.NO + 1)  # 数据命名，用于生成文件名
        self.code = None  # 数据编码，用于分类

        # 仿真信息设置，可根据实际情况更改，在 self._read_inf() 中也需要相应修改
        self.frequency = None
        self.fault_r = None
        self.fault_type = None
        self.fault_time = None

        # 仿真数据设置
        self.data_length = 160  # 此值可修改，这里：0.2s正常+0.4s故障+0.2s正常，采样频率2000Hz，观测数据点为160
        self.data = {}  # 以字典形式存储数据，便于调用
        self.simulation_time = None  # 仿真时间序列

        # 路径设置
        self.root = root  # 仿真数据文件夹
        self.mrunout_name = 'mrunout_01.out'  # 多重运行模块输出变量文件，可根据实际情况修改
        self.out_name = 'T1_r{:05}.out'.format(self.NO + 1)  # out文件以1开始命名
        self.out_path = []  # 列表存储 .out 文件名，因为同一次仿真的.out文件可能不止一个
        self.inf_name = 'T1_r{:05}.inf'.format(self.NO + 1)

        # 属性赋值, 这些函数外部不调用！
        self._generate_out_path()
        self._read_inf()
        self._read_time()
        self._read_data()
        self._decoding()

        # 生成灰度图
        self.gray_root = '../Gray_scale/Gray_image/{}'.format(self.code)  # 灰度图像文件夹
        self.gray_path = os.path.join(self.gray_root, self.name + '.jpg')  # 灰度图像名
        self.gray_inf_root = '../Gray_scale/Label/{}'.format(self.code)  # 灰度图信息文件夹
        self.gray_inf_path = os.path.join(self.gray_inf_root, self.name)  # 灰度图信息文件

    def _generate_out_path(self):
        """
        判断有几个.out文件并返回文件路径列表，为 self.out_path 赋值
        """
        path = os.path.join(self.root, self.out_name)
        num_temp = 1
        if not os.path.exists(path):
            while True:
                path = os.path.join(self.root,
                                    self.out_name.split('.')[0] + '_{:02}.'.format(num_temp) +
                                    self.out_name.split('.')[1])
                if os.path.exists(path):
                    self.out_path.append(path)
                else:
                    break
                num_temp += 1
        else:
            self.out_path.append(path)

    def _read_inf(self):
        """
        从 mrunout_01.out 中获取仿真条件, 为各项仿真信息幅值，可根据实际情况修改
        """
        inf_reader = Read_from_out.ReadInformation(self.root, self.mrunout_name)
        self.frequency = inf_reader.read_information()[self.NO][inf_reader.read_name()['Frequency']]
        self.fault_r = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_R']]
        self.fault_type = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_Type']]
        self.fault_time = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_Time']]

    def _read_time(self):
        """
        从 r00001.out 中获取仿真时间，并返回故障时间对应的索引，在 _read_data(self) 中使用
        """
        self.simulation_time = Read_from_out.ReadValue(self.out_path[0]).read_value()[1]
        return np.argwhere(self.simulation_time == self.fault_time)

    def _read_data(self):
        """
        从 r00001.inf 与 r00001.out 中获取数据，数据长度由 self.data_length 和故障发生时间共同决定，为 self.data 赋值
        """
        index_reader = Read_from_out.ReadIndex(self.root, self.inf_name).read_index()
        value_reader = np.array(Read_from_out.ReadValue(self.out_path[0]).read_value()[0])
        begin = self._read_time()[0][0] - 40
        if len(self.out_path) != 1:
            for path in self.out_path[1:]:
                value_reader = np.concatenate((value_reader, Read_from_out.ReadValue(path).read_value()[0]))
        for item in index_reader.items():
            self.data[item[0]] = value_reader[item[1]][begin: begin + self.data_length]

    def _decoding(self):
        """
        编码，可根据实际情况修改编码方式，这里根据故障类型编码，01 -> fault_type = 1
        """
        self.code = "{:02}".format(int(self.fault_type))

    def generate_gray_information(self):
        """
        生成并存储灰度图信息文件，可外部调用
        """
        if not os.path.exists(self.gray_inf_root):
            os.makedirs(self.gray_inf_root)
        with open(self.gray_inf_path, 'w', encoding='utf-8') as f:
            f.write('name,{},\n'
                    'frequency,{},Hz\n'
                    'fault_R,{},ohms\n'
                    'fault_type,{},\n'
                    'fault_time,{},s'
                    .format(self.name, self.frequency, self.fault_r, self.fault_type, self.fault_time))

    def generate_grayscale(self, data, dimension):
        """
        生成并存储灰度图，可外部调用
        :param data: 用于生成灰度图的数据，1维，
        :param dimension: 像素数
        """
        length = dimension ** 2
        begin = random.randint(0, len(data) - length)
        gray_data_temp = data[begin: begin + length]
        gray_data_temp = np.reshape(gray_data_temp, (dimension, dimension))
        if not os.path.exists(self.gray_root):
            os.makedirs(self.gray_root)
        imageio.imwrite(self.gray_path, gray_data_temp)
        self.generate_gray_information()


if __name__ == '__main__':
    # 测试及调用示例
    # 循环测试
    for ii in range(54):
        test = Data('../Simulation/Project_220704.gf42', ii)
        test.generate_grayscale(np.concatenate((test.data['Ia_2'], test.data['Ib_2'], test.data['Ic_2'])), 21)
    # 单次测试
    # test = Data('../Simulation/Project_220704.gf42', 0)
    # print()
    print("测试通过")

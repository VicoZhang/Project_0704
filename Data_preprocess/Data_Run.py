import Read_from_out
import random
import imageio
import numpy as np
import os


class Data:
    def __init__(self, root, i):
        # 数据属性设置
        self.NO = i
        self.name = 'r{:05}'.format(self.NO + 1)
        self.code = None
        self.frequency = None
        self.fault_r = None
        self.fault_type = None
        self.fault_time = None
        self.data_length = 160  # 0.2s正常+0.4s故障+0.2s正常，采样频率2000Hz，观测数据点为160
        self.data = {}
        self.simulation_time = None

        # 路径设置
        self.root = root
        self.mrunout_name = 'mrunout_01.out'
        self.out_name = 'T1_r{:05}.out'.format(self.NO + 1)  # out文件以1开始命名
        self.out_path = []
        self.inf_name = 'T1_r{:05}.inf'.format(self.NO + 1)

        # 属性赋值
        self._generate_out_path()
        self._read_inf()
        self._read_time()
        self._read_data()
        self._decoding()

        # 生成灰度图
        self.gray_root = '../Gray_scale/Gray_image/{}'.format(self.code)
        self.gray_path = os.path.join(self.gray_root, self.name + '.jpg')
        self.gray_inf_root = '../Gray_scale/Label/{}'.format(self.code)
        self.gray_inf_path = os.path.join(self.gray_inf_root, self.name)

    def _generate_out_path(self):
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
        # 从 mrunout_01.out 中获取仿真条件
        inf_reader = Read_from_out.ReadInformation(self.root, self.mrunout_name)
        self.frequency = inf_reader.read_information()[self.NO][inf_reader.read_name()['Frequency']]
        self.fault_r = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_R']]
        self.fault_type = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_Type']]
        self.fault_time = inf_reader.read_information()[self.NO][inf_reader.read_name()['Fault_Time']]

    def _read_time(self):
        # 从 r00001.out 中获取仿真时间，并返回故障时间对应的索引
        self.simulation_time = Read_from_out.ReadValue(self.out_path[0]).read_value()[1]
        return np.argwhere(self.simulation_time == self.fault_time)

    def _read_data(self):
        # 从 r00001.inf 与 r00001.out 中获取数据
        index_reader = Read_from_out.ReadIndex(self.root, self.inf_name).read_index()
        value_reader = np.array(Read_from_out.ReadValue(self.out_path[0]).read_value()[0])
        begin = self._read_time()[0][0] - 40
        if len(self.out_path) != 1:
            for path in self.out_path[1:]:
                value_reader = np.concatenate((value_reader, Read_from_out.ReadValue(path).read_value()[0]))
        for item in index_reader.items():
            self.data[item[0]] = value_reader[item[1]][begin: begin+self.data_length]

    def _decoding(self):
        self.code = "{:02}".format(int(self.fault_type))

    def generate_gray_information(self):
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
        length = dimension ** 2
        begin = random.randint(0, len(data)-length)
        gray_data_temp = data[begin: begin + length]
        gray_data_temp = np.reshape(gray_data_temp, (dimension, dimension))
        if not os.path.exists(self.gray_root):
            os.makedirs(self.gray_root)
        imageio.imwrite(self.gray_path, gray_data_temp)
        self.generate_gray_information()


if __name__ == '__main__':
    # 循环测试
    for ii in range(54):
        test = Data('../Simulation/Project_220704.gf42', ii)
        test.generate_grayscale(np.concatenate((test.data['Ia_2'], test.data['Ib_2'], test.data['Ic_2'])), 21)
    # 单次测试
    # test = Data('../Simulation/Project_220704.gf42', 0)
    # print()
    print("测试通过")

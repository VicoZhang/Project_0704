"""
encoding = 'utf-8'
author: Vico Zhang
加载训练好的网络，进行验证。
More information: https://github.com/VicoZhang/Project_0704.git
"""

import torch
from torch.utils.data import DataLoader
import Net.Net
import Net.Data_Set

# 设置路径
model_path = '../Result_1wave/Net_result_2022_07_16T16_13_53.pth'
data_path = 'Test_data'
type_1 = '01'
type_2 = '04'
type_3 = '07'
type_4 = '08'

# 加载神经网络
net = Net.Net.Net()
net.load_state_dict(torch.load(model_path))

# 加载验证数据
test_data_1 = Net.Data_Set.ReadData(data_path, type_1)
test_data_2 = Net.Data_Set.ReadData(data_path, type_2)
test_data_3 = Net.Data_Set.ReadData(data_path, type_3)
test_data_4 = Net.Data_Set.ReadData(data_path, type_4)
test_data = test_data_1 + test_data_2 + test_data_3 + test_data_4
test_data_length = test_data.__len__()
data_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)


# 定义评估函数
def calculate(o, l, t):
    """
    :param o: output
    :param l: label
    :param t: type
    :return: (accuracy, recall)
    """
    type_temp = torch.zeros_like(o)
    type_temp = torch.fill_(type_temp, int(t))
    right_num = ((torch.eq(o, l) / torch.eq(l, type_temp)) == 1).sum()  # 判断o, l, t三个向量对应元素是否完全相等
    predict_positive_num = torch.eq(o, type_temp).sum()
    positive_num = torch.eq(l, type_temp).sum()
    return right_num / predict_positive_num, right_num / positive_num


# calculate(o, l, t)的测试
# a = torch.tensor([3, 2, 0, 1, 2, 2, 2, 1])
# b = torch.tensor([3, 2, 0, 2, 2, 2, 3, 2])
# print(calculate_accuracy(a, b, 2))

# 进行验证
total_accuracy = 0
total_recall = 0
test_epoch = 0
test_type = 3
for img, label in data_loader:
    output = net(img)
    output = torch.argmax(output, dim=-1)
    accuracy, recall = calculate(output, label, test_type)
    total_accuracy += accuracy
    total_recall += recall
    test_epoch += 1
print("对于类别{}，准确率为{}%，召回率为{}%".format(test_type,
                                      total_accuracy / test_epoch * 100,
                                      total_recall / test_epoch * 100))

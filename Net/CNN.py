import Net
import Data_Set
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch import nn, optim
from datetime import datetime
import torch
import os


train_data_loader = DataLoader(Data_Set.train_dataset, batch_size=4, shuffle=True)
test_data_loader = DataLoader(Data_Set.test_dataset, batch_size=4, shuffle=True)

print("训练集长度为：{}".format(len(Data_Set.train_dataset)))
print("测试集长度为：{}".format(len(Data_Set.test_dataset)))

epochs = 50
learn_rate = 0.01
train_step = 0
test_step = 0
test_loss = 0
time = "{0:%Y_%m_%dT%H_%M_%S}".format(datetime.now())
tensorboard_path = 'logs_1wave/{}/'.format(time)
net_save_path = '../Result_1wave/Net_result_{}.pth'.format(time)

net = Net.Net().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), learn_rate)
writer = tensorboard.SummaryWriter(log_dir=tensorboard_path)

for epoch in range(epochs):
    print("第{}轮训练开始".format(epoch+1))

    for data in train_data_loader:
        img, label = data
        img = img.cuda()
        label = label.cuda()
        output = net(img)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        writer.add_scalar('train_loss', loss.item(), train_step)
        print("第{}次训练完成, 损失函数值为{}".format(train_step, loss.item()))

    with torch.no_grad():
        for test_data in test_data_loader:
            test_img, test_label = test_data
            test_img = test_img.cuda()
            test_label = test_label.cuda()
            test_output = net(test_img)
            loss = loss_fn(test_output, test_label)
            test_step += 1
            test_loss += loss.item()

        writer.add_scalar('test_loss', test_loss, test_step)
        print("第{}次测试完成, 测试集损失函数值{}".format(test_step, test_loss))
        print("==========================================================")

writer.close()
torch.save(net.state_dict(), f=net_save_path)
print("模型已保存")
with open(os.path.join(tensorboard_path, 'information'), 'w') as f:
    f.write('epochs = {}\n'
            'learn_rate = {}\n'
            'loss_fn = {}\n'
            'optimizer = {}\n'
            'len(Data_Set.train_dataset)={}\n'
            'len(Data_Set.test_dataset)={}\n'.format(epochs,
                                                     learn_rate,
                                                     loss_fn,
                                                     optimizer,
                                                     len(Data_Set.train_dataset),
                                                     len(Data_Set.test_dataset)))
print("模型信息已保存")

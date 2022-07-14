import Net
import Data_Set
from torch.utils.data import DataLoader


train_data_loader = DataLoader(Data_Set.train_dataset, batch_size=16, shuffle=True)
test_data_loader = DataLoader(Data_Set.test_dataset, batch_size=16, shuffle=True)

print("训练集长度为：".format(len(train_data_loader)))
print("测试集长度为：".format(len(test_data_loader)))

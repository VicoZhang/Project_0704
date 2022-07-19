# Project_0704

## 简介 
本文件的目的是将PsCAD中产生的*.out文件转换成numpy数据格式，并采用面对对象思路，将一次仿真设定为一个对象。最后以生产灰度图的形式举例应用。
文件中的其他是配套的仿真、神经网络等。总的来说还是一次完整的CNN故障识别项目，但这次项目的核心关键在于对数据的处理，可以为后续的操作打造一个基础。

## 文件组织
Data_Run.py -> 封装好的对象模型
Read_From_out.py -> 构建底层函数

## 使用方式
```
import Data_Run
import Read_From_ou

data = Data
print(dir(data))
```

## 其他信息
v1.0 2022/7/11 上传文件

author：Vico Zhang
更多信息，参见[Notion VicoZhang](https://www.notion.so/Project-0704-c718f4b8fb4b46e69debeb0244d0d4bf)

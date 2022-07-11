# Project_0704

## 简介 
本文件的目的是将PsCAD中产生的*.out文件转换成numpy数据格式，并采用面对对象思路，将一次仿真设定为一个对象。最后以生产灰度图的形式举例应用。

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
更多信息，参见[Notion VicoZhang](https://www.notion.so/Program-for-turn-out-to-ndarray-0a3e38dad9a54de79b55f27f0afea974)

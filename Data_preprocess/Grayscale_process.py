"""
encoding = 'utf-8'
author: Vico Zhang
此文件生成灰度图，调用 Data_Run.py
More information: https://github.com/VicoZhang/Project_0704.git
"""

import Data_Run
import numpy as np

for run in range(792):
    gray_scale = Data_Run.Data('../Simulation/Project_220704.gf42', run)
    temp = np.stack((gray_scale.data['Ia_1'],
                     gray_scale.data['Ib_1'],
                     gray_scale.data['Ic_1'])).flatten('F')
    img = gray_scale.generate_grayscale(temp, 23)

print("灰度图生成完成！")

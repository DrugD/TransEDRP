# 导入第三方包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import torch

asd = torch.load('/home/lk/project/drug/GraphDRP_go/graphDRP0921/data/processed/CTRPv1_train_normal.pt')

import pdb

# pdb.set_trace()

x = asd[0].y
# 正态分布图
plt.hist(x, # 绘图数据
        bins = np.arange(x.min(),x.max(),5), # 指定直方图的组距
        normed = True, # 设置为频率直方图
        # color = 'blue', # 指定填充色
        edgecolor = 'k') # 指定直方图的边界色

# 设置坐标轴标签和标题
# plt.title('乘客年龄直方图')
# plt.xlabel('年龄')
# plt.ylabel('频率')

# 生成正态曲线的数据
x1 = np.linspace(x.min(), x.max(), 1000)
normal = norm.pdf(x1, x.mean(), x.std())
# 绘制正态分布曲线
line1, = plt.plot(x1,normal,'r-', linewidth = 2) 

# 生成核密度曲线的数据
kde = mlab.GaussianKDE(x)
x2 = np.linspace(x.min(), x.max(), 1000)
# 绘制
line2, = plt.plot(x2,kde(x2),'g-', linewidth = 2)

# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')

# 显示图例
# plt.legend([line1, line2],['正态分布曲线','核密度曲线'],loc='best')
# 显示图形
plt.savefig('/home/lk/project/drug/dataset_DRP/img/hist_CTRPv1__.png')

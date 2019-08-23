#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 上午10:44

import torch

# 直接赋值索引
a = torch.rand(4,3,28,28)
# 同样的数据可以有不同的理解,理解剩余维度,作为基本单位
# torch.Size([3,28,28]), torch.Size([28,28]), torch.Size([])
print(a[0].shape, a[0,0].shape, a[0,0,2,4].shape)

# 单:索引
# 第0维->2,第1维-1->,第2维[1,10),第3维全部
# 注意负数索引是从右往左,最后一个为-1,-1:相当于只取最后一维
b = a[:2,-1:,1:10,:]

# 双:索引,第2个:表示间隔
# 第0和1维全部,第2维[1,10)间隔取值,第3维间隔取值
c = a[:,:,1:10:2,::2]

# 采样指定索引数据
# index_select(dim, index),index必须转化为tensor
d = a.index_select(0, [0, 2]) # 选择第0和2张图片的数据

# ...代替维度索引
e = a[...] # 复制一遍
f = a[0,...] # 第0维0,其余维度全部
g = a[:,1,...] # 第0维全部,第1维1,其余维度全部

# 使用掩码mask索引
h = torch.randn(3,4)
# mask类型为torch.uint8,也即ByteTensor类型,shape与h一致
mask = h.ge(0.5) # 大于等于0.5的元素返回1,否则返回0
i = torch.masked_select(h, mask) # 会将满足条件元素展平,返回一维张量

# 使用take函数全局一维索引
j = torch.take(h, torch.tensor([0,2,6,10]))

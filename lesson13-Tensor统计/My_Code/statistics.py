#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 下午5:23

import torch

# norm
# 向量范数与矩阵范数定义不同
a = torch.full([8], 1)
b = a.view(2,4)
c = a.view(2,2,2)
print(a, b, c, sep='\n')
# 3个张量的l1范数相同
print(a.norm(1), b.norm(1), c.norm(1))
# 可以求指定维度上数据的范数,结果张量形状为去掉该维度的形状
# 先推理出结果张量形状,再求结果张量与原张量对应位置数据的范数
print(b.norm(1, dim=1), b.norm(2, dim=1), sep='\n')
print(c.norm(1, dim=1), c.norm(2, dim=1), sep='\n')

# mean / sum / max / min / prod(连乘)
d = torch.arange(8).view(2,2,2).float()
print(d)
print(d.mean(), d.min(), d.max(), d.sum(), d.prod())

# argmin / argmax
# 若不指定维度,返回的是全局最大值对应的展平的全局索引
# 若指定维度dim,返回的dim上元素的最值索引
# 返回tensor的维度为去掉dim的维度,再求结果张量与原张量对应位置数据的最值索引
print(d.argmax(), d.argmin(), d.argmax(0), d.argmin(1))

# dim / keepdim
# dim在指定维度上操作,难点是该如何找到该维度上的数据
# 一般在该维度处理结果为1,会将该维度消除,keepdim=True是为了保证shape不变
e = torch.randn([4,10])
# 返回值包括两组数据([4],[4]),第2维度最大数据值,及其索引(就是argmax())
max_and_idx1 = e.max(dim=1)
max_and_idx2 = e.max(dim=1,keepdim=True)



# top_k / k_th
# topk()比max()更强大,可以返回某个维度上最大的几个值及索引
e.topk(3,dim=1) # 1维上最大的3个值及其索引([4,3],[4,3])
e.topk(3,dim=1,largest=False) # 求1维最小的3个值及其索引
e.kthvalue(8,dim=1) # 返回1维上第8小的值及其索引


# compare
# 以下比较符是逐元素比较的
# >, >=, <, <=, !=, ==
# (torch.gt, torch.ge, torch.lt, torch.le, torch.ne, torch.eq)
print(torch.eq(e,0))

# 张量相等判断torch.equal()
torch.equal(a,a)
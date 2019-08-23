#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 下午8:14

import torch


# torch.where(condition,x,y) -> Tensor
# 可以高效并行计算,判断赋值
# 要求做判断的张量与做选择的两个张量维度相同
# 张量每个元素做判断,决定赋值x还是y对应位置元素
# True选x,False选y
a = torch.rand(2,2)
b = torch.zeros(2,2)
c = torch.ones(2,2)
d = torch.where(a>0.5, b, c)
print(a,d,sep='\n')


# torch.gather(input,dim,index,out=None) -> Tensor
# 可以高效并行运算,快速查表映射
prob = torch.randn(4,10)
idx = prob.topk(k=3,dim=1)[1]
label = torch.arange(10)+100
result = torch.gather(label.expand(4,10), dim=1, index=idx.long())

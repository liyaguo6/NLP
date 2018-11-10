# import pandas as pd
# import numpy as np
# from numpy import linalg
# a1 = [0,0,0]
# b2 = [2,3,4]
# v1 = np.array(a1)
# v2 = np.array(b2).T
# denom = linalg.norm(v1) * linalg.norm(v2)
# cos = sum(v2*v1)
#
# print(cos/denom)
# 为什么天体都是球形的?
# 恒星、行星和其他天体之所以都是球形，而不是正方形或是别的什么奇形怪状的样子，完全是万有引力作用的结果.
# 这时，分散的物质云在引力的作用下逐渐聚合在了一起，同时由于其本身的非均一性和某些外力的作用而开始自转，于是便形成了一个大致的（不是完美球形的）旋转天体
import heapq
# list2 =[123,7,4,67,45,34,12,45,0]
# temp = map(list2.index,heapq.nlargest(3,list2))
list2 = ['qw','34']
str = ','.join(list2)
print(str)
# print(sorted(list(temp)))
# dict1 ={"a":2,"b":4,"d":3,"c":23}
# print(list(dict1.values()))
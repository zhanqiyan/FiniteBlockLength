import matplotlib.pyplot as plt
import numpy as np
from Optimizer import Optimizer
from Bisection import Bisection
opt = Optimizer()
bis = Bisection()
# B = 30000
# SNR = 20
# theta = np.arange(0.00001, 0.012, 0.0005)
# D_t=0.0025
# theta_bar = bis.calculate_theta_bar_by_bisection(B,20,1e-8,0.2)
# res=[]
# for t in theta:
#     res.append(opt.f_function(t,D_t,B,SNR))
#
# plt.plot(theta,res,'b-.')
# plt.xlim([0, 0.0115])
# plt.ylim([0, 1])
# plt.grid(ls='--')  # 设置网格
# plt.rcParams['font.sans-serif'] = ['SimHei'] #STSong
# x_ticks = np.linspace(0, 0.0115, 6)  # 产生区间在-5至4间的10个均匀数值
# plt.xticks(x_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
# y_ticks = np.linspace(0, 1, 6)  # 产生区间在-5至4间的10个均匀数值
# plt.yticks(y_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
# plt.xlabel(r'QoS指数$\theta_k$', fontsize=18)  # 设置 x轴标签
# plt.ylabel("目标函数值", fontsize=18)  # 设置 y轴标签
# plt.show()

B = 30000
SNR = 20
theta = 0.005
D_t=np.arange(0.000001,0.005,0.000005)
res=[]
for t in D_t:
    res.append(opt.f_function(theta,t,B,SNR))

plt.plot(D_t,res,'b-.')
plt.xlim([0, 0.005])
plt.ylim([0.5, 1])
plt.grid(ls='--')  # 设置网格
plt.rcParams['font.sans-serif'] = ['SimHei']
x_ticks = np.linspace(0, 0.005, 6)  # 产生区间在-5至4间的10个均匀数值
plt.xticks(x_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
y_ticks = np.linspace(0.5, 1, 6)  # 产生区间在-5至4间的10个均匀数值
plt.yticks(y_ticks, fontsize=20)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
plt.xlabel(r'传输延迟$D_k^t$(s)', fontsize=18)  # 设置 x轴标签

plt.ylabel("目标函数值", fontsize=18)  # 设置 y轴标签
plt.show()


# 论文修改页数：
# 1、英文摘要
# 2、第3页，修改了可观测性小节，增加了参考文献
# 3、第45页，修改了图3-1，有效容量添加了单位
# 4、第63页，修改了图4-3，传输延迟添加了单位
# 5、第66页，修改了图4-4，有效容量添加了单位


# 论文修改情况说明:
# 1.	将英文摘要中的paper修正为了thesis。
# 2.	可观测性是研究重点，但“1.2.1 可观测性”的文献综述过少，补充了若干文献来阐明可观测性研究现状。  见第3页
# 3.	对文章中的一些错别字和语句不通顺的地方进行了修改。
# 4.	论文重点关注通信时延指标下的可观测性指标，但是对两者之间的联系没有描述清楚，在文章中的3.2.1小节添加了相关内容。 见27和28页
# 5.	论文格式需要修正，对不清晰的图进行修改，修改了图2-5、图2-6、图2-7、图2-8、图4-1，对没有标注单位的图添加了单位，主要修改了图3-1、图4-3、图4-4。






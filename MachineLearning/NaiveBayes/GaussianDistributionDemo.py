import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
# 设置随机种子以获得可重复的结果  
np.random.seed(0)  
  
# 生成高斯分布的数据集  
# 假设均值为0，标准差为1  
data = np.random.normal(loc=0, scale=1, size=1000)  
  
# 计算用于拟合曲线的参数（实际上，因为我们知道这是标准正态分布，所以均值=0，标准差=1）  
# 但这里我们使用数据来计算，以展示过程  
mean, std = np.mean(data), np.std(data)  
  
# 创建x轴的数据点以绘制拟合曲线  
x = np.linspace(mean - 3*std, mean + 3*std, 1000)  
  
# 绘制直方图  
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Histogram of data')  
  
# 绘制拟合的高斯曲线  
plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='Fitted Gaussian curve')  
  
# 添加图例  
plt.legend(loc='upper left')  
  
# 设置标题和轴标签  
plt.title('Gaussian Distribution Example')  
plt.xlabel('Value')  
plt.ylabel('Probability density')  
  
# 显示图形  
plt.show()
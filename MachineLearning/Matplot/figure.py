import matplotlib.pyplot as plt
import numpy as np

# 1. add title
plt.title('This is Title')
# 2.add X axis name
plt.xlabel('age')
# 3. add scope of X axis
plt.xlim((0, 1))
# 4. add scale num on X axis
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# 2.add Y axis name
plt.ylabel('money')
# 3. add scope of Y axis
plt.ylim((0, 1))
# 4. add scale num on Y axis
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

# 8. load data
data = np.arange(0.1, 1.1, 0.01)
plt.plot(data, data ** 2, label='$y=x^2$')  # $ 支持 latex 语法
plt.plot(data, data ** 4, label='y=x^4')
# 9. 添加图例 legend
plt.legend(loc='best')
# plt.legend(["red","Blue"])
# plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')
# 10. save pic
plt.savefig('y=1.png')
plt.savefig('y=1.pdf')
plt.savefig('y=1.jpg')

plt.show()

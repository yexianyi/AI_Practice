import matplotlib.pyplot as plt  
import numpy as np  
import mplcursors  
  
# 生成一些数据  
x = np.linspace(0, 10, 100)  
y = np.sin(x)  
  
# 绘制数据  
plt.plot(x, y, 'o-')  # 使用 'o-' 以便同时显示点和线  
  
# 启用 mplcursors  
cursor = mplcursors.cursor(hover=True)  
  
# 自定义悬停时显示的信息  
cursor.connect("add", lambda sel: sel.annotation.set_text(f'x={sel.target[0]:.2f}, y={sel.target[1]:.2f}'))  
  
plt.show()
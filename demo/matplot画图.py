import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print(matplotlib.__version__)

xpoints = np.array([0,6])
ypoints = np.array([0,100])

print(xpoints)

plt.plot(xpoints, ypoints,marker = 'o')
plt.show()
plt.plot([1,2,6,8], [3,8,1,10])
plt.show()

#饼图
plt.pie([1,2,3,4])
plt.title("饼图") # 设置标题

plt.show()
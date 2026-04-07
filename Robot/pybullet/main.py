import pybullet as p
import pybullet_data
import time

# 连接物理引擎
physicsClient = p.connect(p.GUI) # 使用GUI窗口可视化
# p.connect(p.DIRECT) # 或无头模式

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 载入地面模型
p.loadURDF("plane.urdf")

# 载入一个简单的机器人模型，如KUKA机械臂
robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])

# 进行简单的仿真步骤
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.) # 模拟实时

# 断开连接
p.disconnect()
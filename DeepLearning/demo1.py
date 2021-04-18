'''
感知器
单层感知器：XOR 1 ^ 1 = 0， 1 ^ 0 = 1， 0 ^ 1 = 1， 0 ^ 10 = 1
y = W*X + b
1.  输入值 X = [x0, x1, x2 ...]
    定义W（W和X有关系） ---> 定义为0
    偏置值： b ---> 定义为0
    激活函数：f用来判断y
2. 预测函数，输入x，返回预测值
3. 更新W和b, 根据预测值和真实值的误差，来改变W和b
4. 训练一次的效果：获得预测值，重复步骤3
5. 训练多次的效果，循环多个单词
6. 定义激活函数
7. 数据集
8. 训练方法
9. 预测
'''

import numpy as np


# 定义感知器
class Perceptron:
    # 定义构造器： 输入的参数个数，激活函数的定义
    def __init__(self, input_num, activation):
        # 定义激活函数
        self.activation = activation
        # 定义偏置值
        self.bias = 0
        # 定义W 和X有关，有多少个X就有多少个W
        self.weight = [0.0 for _ in range(input_num)]

    # 输出W和b的改变
    def __str__(self):
        return "weight: %s, bias: %s"%(self.weight, self.bias)

    # 预测函数：
    def predict(self, input_xs):
        # y = w0*x0 + w1*x1 +b
        return self.activation(np.sum(np.multiply(self.weight, input_xs)) + self.bias)

    '''
    更新W和b值:
    input_xs: X
    out: 预测值
    label:真实值
    reta：斜率
    '''
    def update(self, input_xs, out, label, reta):
        # 根据真实值和预测值，来修改W和b
        delta = label - out
        # 修改W和b
        self.weight = list(map(lambda x, w: w + delta*reta*x, input_xs, self.weight))
        self.bias += delta*reta

    '''
    训练一次
    input_xs:输入值
    label:真实值y
    reta：学习率
    '''
    def one_train(self, input_xs, labels, reta): #[[1,1], [1,0], [0,1]] -> [0, 1, 1]
        sample = zip(input_xs, labels) # ([x0, x1], y)
        for(input_x, label) in sample:
            # 预测值
            out = self.predict(input_x)
            # 根据误差更新W和b
            self.update(input_x, out, label, reta)

    '''
    训练多次
    input_xs:输入值
    label:真实值y
    reta：学习率
    iter_num: 迭代次数
    '''
    def train(self, input_xs, labels, iter_num, reta):
        for i in range(iter_num):
            self.one_train(input_xs, labels, reta)

# 自定义激活函数
def f(y):
    return 1 if y > 0 else 0

# 数据集的生成
def data_set():
    input_xs = [[1,1], [1,0], [0,0], [0,1]]
    labels = [0, 1, 0, 1]
    return input_xs, labels

# 训练
def create_perceptron():
    p2 = Perceptron(2, f)
    input_xs, labels = data_set()
    p2.train(input_xs, labels, 20, 0.01)
    return p2

if __name__ == "__main__":
    p = create_perceptron()
    print("[1,1]:", p.predict([1, 1]))
    print("[1,0]:", p.predict([1, 0]))
    print("[0,1]:", p.predict([0, 1]))
    print("[0,0]:", p.predict([0, 0]))

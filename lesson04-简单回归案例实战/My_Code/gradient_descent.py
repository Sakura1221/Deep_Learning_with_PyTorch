#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/21 下午7:41

"""
简单回归案例
手写随机梯度下降
"""

import numpy as np

# 计算loss
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

#　计算梯度
def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - learning_rate * b_gradient
    new_w = w_current - learning_rate * w_gradient
    return [new_b, new_w]

# 梯度下降，需要输入训练数据,初始化权重,学习率,迭代次数
def gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations):
    b = initial_b
    w = initial_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learning_rate)
    return [b, w]

#　主函数,准备并传入数据,设定参数,输出结果
def run():
    # np.genfromtxt读入.csv格式数据,一行为一row,指定分隔符得到col
    # 返回array
    # x用作输入,y用作计算loss的target
    points = np.genfromtxt("data.csv", delimiter=",")
    # 学习率过大甚至会溢出(nan)报错
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format
          (initial_b, initial_w, compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format
          (num_iterations, b, w,
           compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()
# -*- coding:utf8 -*-
# @TIME : 2018/4/28 上午6:15
# @Author : Allen
# @File : tensorflow_11_建造神经网络.py

#1）构造定义一个神经层函数
#2）【导入数据】x_data, y_data, noise,定义placeholder()为神经网络需要的输入
#3）【搭建网络】，定义隐藏层，l1 = add_layer()利用之前的add_layer()函数+自带激励函数tf.nn.relu
#4) 定义输出层：prediction = add_layer()
#5) 计算loss值
#6）定义训练方式GradientDescentOptimizer
#7) 使用变量时候，都要对它初始化 init=tf.global_variables_initializer()
#8）定义Session, 用Session执行init初始化步骤
#9）【开始训练】

import tensorflow as tf
import numpy as np

#定义一个神经层函数
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weight) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#【导入数据】
#np.newaxis用法：https://blog.csdn.net/lanchunhui/article/details/49725065
#np.linspace用法：https://blog.csdn.net/you_are_my_dream/article/details/53493752
#np.random.normal用法：https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis] #相当于在array中取列
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#定义神经网络的输入，用tf.placeholder()
#TensorFlow 辨异 —— tf.placeholder 与 tf.Variable: https://blog.csdn.net/lanchunhui/article/details/61712830
xs = tf.placeholder(tf.float32, [None, 1]) #类似与形参
ys = tf.placeholder(tf.float32, [None, 1])

#接下来，我们就可以开始定义神经层了。 通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元； 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

#【搭建网络】
#隐藏层: 利用之前的add_layer()函数, 激励函数用tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function= tf.nn.relu)
#输出层: 是隐藏层的输出
prediction = add_layer(l1, 10, 1, activation_function=None)
#预测值与真实值的误差，对二者差对平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
#最小化loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#使用变量时，初始化, 用Session执行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#【训练】
# 比如这里，我们让机器学习2000次。机器学习的内容是train_step, 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。 (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
for i in range(2000):
    sess.run(train_step, feed_dict={xs: x_data, ys:y_data})
    if i % 200 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))

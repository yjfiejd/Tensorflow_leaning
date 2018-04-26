# -*- coding:utf8 -*-
# @TIME : 2018/4/26 上午6:51
# @Author : Allen
# @File : tensorflow_5.py

import tensorflow as tf
import pandas as pd
import numpy as np

import pip
print(pip.__version__)

#生成随机数据，float32, 100个
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

#构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

#最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#启动图
sess = tf.Session()
sess.run(init)

#拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


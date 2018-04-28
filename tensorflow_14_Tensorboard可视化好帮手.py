# -*- coding:utf8 -*-
# @TIME : 2018/4/28 下午6:51
# @Author : Allen
# @File : tensorflow_14_Tensorboard可视化好帮手.py

#可以看到整个神经网络的框架
#先导入之前的代码


import tensorflow as tf
import numpy as np

#定义一个神经层函数
def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weight), biases)
        #这里不用画出来，因为后面调用列激励函数，tensorflow会默认添加
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input') #类似与形参,可以输入名字
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add hidden layer(每加一个层，就会出现定义中的图示)
l1 = add_layer(xs, 1, 10, activation_function= tf.nn.relu)

#add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#the error
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name='Loss_NAME')
#最小化loss
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#注意初始化的方式变列
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("Tensorflow learning/", sess.graph)
sess.run(init)

#最后在你的terminal（终端）中 ，使用以下命令
#1） cd ~/desktop
#2） tensorboard --logdir='Tensorflow learning/'
#3） http://wubindeMacBook-Air.local:6006 注意用谷歌浏览器打开，注意网址换成http://localhost:6006
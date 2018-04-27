# -*- coding:utf8 -*-
# @TIME : 2018/4/27 下午10:53
# @Author : Allen
# @File : tensorflow_10_def add_layer().py

#首先，我们需要导入tensorflow模块。
import  tensorflow as tf

#然后定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。
def add_layer(inputs, in_size, out_size, activation_function=None):
    #接下来，我们开始定义weights和biases
    #因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    Weights = tf.variable(tf.random_normal([in_size, out_size]))

   #因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    biases =  tf.variable(tf.zeros([1, out_size]) + 0.1)

    #下面，我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    #当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    ##最后，返回输出，添加一个神经层的函数——def add_layer()就定义好了
    return outputs
# -*- coding:utf8 -*-
# @TIME : 2018/4/29 下午5:22
# @Author : Allen
# @File : tensorflow_15_可视化帮手2.py

#https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-2-tensorboard2/


import tensorflow as tf
import numpy as np


#定义一个神经层函数 + 在layer中为Weights, biases设置变化图表
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer #定义新的变量
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
            #【我们层中的Weights设置变化图】, tensorflow中提供了tf.histogram_summary()方法,用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量
            tf.summary.histogram(layer_name +'/weight', Weight)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            #【同样对biases进行绘制图标】
            tf.summary.histogram(layer_name +'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weight), biases)
        #这里不用画出来，因为后面调用列激励函数，tensorflow会默认添加
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name +'/outputs', outputs)
        return outputs
        #修改之后的名称会显示在每个tensorboard中每个图表的上方显示, 如下图所示:

#make up some real data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis] #相当于在array中取列
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input') #类似与形参,可以输入名字
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add hidden layer(每加一个层，就会出现定义中的图示)
l1 = add_layer(xs, 1, 10, n_layer = 1, activation_function= tf.nn.relu)

#add output layer
prediction = add_layer(l1, 10, 1, n_layer = 2, activation_function=None)

##【设置loss对变化图】
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name='Loss_NAME')
# 看图栏有4栏，刚才都在GRAPHS中，这个loss在EVENTS中,
# loss是在tesnorBorad 的event下面的, 这是由于我们使用的是tf.scalar_summary() 方法.
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#【给所有对训练图合并】训练图合并
sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)



#训练数据：假定给出列x_data, y_data，并且训练100次
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    #以上这些仅仅可以记录很绘制出训练的图表， 但是不会记录训练的数据。 为了较为直观显示训练过程中每个参数的变化，我们每隔上50次就记录一次结果 , 同时我们也应注意, merged 也是需要run 才能发挥作用的,所以在for循环中写下：
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)

#在tensorboard中查看效果
#最后在你的terminal（终端）中 ，使用以下命令
#1） cd ~/desktop
#2） tensorboard --logdir='Tensorflow learning/'
#3） http://wubindeMacBook-Air.local:6006 注意用谷歌浏览器打开，注意网址换成http://localhost:6006
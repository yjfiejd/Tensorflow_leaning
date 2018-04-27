# -*- coding:utf8 -*-
# @TIME : 2018/4/27 下午9:29
# @Author : Allen
# @File : tensorflow_08_placeholder传入值.py

import tensorflow as tf

#在tensorflow 中需要定义placeholder的type 一般为float32形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

#mul = multiply 是将input1 和 input2 做乘法运算,输出为output
output = tf.multiply(input1, input2)

#接下来
# , 传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={} 并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的。
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))


# -*- coding:utf8 -*-
# @TIME : 2018/4/27 下午9:05
# @Author : Allen
# @File : tensorflow_07_变量.py

import tensorflow as tf

state = tf.Variable(0, name='counter') #可以给变量初始值，名字
#print(state.name)

#定义常量one
one = tf.constant(1)

#定义加法（注意：此步并没有直接计算）
new_value = tf.add(state, one)

#将state更新为 new_value
update = tf.assign(state, new_value)

#如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

#使用Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


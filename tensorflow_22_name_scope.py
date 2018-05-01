# -*- coding:utf8 -*-
# @TIME : 2018/4/30 下午7:42
# @Author : Allen
# @File : tensorflow_22_name_scope.py

from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(1)

# with tf.name_scope("a_name_scope"):
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
#     var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
#     var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
#     var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var1.name)
#     print(sess.run(var1))
#
#     print(var2.name)
#     print(sess.run(var2))
#
#     print(var21.name)
#     print(sess.run(var21))
#
#     print(var22.name)
#     print(sess.run(var22))

# 使用tf.name_scope("a_name_scope")输出结果
# var1:0     ###tf.name_socpe("a_name_scope")对tf.get_variable是无效，名字前不会出现"a_name_scope"
# [1.]
# a_name_scope/var2:0    ### 虽然我们后面3个变量的名字都是name='var2',注意tf.variable创建变量的时候，会先检查是否已经创建，如果有，则在名字后面加上_1
# [2.]
# a_name_scope/var2_1:0
# [2.1]
# a_name_scope/var2_2:0
# [2.2]

with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    ##这里想利用tf.Variable(name='var4')重复调用上面的那个变量var4，但是我们知道,下面这行中，name虽然一样，但是实际中会变成var4_1
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

    ##试试tf.get_variable(name='var3')重复调用呢
    #scope.reuse_variables() 实验证明需要加上这句才不会报错
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)
    print(sess.run(var3))

    print(var4.name)
    print(sess.run(var4))

    print(var4_reuse.name)
    print(sess.run(var4_reuse))

    #试试tf.get_variable(name='var3')重复调用呢
    #这里出现报错：ValueError: Variable a_variable_scope/var3 already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
    #解决方法：在重复调用前，强调一下，scope.reuse_variables()
    print(var3_reuse.name)
    print(sess.run(var3_reuse))


#输出结果： 使用tf.variable_scope("a_variable_scope")输出的结果：

#a_variable_scope/var3:0
#[3.]
#a_variable_scope/var4:0
#[4.]
#a_variable_scope/var4_1:0  #注意看这里，tf.Variable(...)本来想重复调用上方的var4,实际中变量名字自动变成列var4_1
#[4.]

#我们继续实验，如果使用tf.get_variable，试试重复调用呢

#输出结果： 使用tf.variable_scope("a_variable_scope") + 加上强调：scope.reuse_variables() 输出的结果
# a_variable_scope/var3:0  看第一次使用var3
# [3.]
# a_variable_scope/var4:0
# [4.]
# a_variable_scope/var4_1:0
# [4.]
# a_variable_scope/var3:0   看第二次使用var3，ok
# [3.]

#这里需要注意，第二次get_variable(name='var3')调用的var3, 与第一次调用的var3其实是同一个变量，



#【如何运用呢】：RNN里面重复利用的机制中使用，Train中  test中
# -*- coding:utf8 -*-
# @TIME : 2018/4/30 上午11:34
# @Author : Allen
# @File : RNN_lstm 循环神经网络.py

#参考学习：
# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/
# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

#这次我们会使用 RNN 来进行分类的训练 (Classification). 会继续使用到手写数字 MNIST 数据集. 让 RNN 从每张图片的第一行像素读到最后一行, 然后再进行分类判断. 接下来我们导入 MNIST 数据并确定 RNN 的各种参数(hyper-parameters):
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

#set random seed for comparing the two result calculations
tf.set_random_seed(1)

#导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#hyperparameters
lr = 0.001 #学习率
training_iters = 100000  #循环次数上限
batch_size = 128
n_inputs = 28 #MNIST data input(img shape:28*28) 每一行28个像素点，28列
n_steps = 28 #time steps 每次input一行，一共28行
n_hidden_units = 128 #neurons in hidden layer
n_classes = 10 #MNIST classes(0-9 digits) 把像书店分成10个类别

#######################################################
#接着定义 x, y 的 placeholder 和 weights, biases 的初始状况.
#tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) #设置形参传递
y = tf.placeholder(tf.float32, [None, n_classes])

#Define weights, input -> intput_hidden_layer1 -> RNN cell -> output_hidden_layer2 -> output_layer
#对weights biases 初始值定义
weights = {
    #(28, 128)
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    #(128, )
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    #(10,)
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}


# 定义RNN主体结构：共3个部分（input_layer, cell, output_layer）
def RNN(X, weights, biases):
    #hidden layer for input to cell
    ###############################

    #原始对X是3维数据， 需要把它变为2维数据才能使用weights对矩阵乘法
    #transpose the inputs shape from X， ==> (128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    #X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 格式重新变为3维数据 X_in ==> (128batch * 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    #cell 中对计算：使用 tf.nn.dynamic_rnn(cell, inputs) (推荐). 这次的练习将使用这种方式.
    ###############################

    #basic LSTM cell:
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    #lstm cell is devide into two parts(c_state, m_state) 主线剧情，分线剧情
    init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32) #初始化全为0 state
    #如果使用tf.nn.dynamic_rnn(cell, inputs), 我们要确定 inputs 的格式. tf.nn.dynamic_rnn 中的 time_major 参数会针对不同 inputs 格式有不同的值.
    #如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    #如果 inputs 为 (steps, batches, inputs) ==> time_major=True;

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)


    #hidden layer for output as the final results
    ##################################
    #方法1：直接调用final_state 中的 h_state (final_state[1]) 来进行运算:
    results = tf.matmul(states[1], weights['out']) + biases['out']

    #方法2：
    #outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
    #results = tf.matmul(outputs[-1], weights['out'] + biases['out'])

    return results

#定义好了 RNN 主体结构后, 我们就可以来计算 cost 和 train_op:
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels =  y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 训练RNN
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1


#结果如下：
#2018-04-30 14:05:50.942937: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
#0.265625
#0.7265625
#0.828125
#0.8828125
#0.84375
#0.859375
#0.8984375
#0.890625
#0.84375
#0.90625
#0.921875
#0.90625
#0.9140625
#0.9140625
#0.9375
#0.9609375
#0.953125
#0.921875
#0.9453125
#0.96875
#0.9375
#0.9609375
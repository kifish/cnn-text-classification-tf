import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        #这里的输入是句子各词的索引？
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #定义一个operation，名称input_x,利用参数sequence_length，None表示样本数不定，
        #训练和验证的时候，batchsize可能不一样
        #数据类型int32，（样本数*句子长度）的tensor，每个元素为一个单词

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #这个placeholder的数据输入类型为float，（样本数*类别）的tensor
        #placeholder表示图的一个操作或者节点，用来喂数据，进行name命名方便可视化
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #初始化
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # 指定运算结构的运行位置在cpu非gpu,因为"embedding"无法运行在gpu
        # 通过tf.name_scope指定"embedding"
        with tf.device('/cpu:0'), tf.name_scope("embedding"): ##封装了一个叫做“embedding'的模块，使用设备cpu，模块里3个operation
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W") ##operation1，一个（词典长度*embedsize）tensor，作为W，也就是最后的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #onehot矩阵相乘，相当于查表操作（表为W）
            #operation2，input_x的tensor维度为[none，seq_len],那么这个操作的输出为none*seq_len*em_size
            #em_size相当于单个词的特征维度

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #增加一个维度，变成，batch_size*seq_len*em_size*channel(=1)的4维tensor，符合图像的习惯
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes): ##比如（0，3），（1，4），（2，5）
            with tf.name_scope("conv-maxpool-%s" % filter_size):#每一尺寸的filter
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #operation，卷积核参数，高（即行数）*宽（列宽）*通道（channel）*该尺寸的卷积个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #operation，名称”W“，变量维度为num_filters的tensor
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],##样本，height，width，channel移动距离
                    padding="VALID",# 这里不需要padding
                    name="conv")
                # operation，卷积操作，名称”conv“，与W系数相乘得到一个矩阵
                # Apply nonlinearity
                # 可以理解为,正面或者负面评价有一些标志词汇,这些词汇概率被增强，即一旦出现这些词汇,倾向性分类进正或负面评价,
                # 该激励函数可加快学习进度，增加稀疏性,因为让确定的事情更确定,噪声的影响就降到了最低。
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                #同一尺寸的每个卷积核对应的卷积结果经pooling处理一个样本后会得到一个值。这里维度如batchsize*1*1*卷积核个数
                #在本例中，同一尺寸的卷积核共有三个，append3次


        # Combine all the pooled features
        #为简化，假设num_filters为3.实际上train.py中写的是128
        num_filters_total = len(filter_sizes) * num_filters
        #operation，卷积核的不同尺寸个数与每种尺寸的卷积核个数
        self.h_pool = tf.concat(pooled_outputs, 3)
        #operation，将outpus在第4个维度上拼接，如本来是同一尺寸：128*1*1*3，尺寸个数有三个，拼接后为128*1*1*9的tensor
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #operation，结果reshape为128*9的tensor

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            #输出为dropout过后的128*9的tensor

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #operation，系数tensor，如9*2，9个features分2类，
            #名称为"W"，注意这里用的是get_variables
            #与之前的查表W无关
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #operation2,偏置tensor，如[0.1 0.1](可能更新为[0.11 0.27])，名称"b"
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #实际上这里b可能不需要正则化
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #operation，计算预测值，输出最大值的索引，这里只有2类，所以应该是0或者1，名称”predictions“
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #operation，定义losses，交叉熵，如果是一个batch，那么是一个长度为batchsize的tensor？
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #operation，计算一个batch的平均交叉熵，加上全连接层参数的正则

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #根据input_y和predictions是否相同，得到一个矩阵batchsize大小的tensor
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

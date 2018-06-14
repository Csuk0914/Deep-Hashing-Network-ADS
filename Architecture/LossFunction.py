# This file is for pairwise_cross_entropy_loss for DHN
# For pairwise_quantization_loss, we directly added it into train process
import tensorflow as tf

# 输入hash_code为fch层，oneHotLabel为标签矩阵，默认不对输出进行归一化
def pairwise_cross_entropy_loss(hash_code, oneHotLabel, alpha = 1., normed = False):
    # 获取相似度
    label_ip = tf.cast(
        tf.matmul(oneHotLabel, tf.transpose(oneHotLabel)), tf.float32)
    s = tf.clip_by_value(label_ip, 0.0, 1.0)

    # compute balance param
    # s_t \in {-1, 1}
    s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
    sum_1 = tf.reduce_sum(s)
    sum_all = tf.reduce_sum(tf.abs(s_t))
    balance_param = tf.add(tf.abs(tf.add(s, tf.constant(-1.0))),
                           tf.multiply(tf.div(sum_all, sum_1), s))

    if normed:
        # ip = tf.clip_by_value(tf.matmul(u, tf.transpose(u)), -1.5e1, 1.5e1)
        ip_1 = tf.matmul(hash_code, tf.transpose(hash_code))

        def reduce_shaper(t):
            return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(hash_code)),
                                  reduce_shaper(tf.square(hash_code)), transpose_b=True))
        ip = tf.div(ip_1, mod_1)
    else:
        ip = tf.clip_by_value(tf.matmul(hash_code, tf.transpose(hash_code)), -1.5e1, 1.5e1)

    ones = tf.ones([tf.shape(hash_code)[0], tf.shape(hash_code)[0]])
    return tf.reduce_mean(tf.multiply(tf.log(ones + tf.exp(alpha * ip)) - s * alpha * ip, balance_param))
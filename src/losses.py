import math
import tensorflow as tf


def arcface_loss(x, normx_cos, labels, m1, m2, m3, s):
    norm_x = tf.norm(x, axis=1, keepdims=True)
    cos_theta = normx_cos / norm_x
    theta = tf.acos(cos_theta)
    mask = tf.one_hot(labels, depth=normx_cos.shape[-1])
    zeros = tf.zeros_like(mask)
    cond = tf.where(tf.greater(theta * m1 + m3, math.pi), zeros, mask)
    cond = tf.cast(cond, dtype=tf.bool)
    m1_theta_plus_m3 = tf.where(cond, theta * m1 + m3, theta)
    cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
    prelogits = tf.where(cond, cos_m1_theta_plus_m3 - m2, cos_m1_theta_plus_m3) * s

    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # do softmax
    loss = cce(labels, prelogits)

    return loss
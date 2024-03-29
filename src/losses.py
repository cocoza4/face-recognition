import tensorflow as tf
import tensorflow.keras.backend as K


def arcface_loss(embeddings, weights, labels, n_classes, m1, m2, m3, s, reduction=tf.keras.losses.Reduction.AUTO):
    norm_embeddings = tf.nn.l2_normalize(embeddings, axis=1) * s
    norm_weights = tf.nn.l2_normalize(weights, axis=1)
    fc7 = tf.matmul(norm_embeddings, tf.transpose(norm_weights), name='cos_t')
    indices = tf.stack([tf.range(tf.shape(norm_embeddings)[0])[:, None], labels[:, None]], axis=-1)
    zy = tf.gather_nd(fc7, indices=indices)
    cos_t = zy / s
    cos_t = tf.clip_by_value(cos_t, -1.0+K.epsilon(), 1.0-K.epsilon()) # clip to prevent nan
    theta = tf.acos(cos_t)

    # m1 -> SphereFace
    # m2 -> ArcFace
    # m3 -> CosFace
    new_zy = (tf.cos(theta*m1 + m2) - m3) * s
    diff = new_zy - zy
    prelogits = fc7 + tf.one_hot(labels, n_classes) * diff
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)
    loss = cce(labels, prelogits)

    return loss

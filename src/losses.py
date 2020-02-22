import tensorflow as tf

def arcface_loss(norm_embeddings, weights, labels, n_classes, m1, m2, m3, s):
    # norm_embedding = tf.nn.l2_normalize(embeddings, axis=1) * s
    norm_embeddings *= s
    norm_weights = tf.nn.l2_normalize(weights, axis=1)
    fc7 = tf.matmul(norm_embeddings, tf.transpose(norm_weights), name='cos_t')
    indices = tf.stack([tf.range(norm_embeddings.shape[0])[:, None], labels[:, None]], axis=-1)
    zy = tf.gather_nd(fc7, indices=indices)
    cos_t = zy / s
    theta = tf.acos(cos_t)
    new_zy = (tf.cos(theta*m1 + m2) - m3) * s
    diff = new_zy - zy
    prelogits = fc7 + tf.one_hot(labels, n_classes) * diff
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cce(labels, prelogits)

    return loss

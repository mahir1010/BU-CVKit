import tensorflow as tf


def build_mpjpe_metric(per_kp=False):
    if not per_kp:
        @tf.function
        def mpjpe(y_t, y_p):
            y_t = tf.cast(y_t, dtype=tf.float32)
            y_p = tf.cast(y_p, dtype=tf.float32)
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1)), dtype=tf.float32)
            total = tf.reduce_sum(mask, axis=-1)
            l = tf.reduce_mean(tf.sqrt(1e-9 + tf.reduce_sum(tf.square(y_t - y_p), axis=-1)), axis=-1) * mask
            l = tf.reduce_sum(l, axis=-1) / total
            return tf.reduce_mean(l)

        mpjpe._name = 'mpjpe'
    else:
        @tf.function
        def mpjpe(y_t, y_p):
            y_t = tf.cast(y_t, dtype=tf.float32)
            y_p = tf.cast(y_p, dtype=tf.float32)
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1, keepdims=True)),
                           dtype=tf.float32)
            l = tf.sqrt(tf.reduce_sum(tf.square(y_t - y_p), axis=-1) + 1e-9) * mask
            mask = tf.cast(tf.logical_not(tf.reduce_all(tf.reduce_all(y_t == 0, axis=-1), axis=-1)), dtype=tf.float32)
            total = tf.reduce_sum(mask, axis=-1, keepdims=True)
            l = tf.reduce_sum(l, axis=-2) / total
            return tf.reduce_mean(l, axis=0)

        mpjpe._name = 'mpjpe_per_kp'
    return mpjpe

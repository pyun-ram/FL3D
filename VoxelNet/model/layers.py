import tensorflow as tf

def batch_norm(inputs, phase_train, decay=0.9, eps=1e-5,name=''):
    """Batch Normalization

       Args:
           inputs: input data(Batch size) from last layer
           phase_train: when you test, please set phase_train "None"
       Returns:
           output for next layer
    """
    with tf.variable_scope('BN'+name):
        gamma = tf.get_variable("gamma", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_mean = tf.get_variable("pop_mean", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable("pop_var", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        axes = list(range(len(inputs.get_shape()) - 1))

        if phase_train != None:
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)
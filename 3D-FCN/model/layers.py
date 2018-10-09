import tensorflow as tf

def fully_connected(input_layer, shape, name="", is_training=True):
    with tf.variable_scope("fully" + name):
        kernel = tf.get_variable("weights", shape=shape, \
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        fully = tf.matmul(input_layer, kernel)
        fully = tf.nn.relu(fully)
        fully = batch_norm(fully, is_training)
        return fully

def max_pool(input_layer, ksizes, stride, name="", padding='SAME'):
    with tf.variable_scope("Pool3D"+name):
        pool =tf.nn.max_pool3d(input=input_layer, ksize=ksizes, strides=stride, padding=padding, name=name)
    return pool

def average_pool(input_layer, ksizes, stride, name="", padding='SAME'):
    with tf.variable_scope("average3d"+name):
        avg =tf.nn.avg_pool3d(input=input_layer,ksize=ksizes,strides=stride,padding=padding,name=name)
    return avg

def conv3d_layer(input, filter, kernel, stride, use_bias, activation, name="CONV3d"):
    with tf.name_scope(name):
        network =tf.layers.conv3d(inputs=input,filters=filter,kernel_size=kernel, strides=stride, 
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
            use_bias=use_bias, activation = activation, padding="SAME",name=name)
        return network         

def conv3d_transpose_layer(input, filter, kernel, stride, use_bias, activation, name="DECONV3d"):
    with tf.name_scope(name):
        network =tf.layers.conv3d_transpose(inputs=input,filters=filter,kernel_size=kernel, strides=stride, 
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
            use_bias=use_bias, activation = activation, padding="SAME",name=name)
        return network          

def batch_norm(inputs, is_training, decay=0.9, eps=1e-5,name=''):
    """Batch Normalization

       Args:
           inputs: input data(Batch size) from last layer
           is_training: True if train phase, None if test phase
       Returns:
           output for next layer
    """
    with tf.variable_scope('BN'+name):
        gamma = tf.get_variable("gamma", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_mean = tf.get_variable("pop_mean", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable("pop_var", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        axes = list(range(len(inputs.get_shape()) - 1))

        if is_training is not None:
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)
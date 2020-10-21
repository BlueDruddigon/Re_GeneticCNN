import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# STAGES = np.array(['s1', 's2'])
# NUM_NODES = np.array([3, 5])

# L = 0
# BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0
# for nn in NUM_NODES:
#     t = nn * (nn - 1)
#     BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
#     l_bpi = int(0.5 * t)
#     L += t
# L = int(0.5 * L)

# TRAINING_EPOCHS = 3
# BATCH_SIZE = 20
# TOTAL_BATCHES = x_train.shape[0] // BATCH_SIZE

def weight_variable(weight_name, weight_shape):
    return tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name=''.join(['weight', weight_name]))

def bias_variable(bias_name, bias_shape):
    return tf.Variable(tf.constant(0.01, shape=bias_shape), name=''.join(['bias', bias_name]))

def linear_layer(x, n_hidden_units, layer_name):
    n_input = int(x.shape[1])
    weights = weight_variable(layer_name, [n_input, n_hidden_units])
    biases = bias_variable(layer_name, [n_hidden_units])
    return tf.add(tf.matmul(x, weights), biases)

def apply_convolution(x, kernel_height, kernel_width, in_channels, out_channels, layer_name):
    weights = weight_variable(layer_name, [kernel_height, kernel_width, in_channels, out_channels])
    biases = bias_variable(layer_name, [out_channels])

    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 2, 2], padding='SAME'), biases))

def apply_pool(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, kszie=[1, kernel_height, kernel_width], strides=[1, stride_size, stride_size], padding='SAME')

def add_node(node_name, connector_node_name, h=5, w=5, ic=1, oc=1):
    with tf.name_scope(node_name) as scope:
        conv = apply_convolution(tf.get_default_graph().get_tensor_by_name(connector_node_name),
                                kernel_height=h, kernel_width=w, in_channels=ic, out_channels=oc,
                                layer_name=''.join(['conv_', node_name]))

def sum_tensors(tensor_a, tensor_b, activation_function_pattern):
    if not tensor_a.startswith('Add'):
        tensor_a = ''.join([tensor_a, activation_function_pattern])

    return tf.add(tf.get_default_graph().get_tensor_by_name(tensor_a), 
                  tf.get_default_graph().get_tensor_by_name(''.join([tensor_b, activation_function_pattern])))

def has_same_elements(x):
    return len(set(x)) <= 1
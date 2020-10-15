import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

STAGES = np.array(['s1', 's2'])
NUM_NODES = np.array([3, 5])

L = 0
BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0
for nn in NUM_NODES:
    t = nn * (nn - 1)
    # BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
    # print(BITS_INDICES)
    # l_bpi = int(0.5 * t)
    L += t
L = int(0.5 * L)
print(L)

TRAINING_EPOCHS = 20
BATCH_SIZE = 20
TOTAL_BATCHES = x_train.shape[0] // BATCH_SIZE

population_size = 20
num_generations = 50
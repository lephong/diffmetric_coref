import os
import tensorflow as tf
import numpy as np

PROJECT_PATH = (os.environ.get('PROJECT_HOME') or "/media/phong/Storage/workspace/coen") + "/"

MAX_FLOAT32 = np.finfo(np.float32).max
MIN_FLOAT32 = np.finfo(np.float32).min

N_MENTION_FEATS = 14500
N_PAIRWISE_FEATS = 130

N_MENT_TYPES = 4
N_MENT_GENDERS = 4
N_MENT_NUMBERS = 3

N_IMPORTANT_WORDS = 8

MAX_N_ENTITIES_PER_MENT = 10

NUM_THREADS = 8
NUM_GPUS = 1

LINK_DELTA_FALSE_LINK = 0.1
LINK_DELTA_FALSE_NEW = 3.
LINK_DELTA_WRONG_LINK = 1.

CLUSTER_DELTA_FALSE_LINK = 0.5
CLUSTER_DELTA_FALSE_NEW = 3.
CLUSTER_DELTA_WRONG_LINK = 0.1

N_BLOCK_PW_FEATS = 300000

OPTIMIZER = None

MAX_N_MENTIONS = 3000
MAX_MENT_LEN = 20

N_ANTE_TO_GEN = 20
SCALE = 0.5

HEAD_WEIGHT = 1

MENT_LEN_RANGES = [0, 3, 5, 8, 12, 20, 1000]

# duplicate
DUP_PRE_MENT_ID = []
for i in range(MAX_N_MENTIONS):
    DUP_PRE_MENT_ID.extend(range(i))
DUP_PRE_MENT_ID = np.array(DUP_PRE_MENT_ID)

DUP_CUR_MENT_ID = []
for i in range(MAX_N_MENTIONS):
    DUP_CUR_MENT_ID.extend([i + 1] * (i + 1))
DUP_CUR_MENT_ID = np.array(DUP_CUR_MENT_ID)

# create prior
PRIOR_MEAN_LINK = np.identity(MAX_N_MENTIONS, dtype=np.float32) * 0.
PRIOR_MEAN_LINK[0, 0] = 0

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': NUM_GPUS},
                                        intra_op_parallelism_threads=NUM_THREADS))

np.set_printoptions(linewidth=75 * 3)

# utility functions
def get_k_max_tril(x, k):
    k = min(k, x.shape[1])
    x = np.tril(x) + np.triu(MIN_FLOAT32 * np.ones(x.shape), 1)
    sorted_x = np.sort(x)[:, ::-1]
    threshold = np.concatenate([np.diag(sorted_x[:k, :k]), sorted_x[k:, k - 1]])
    threshold = np.reshape(threshold, [-1, 1])
    return np.tril(np.int32(np.greater_equal(x, threshold)))

# for testing
# x = tf.constant([1., -100, 2.])
# print(sess.run(tf.nn.softmax(x)))




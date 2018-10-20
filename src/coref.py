import argparse
import sys
import subprocess as sub
from time import strftime

from corpus import Corpus
from default import *
from vocabulary import Vocabulary
import tiktok as clock

#################### read arguments ##################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default="train")
parser.add_argument("--score_type", type=str,
                    help="link or cluster",
                    default="link")
parser.add_argument("--model_path", type=str,
                    help="path to model. If mode=train then continue training with the given model",
                    default=None)
parser.add_argument("--init_model_path", type=str,
                    help="path to init model.",
                    default=None)
parser.add_argument("--eval_every_k_epochs", type=int,
                    help="evaluate after k epochs (-1 means not evaluating)",
                    default=-1)
parser.add_argument("--eval_output", type=str,
                    help="when mode=eval, output file",
                    default=None)
parser.add_argument("--unsup", action="store_true",
                    help="if true, using unsupervised learning")

parser.add_argument("--experiment_dir", type=str,
                    help="directory of this experiment",
                    default=PROJECT_PATH + "experiment_notused")

parser.add_argument("--max_length", type=int,
                    help="ignore all n_th mentions (n > max_length)",
                    default=350)

parser.add_argument("--train_na_prefix", type=str,
                    help="prefix of training files for mention features and offsets",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/train-notused/na.0.txt-na-")
parser.add_argument("--train_na_lex", type=str,
                    help="prefix of training files for mention surface (words)",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/train-notused/na_lex.0.txt")
parser.add_argument("--train_pw_prefix", type=str,
                    help="prefix of training files for pairwise features and offsets",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/train-notused/pw.0.txt-pw-")
parser.add_argument("--train_oracle_cluster", type=str,
                    help="training file for oracle clusters",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/train-notused/OPCs.0.txt")

parser.add_argument("--dev_na_prefix", type=str,
                    help="prefix of dev files for mention features and offsets",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/dev_small-na-")
parser.add_argument("--dev_na_lex", type=str,
                    help="prefix of dev files for mention features and offsets",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/dev_small-na-lex.txt")
parser.add_argument("--dev_pw_prefix", type=str,
                    help="prefix of dev files for pairwise features and offsets",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/dev_small-pw-")
parser.add_argument("--dev_oracle_cluster", type=str,
                    help="dev file for oracle clusters",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/DevOPCs.txt")

parser.add_argument("--voca", type=str,
                    help="prefix of vocabulary files",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/voca.")

parser.add_argument("--ana_det_model_path", type=str,
                    help="anaphoricity detection model path (using pretraining)",
                    default=None)
parser.add_argument("--ante_rank_model_path", type=str,
                    help="antecedent reranking model path (using pretraining)",
                    default=None)
parser.add_argument("--vanilla_model_path", type=str,
                    help="vanilla model path (using pretraining)",
                    default=None)

parser.add_argument("--a_feat_dim", type=int,
                    help="dimensions of mention features",
                    default=12)
parser.add_argument("--pw_feat_dim", type=int,
                    help="dimensions of pair-wise features",
                    default=70)
parser.add_argument("--cluster_emb_dim", type=int,
                    help="dimensions of cluster representations, set -1 if not used",
                    default=10)
parser.add_argument("--word_emb_dim", type=int,
                    help="dimensions of word embeddings, set -1 if not used",
                    default=10)

parser.add_argument("--hard_attention", action="store_true",
                    help="if true, hard attention is used. Otherwise, soft attention")
parser.add_argument("--oracle_cluster", action="store_true",
                    help="if true, oracle clusters are used.")

parser.add_argument("--optimizer", type=str,
                    help="optimizer (AdaGrad/Adam)",
                    default="AdaGrad")

parser.add_argument("--layer_1_learning_rate", type=float,
                    help="learning rate for the first layer",
                    default=0.1)
parser.add_argument("--layer_2_learning_rate", type=float,
                    help="learning rate for the second layer",
                    default=0.002)
parser.add_argument("--cluster_learning_rate", type=float,
                    help="learning rate for the recurrent connection",
                    default=0.005)
parser.add_argument("--gen_learning_rate", type=float,
                    help="learning rate for the decoder",
                    default=0.1)
parser.add_argument("--keep_prob", type=float,
                    help="keep probability for dropout",
                    default=0.6)

parser.add_argument("--n_epochs", type=int,
                    help="number of epochs",
                    default=100)
parser.add_argument("--no_gpu", action="store_true",
                    help="if true, not using gpu")

args = parser.parse_args()

if args.optimizer == "AdaGrad":
    OPTIMIZER = tf.train.AdagradOptimizer
elif args.optimizer == "Adam":
    OPTIMIZER = tf.train.AdamOptimizer
else:
    raise Exception("undefined optimizer", args.optimizer)

args.experiment_dir = os.path.abspath(args.experiment_dir)
args.train_na_prefix = os.path.abspath(args.train_na_prefix)
args.train_na_lex = os.path.abspath(args.train_na_lex)
args.train_pw_prefix = os.path.abspath(args.train_pw_prefix)
args.train_oracle_cluster = os.path.abspath(args.train_oracle_cluster)

args.dev_na_prefix = os.path.abspath(args.dev_na_prefix)
args.dev_na_lex = os.path.abspath(args.dev_na_lex)
args.dev_pw_prefix = os.path.abspath(args.dev_pw_prefix)
if args.dev_oracle_cluster is not None:
    args.dev_oracle_cluster = os.path.abspath(args.dev_oracle_cluster)

if args.model_path is not None:
    args.model_path = os.path.abspath(args.model_path)
if args.init_model_path is not None:
    args.init_model_path = os.path.abspath(args.init_model_path)

if args.model_path is not None:
    args.model_path = os.path.abspath(args.model_path)
if args.vanilla_model_path is not None:
    args.vanilla_model_path = os.path.abspath(args.vanilla_model_path)
if args.ante_rank_model_path is not None:
    args.ante_rank_model_path = os.path.abspath(args.ante_rank_model_path)
if args.eval_output is not None:
    args.eval_output = os.path.abspath(args.eval_output)
if args.voca is not None:
    args.voca = os.path.abspath(args.voca)

#################### create session #################

NUM_THREADS = 10
if args.no_gpu:
    config = tf.ConfigProto(
        device_count = {'GPU': 0},
        intra_op_parallelism_threads=NUM_THREADS
    )
else:
    config = tf.ConfigProto(
        device_count= {'CPU': 1},
        intra_op_parallelism_threads=NUM_THREADS
    )
sess = tf.Session(config=config)

#################### placeholders for graphs #################

# x_n_docs_ge_i[i] : 1-D the number of documents which contain not less than i mentions
x_doc_len = tf.placeholder(tf.int32)

# x_phi_a : 2-D containing a list of a_feature indices (row, feat_id)
x_phi_a = tf.sparse_placeholder(tf.int32)
x_phi_a_offsets = tf.placeholder(tf.int32)

# x_phi_p : 2-D containing a list of pairwise_feature indices (row, feat_id)
x_phi_p = tf.sparse_placeholder(tf.int32)
x_phi_p_offsets = tf.placeholder(tf.int32)

# x_gold_antecedents : 1-D containing
x_gold_class_antecedents = tf.placeholder(tf.float32)
x_lost_weight_link = tf.placeholder(tf.float32)

x_gold_class_cluster_ids = tf.placeholder(tf.float32)
x_lost_weight_cluster = tf.placeholder(tf.float32)

x_ment_words = tf.placeholder(tf.int32)
x_ment_words_offsets = tf.placeholder(tf.int32)

# keep prob for dropout
keep_prob = tf.placeholder(tf.float32)

############################# Coreference resolution class ######################


class CoEn:
    def __init__(self, voca_head, voca_gov, voca_ment, a_feat_dim, pw_feat_dim, clus_emb_dim, word_emb_dim):
        self.a_feat_dim = a_feat_dim
        self.pw_feat_dim = pw_feat_dim
        self.clus_emb_dim = clus_emb_dim
        self.word_emb_dim = word_emb_dim

        self.voca_head = voca_head
        self.voca_ment = voca_ment
        self.voca_gov = voca_gov

        # computing "local" scores
        self.link_Wa = tf.get_variable("link_Wa", shape=[N_MENTION_FEATS, a_feat_dim],
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_ba = tf.get_variable("link_ba", shape=[a_feat_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_Wp = tf.get_variable("link_Wp", [N_PAIRWISE_FEATS, pw_feat_dim],
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_bp = tf.get_variable("link_bp", [pw_feat_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.link_ua = tf.get_variable("link_ua", [a_feat_dim, 1],
                                       initializer=tf.uniform_unit_scaling_initializer(1),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_up = tf.get_variable("link_up", [pw_feat_dim, 1],
                                       initializer=tf.uniform_unit_scaling_initializer(1),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_u0 = tf.get_variable("link_u0", [1],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.link_v = tf.get_variable("link_va", [a_feat_dim, 1],
                                      initializer=tf.uniform_unit_scaling_initializer(1),
                                      regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_v0 = tf.get_variable("link_v0", [1],
                                       initializer=tf.constant_initializer(0),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # computing "global" scores
        self.Wc = tf.get_variable("Wc", shape=[N_MENTION_FEATS, clus_emb_dim],
                                  initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                  regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.bc = tf.get_variable("bc", shape=[clus_emb_dim],
                                  initializer=tf.constant_initializer(0.),
                                  regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # update cluster embeddings, using LSTM
        self.clus_eps_mult = tf.get_variable("clus_mult", [1],
                                             initializer=tf.constant_initializer(5),
                                             regularizer=tf.contrib.layers.l1_regularizer(1e-10))

        # input gate
        self.clus_Wie = tf.get_variable("clus_Wie", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.uniform_unit_scaling_initializer(1),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        self.clus_Wim = tf.get_variable("clus_Wim", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.uniform_unit_scaling_initializer(1),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        self.clus_bi = tf.get_variable("clus_bi", shape=[clus_emb_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l2_regularizer(1e-6))

        # forget gate
        g = 1.1
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Wfe = tf.get_variable("clus_Wfe", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Wfm = tf.get_variable("clus_Wfm", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        self.clus_bf = tf.get_variable("clus_bf", shape=[clus_emb_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l2_regularizer(1e-6))

        # output gate
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Woe = tf.get_variable("clus_Woe", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Wom = tf.get_variable("clus_Wom", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        self.clus_bo = tf.get_variable("clus_bo", shape=[clus_emb_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l2_regularizer(1e-6))

        # current mem gate
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Wce = tf.get_variable("clus_Wce", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        w, _ = np.linalg.qr(np.random.randn(clus_emb_dim, clus_emb_dim))
        self.clus_Wcm = tf.get_variable("clus_Wcm", shape=[clus_emb_dim, clus_emb_dim],
                                        initializer=tf.constant_initializer(w.flatten() * g, dtype=tf.float32),
                                        regularizer=tf.contrib.layers.l2_regularizer(1e-6))
        self.clus_bc = tf.get_variable("clus_bc", shape=[clus_emb_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l2_regularizer(1e-6))


        self.clus_2_gen_W = tf.get_variable("clus_2_gen_W", shape=[clus_emb_dim, word_emb_dim],
                                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.clus_2_gen_b = tf.get_variable("clus_2_gen_b", shape=[word_emb_dim],
                                            initializer=tf.constant_initializer(0.),
                                            regularizer=tf.contrib.layers.l2_regularizer(1e-6))

        self.gen_head_W = tf.get_variable("gen_head_W", shape=[word_emb_dim, voca_head.size()],
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.gen_head_b = tf.get_variable("gen_head_b", shape=[voca_head.size()],
                                          initializer=tf.constant_initializer(0.),
                                          regularizer=tf.contrib.layers.l2_regularizer(1e-6))

        # group params
        self.vanilla_params_layer1 = [self.link_Wa, self.link_ba, self.link_Wp, self.link_bp]
        self.vanilla_params_layer2 = [self.link_ua, self.link_up, self.link_u0, self.link_v, self.link_v0, self.clus_eps_mult]
        self.clustering_params = [self.Wc, self.bc,
                                  self.clus_Wie, self.clus_Wim, self.clus_bi,
                                  self.clus_Wfe, self.clus_Wfm, self.clus_bf,
                                  self.clus_Woe, self.clus_Wom, self.clus_bo,
                                  self.clus_Wce, self.clus_Wcm, self.clus_bc]
        self.gen_params = [self.clus_2_gen_W, self.clus_2_gen_b,
                           self.gen_head_W, self.gen_head_b]

    def build_core_graph(self, gen_ment):
        # prepare for computing p(m_j | m_i)
        ha = tf.nn.embedding_lookup_sparse(self.link_Wa, x_phi_a, None, combiner="sum")
        ha = tf.tanh(ha + self.link_ba)
        hp = tf.nn.embedding_lookup_sparse(self.link_Wp, x_phi_p, None, combiner="sum")
        hp = tf.tanh(hp + self.link_bp)

        ga = tf.matmul(ha, self.link_ua)
        ga = tf.reshape(ga, [1, x_doc_len])
        gp = tf.matmul(hp, self.link_up)

        # init while loop
        init_i = tf.constant(1)
        init_p_m_link_antecendents = tf.zeros([1, x_doc_len])

        a_nrows = tf.shape(x_phi_a.indices)[0]
        x_phi_a_new = tf.SparseTensor(
            indices=tf.concat(1, [tf.zeros([a_nrows, 1], dtype=tf.int64), tf.slice(x_phi_a.indices, [0, 1], [-1, 1])]),
            values=x_phi_a.values,
            shape=x_phi_a.shape)

        # the first cluster contain the first mention
        offsets = tf.gather(x_phi_a_offsets, [0, 0 + 1, tf.shape(x_phi_a_offsets)[0] - 1])
        mask_prefix = tf.fill([offsets[0]], False)
        mask_inner = tf.fill([offsets[1] - offsets[0]], True)
        mask_suffix = tf.fill([offsets[2] - offsets[1]], False)
        x_phi_a_0 = tf.sparse_retain(x_phi_a, tf.concat(0, [mask_prefix, mask_inner, mask_suffix]))

        hc = tf.tanh(tf.nn.embedding_lookup_sparse(self.Wc, x_phi_a_0, None, combiner="sum") + self.bc)
        input_gate = tf.nn.sigmoid(tf.matmul(hc, self.clus_Wim) + self.clus_bi)
        output_gate = tf.nn.sigmoid(tf.matmul(hc, self.clus_Wom) + self.clus_bo)
        cur_mems = tf.nn.tanh(tf.matmul(hc, self.clus_Wcm) + self.clus_bc)
        init_cluster_mems = input_gate * cur_mems
        init_cluster_embs = output_gate * tf.nn.tanh(init_cluster_mems)

        init_p_m_in_cluster = tf.pad([[1.]], [[0, 0], [0, x_doc_len - 1]], mode="CONSTANT")

        # don't care about the first mention
        if gen_ment:
            init_gen_word_h = tf.zeros([1, self.clus_emb_dim])
        else:
            init_gen_word_h = tf.constant(0.)

        def body(i, p_m_link_antecedents, cluster_embs, cluster_mems, p_m_in_cluster, gen_word_h):
            # compute local score
            ga_i = tf.slice(ga, [0, i], [1, 1])
            ga_i = tf.tile(ga_i, [1, i])

            gp_i = tf.slice(gp, [tf.to_int32((i - 1) * i / 2), 0], [i, -1])
            gp_i = tf.reshape(gp_i, [1, i])

            p_m_i_link_not_eps = gp_i + ga_i + self.link_u0
            p_m_i_link_eps = tf.matmul(tf.gather(ha, [i]), self.link_v) + self.link_v0

            p_m_i_link_antecedents = tf.concat(1, [p_m_i_link_not_eps, p_m_i_link_eps])
            p_m_i_link_antecedents = tf.nn.softmax(p_m_i_link_antecedents)

            # compute membership prob p(cluster | mention)
            p_m_i_link_not_eps = tf.slice(p_m_i_link_antecedents, [0, 0], [-1, i])
            p_m_i_link_eps = tf.slice(p_m_i_link_antecedents, [0, i], [-1, -1]) * self.clus_eps_mult
            p_m_i_in_cluster = tf.matmul(p_m_i_link_not_eps, tf.slice(p_m_in_cluster, [0, 0], [-1, i]))
            p_m_i_in_cluster = tf.concat(1, [p_m_i_in_cluster, p_m_i_link_eps])
            p_m_i_in_cluster = p_m_i_in_cluster / tf.reduce_sum(p_m_i_in_cluster)

            # compute vectors to generate mentions' surface
            if gen_ment:
                cluster_embs = tf.pad(cluster_embs, [[0, 1], [0, 0]], mode="CONSTANT")
                cluster_mems = tf.pad(cluster_mems, [[0, 1], [0, 0]], mode="CONSTANT")

                hw = tf.matmul(p_m_i_in_cluster, cluster_embs)
                gen_word_h = tf.concat(0, [gen_word_h, hw])

                # update cluster embeddings
                offsets = tf.gather(x_phi_a_offsets, [i, i + 1, tf.shape(x_phi_a_offsets)[0] - 1])
                mask_prefix = tf.fill([offsets[0]], False)
                mask_inner = tf.fill([offsets[1] - offsets[0]], True)
                mask_suffix = tf.fill([offsets[2] - offsets[1]], False)
                x_phi_a_i = tf.sparse_retain(x_phi_a_new, tf.concat(0, [mask_prefix, mask_inner, mask_suffix]))
                hc = tf.tanh(tf.nn.embedding_lookup_sparse(self.Wc, x_phi_a_i, None, combiner="sum") + self.bc)

                input_gate = tf.nn.sigmoid(
                    tf.matmul(cluster_embs, self.clus_Wie) + tf.matmul(hc, self.clus_Wim) + self.clus_bi)
                output_gate = tf.nn.sigmoid(
                    tf.matmul(cluster_embs, self.clus_Woe) + tf.matmul(hc, self.clus_Wom) + self.clus_bo)
                forget_gate = tf.nn.sigmoid(
                    tf.matmul(cluster_embs, self.clus_Wfe) + tf.matmul(hc, self.clus_Wfm) + self.clus_bf)
                cur_mems = tf.nn.tanh(
                    tf.matmul(cluster_embs, self.clus_Wce) + tf.matmul(hc, self.clus_Wcm) + self.clus_bc)
                cluster_mems = forget_gate * cluster_mems + \
                               input_gate * tf.tile(tf.transpose(p_m_i_in_cluster), [1, self.clus_emb_dim]) * cur_mems
                cluster_embs = output_gate * tf.nn.tanh(cluster_mems)

            # concate current probs with probs in prev steps
            p_m_i_in_cluster = tf.pad(p_m_i_in_cluster, [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            p_m_in_cluster = tf.concat(0, [p_m_in_cluster, p_m_i_in_cluster])

            p_m_i_link_antecedents = tf.pad(p_m_i_link_antecedents, [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            p_m_link_antecedents = tf.concat(0, [p_m_link_antecedents, p_m_i_link_antecedents])

            return i + 1, p_m_link_antecedents, cluster_embs, cluster_mems, p_m_in_cluster, gen_word_h

        def condition(i, *args):
            return tf.less(i, x_doc_len)

        _, p_m_link_antecedents, _, _, p_m_in_clusters, gen_word_h = \
                tf.while_loop(condition, body, [init_i,
                                                init_p_m_link_antecendents,
                                                init_cluster_embs,
                                                init_cluster_mems,
                                                init_p_m_in_cluster,
                                                init_gen_word_h])

        if gen_ment:
            gen_word_h = tf.tanh(tf.matmul(gen_word_h, self.clus_2_gen_W) + self.clus_2_gen_b)
            p_m_gen_head = tf.nn.softmax(tf.matmul(gen_word_h, self.gen_head_W) + self.gen_head_b)
        else:
            p_m_gen_head = None

        return p_m_link_antecedents, p_m_in_clusters, p_m_gen_head

    def build_graph_computing_sup_cost(self, p_m_link_antecedents, p_m_in_clusters, score_type="link"):
        # compute cost
        init_i = tf.constant(1)
        init_cost = tf.constant(0.)
        init_offset = tf.constant(0)

        if score_type == "link":
            scores = p_m_link_antecedents
            x_gold_class = x_gold_class_antecedents
            x_lost_weight = x_lost_weight_link
        elif score_type == "cluster":
            scores = p_m_in_clusters
            x_gold_class = x_gold_class_cluster_ids
            x_lost_weight = x_lost_weight_cluster
        else:
            raise Exception("invalid option")

        def body(i, cost, offset):
            n_cols = i + 1

            gold_class_i = tf.slice(x_gold_class, [offset], [n_cols])
            lost_weight_i = tf.slice(x_lost_weight, [offset], [n_cols])
            p_m_i = tf.slice(scores, [i, 0], [1, n_cols])

            gold_class_i = tf.reshape(gold_class_i, [1, n_cols])
            lost_weight_i = tf.reshape(lost_weight_i, [1, n_cols])
            p_m_i = tf.reshape(p_m_i, [1, n_cols])
            p_m_i = p_m_i * tf.exp(lost_weight_i)
            p_m_i = p_m_i / tf.reduce_sum(p_m_i)

            # log-loss
            before_log = tf.reduce_sum(tf.mul(p_m_i, gold_class_i), 1)
            cost_i = - tf.reduce_sum(tf.log(before_log))

            return i + 1, tf.add(cost, cost_i), offset + i + 1

        def condition(i, *args):
            return tf.less(i, x_doc_len)

        _, cost, _ = tf.while_loop(condition, body, [init_i, init_cost, init_offset])

        return cost

    def build_graph_computing_unsup_cost(self, p_m_gen_head, p_m_gen_gov=None, p_m_gen_ment=None):
        # compute cost
        init_i = tf.constant(0)
        init_cost = tf.constant(0.)

        def body(i, cost):
            offsets = tf.gather(x_ment_words_offsets, [i, i + 1])
            ment_words_i = tf.slice(x_ment_words, [offsets[0]], [offsets[1] - offsets[0]])

            p_m_i_gen_head = tf.gather(p_m_gen_head, i)
            # p_m_i_gen_gov = tf.gather(p_m_gen_gov, i)
            # p_m_i_gen_ment = tf.gather(p_m_gen_ment, i)

            cost_head = - tf.log(tf.gather(p_m_i_gen_head, ment_words_i[0]))
            # cost_gov = - tf.log(tf.gather(p_m_i_gen_gov, ment_words_i[1]))
            # cost_ment = - tf.log(tf.reduce_sum(tf.gather(p_m_i_gen_ment, tf.slice(ment_words_i, [2], [-1]))))
            return i + 1, cost + cost_head #+ cost_gov + cost_ment

        def condition(i, _):
            return tf.less(i, x_doc_len)

        _, cost = tf.while_loop(condition, body, [init_i, init_cost], name="while_unsup_cost")

        return cost / tf.to_float(x_doc_len)

    def train(self, args):
        p_m_link_antecedents, p_m_in_clusters, p_m_gen_head = model.build_core_graph(args.unsup)

        unsup_cost_func = None
        sup_unsup_cost_func = None
        unsup_optimizer = None
        sup_unsup_optimizer = None

        if args.score_type == "link":
            sup_cost_func = self.build_graph_computing_sup_cost(p_m_link_antecedents, p_m_in_clusters, score_type="link")
        elif args.score_type == "cluster":
            sup_cost_func = self.build_graph_computing_sup_cost(p_m_link_antecedents, p_m_in_clusters, score_type="cluster")
        elif args.score_type == "mix":
            sup_cost_func = self.build_graph_computing_sup_cost(p_m_link_antecedents, p_m_in_clusters, score_type="link") + \
                            self.build_graph_computing_sup_cost(p_m_link_antecedents, p_m_in_clusters, score_type="cluster")
        else:
            raise Exception("invalid option")

        # create optimizer
        layer_1_optimizer = OPTIMIZER(learning_rate=args.layer_1_learning_rate)
        layer_2_optimizer = OPTIMIZER(learning_rate=args.layer_2_learning_rate)
        clus_optimizer = OPTIMIZER(learning_rate=args.cluster_learning_rate)
        gen_optimizer = OPTIMIZER(learning_rate=args.gen_learning_rate)
        lens = np.cumsum([len(self.vanilla_params_layer1), len(self.vanilla_params_layer2),
                          len(self.clustering_params), len(self.gen_params)])

        # supervised only
        grads = tf.gradients(sup_cost_func, self.vanilla_params_layer1 + self.vanilla_params_layer2)
        grads_layer_1 = grads[:lens[0]]
        grads_layer_2 = grads[lens[0]:lens[1]]

        train_opt_l1 = layer_1_optimizer.apply_gradients(zip(grads_layer_1, self.vanilla_params_layer1))
        train_opt_l2 = layer_2_optimizer.apply_gradients(zip(grads_layer_2, self.vanilla_params_layer2))

        sup_optimizer = tf.group(train_opt_l1, train_opt_l2)

        if args.unsup:
            unsup_cost_func = self.build_graph_computing_unsup_cost(p_m_gen_head)

            # unsupervised only
            grads = tf.gradients(unsup_cost_func, self.vanilla_params_layer1 + self.vanilla_params_layer2 +
                                 self.clustering_params + self.gen_params)
            grads_layer_1 = grads[:lens[0]]
            grads_layer_2 = grads[lens[0]:lens[1]]
            grads_cluster = grads[lens[1]:lens[2]]
            grads_gen = grads[lens[2]:lens[3]]

            train_opt_l1 = layer_1_optimizer.apply_gradients(zip(grads_layer_1, self.vanilla_params_layer1))
            train_opt_l2 = layer_2_optimizer.apply_gradients(zip(grads_layer_2, self.vanilla_params_layer2))
            train_opt_cluster = clus_optimizer.apply_gradients(zip(grads_cluster, self.clustering_params))

            train_opt_gen = gen_optimizer.apply_gradients(zip(grads_gen, self.gen_params))
            unsup_optimizer = tf.group(train_opt_l1, train_opt_l2, train_opt_cluster, train_opt_gen)

            # both sup and unsup
            sup_unsup_cost_func = sup_cost_func + unsup_cost_func
            grads = tf.gradients(sup_unsup_cost_func,
                                 self.vanilla_params_layer1 + self.vanilla_params_layer2 +
                                 self.clustering_params + self.gen_params)
            grads_layer_1 = grads[:lens[0]]
            grads_layer_2 = grads[lens[0]:lens[1]]
            grads_cluster = grads[lens[1]:lens[2]]
            grads_gen = grads[lens[2]:lens[3]]

            train_opt_l1 = layer_1_optimizer.apply_gradients(zip(grads_layer_1, self.vanilla_params_layer1))
            train_opt_l2 = layer_2_optimizer.apply_gradients(zip(grads_layer_2, self.vanilla_params_layer2))
            train_opt_cluster = clus_optimizer.apply_gradients(zip(grads_cluster, self.clustering_params))

            train_opt_gen = gen_optimizer.apply_gradients(zip(grads_gen, self.gen_params))
            sup_unsup_optimizer = tf.group(train_opt_l1, train_opt_l2, train_opt_cluster, train_opt_gen)

        # for saving parameters
        saver = tf.train.Saver(max_to_keep=None)

        # training
        if args.model_path is None:
            print("initializing params")
            sess.run(tf.initialize_all_variables())
            args.model_path = args.experiment_dir + "/coref.model"
        else:
            print("loading model from file", args.model_path)
            saver.restore(sess, args.model_path)

        # load subgraph models
        if args.ana_det_model_path is not None:
            print("load ana_det model from file", args.ana_det_model_path)
            tf.train.Saver([self.link_Wa, self.link_ba]).restore(sess, args.ana_det_model_path)

        if args.ante_rank_model_path is not None:
            print("load ante_rank model from file", args.ante_rank_model_path)
            tf.train.Saver([self.link_Wp, self.link_bp]).restore(sess, args.ante_rank_model_path)

        if args.vanilla_model_path is not None:
            print("load vanilla model from file", args.vanilla_model_path)
            tf.train.Saver(self.vanilla_params_layer1 + self.vanilla_params_layer2).restore(sess, args.vanilla_model_path)

        if args.init_model_path is not None:
            print("load model from file", args.init_model_path)
            tf.train.Saver(self.vanilla_params_layer1 + self.vanilla_params_layer2 + self.clustering_params + self.gen_params).restore(sess, args.init_model_path)

        # load train data
        train_data = Corpus(args.train_na_prefix + "feats.h5",
                            args.train_na_lex,
                            args.train_pw_prefix + "feats.h5",
                            args.train_na_prefix + "offsets.h5",
                            args.train_pw_prefix + "offsets.h5",
                            args.train_oracle_cluster)

        for epoch in range(args.n_epochs):
            total_cost = 0

            train_data.shuffle()
            next_complete = 10

            for doc_id in range(train_data.n_documents):
                if doc_id > 100:
                    break

                clock.tik("data")
                flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                    flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                    flat_gold_antecedents, flat_gold_cluster_ids, \
                    flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                    ment_words, ment_words_offsets, doc_len = \
                    train_data.get_doc(doc_id, self.voca_head, self.voca_gov, self.voca_ment, args.max_length)

                supervised = False
                # if doc_id % 10 == 0:
                #     supervised = True
                # else:
                #     flat_gold_antecedents = [] # unsupervised

                flat_gold_antecedents = []

                # create feed_dict
                feed_dict = {
                    x_phi_a: (flat_a_feats_indices, flat_a_feats_ids_val,
                              [flat_a_feats_indices[:, 0].max(), flat_a_feats_indices[:, 1].max()]),
                    x_phi_a_offsets: a_feats_offsets,
                    x_phi_p: (flat_pw_feats_indices, flat_pw_feats_ids_val,
                              [flat_pw_feats_indices[:, 0].max(), flat_pw_feats_indices[:, 1].max()]),
                    x_phi_p_offsets: pw_feats_offsets,
                    x_ment_words: ment_words,
                    x_ment_words_offsets: ment_words_offsets,
                    x_doc_len: doc_len,
                    keep_prob: args.keep_prob
                }

                clock.tok("data")

                clock.tik("process")
                if not args.unsup:
                    if supervised:
                        #print("sup")
                        feed_dict[x_gold_class_antecedents] = flat_gold_antecedents
                        feed_dict[x_gold_class_cluster_ids] = flat_gold_cluster_ids
                        feed_dict[x_lost_weight_link] = flat_lost_weight_antecedent
                        feed_dict[x_lost_weight_cluster] = flat_lost_weight_cluster
                        _, c = sess.run([sup_optimizer, sup_cost_func], feed_dict=feed_dict)
                else:
                    if len(flat_gold_antecedents) == 0:
                        #print("unsup")
                        _, c = sess.run([unsup_optimizer, unsup_cost_func], feed_dict=feed_dict)
                    else:
                        #print("sup+unsup")
                        feed_dict[x_gold_class_antecedents] = flat_gold_antecedents
                        feed_dict[x_gold_class_cluster_ids] = flat_gold_cluster_ids
                        feed_dict[x_lost_weight_link] = flat_lost_weight_antecedent
                        feed_dict[x_lost_weight_cluster] = flat_lost_weight_cluster
                        _, c = sess.run([sup_unsup_optimizer, sup_unsup_cost_func], feed_dict=feed_dict)

                total_cost += c
                clock.tok("process")
                # clock.print_time("data")
                # clock.print_time("process")
                clock.reset()

                print("\t\t", strftime("%Y-%m-%d %H:%M:%S"), doc_len, "\tcost", c)

                complete = float(doc_id) / train_data.n_documents * 100
                if int(complete) > next_complete:
                    print("\t", strftime("%Y-%m-%d %H:%M:%S"), int(complete), "%\tcost", total_cost)
                    next_complete += 10

                c = 0

            print("\t", strftime("%Y-%m-%d %H:%M:%S"), "epoch", epoch, "\tcost", total_cost)

            save_path = saver.save(sess, args.model_path, global_step=epoch)
            print("\tModel saved in file: ", save_path, "\n")

    @staticmethod
    def eval(dev=True, suffix=""):
        sub.call("rm " + args.experiment_dir + "/*.out", shell=True)

        """based on https://github.com/swiseman/nn_coref/blob/master/run_experiments.py"""

        def get_conll_fmt_output(dev=True):
            """
            Assumes caller in modifiedBCS/ directory.
            """
            print("getting predictions in CoNLL format...")
            sys.stdout.flush()
            sub.call("chmod +x WriteCoNLLPreds.sh", shell=True)
            if dev:
                sub.call("./WriteCoNLLPreds.sh " + args.experiment_dir + " " +
                         args.experiment_dir + "/load_and_pred.bps" + suffix + " " +
                         args.experiment_dir + " " + args.experiment_dir + "/flat_dev_2012 " +
                         args.experiment_dir + "/gender.data",
                         shell=True)
            else:
                sub.call("./WriteCoNLLPreds.sh " + args.experiment_dir + " " +
                         args.experiment_dir + "/load_and_pred.bps" + suffix + " " +
                         args.experiment_dir + " " + args.experiment_dir + "/flat_test_2012 " +
                         args.experiment_dir + "/gender.data",
                         shell=True)
            # sometimes these java procs refuse to die
            print("killing any remaining java processes...")
            sys.stdout.flush()
            sub.call("pkill java", shell=True)

        def call_scorer_script(dev=True):
            """
            Assumes caller in main directory.
            """
            if dev:
                out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl" \
                                       " all " + args.experiment_dir + "/dev.key " +
                                       args.experiment_dir + "/load_and_pred.bps" + suffix + ".out none",
                                       shell=True)
                print("conll scorer output:\n\n", out)
                sys.stdout.flush()
            else:
                out = sub.check_output("reference-coreference-scorers/v8.01/scorer.pl" \
                                       " all " + args.experiment_dir + "/test.key " +
                                       args.experiment_dir + "/load_and_pred.bps" + suffix + ".out none",
                                       shell=True)
                print("conll scorer output:\n\n", out)
                sys.stdout.flush()

        os.chdir("../modifiedBCS")
        get_conll_fmt_output(dev)
        os.chdir("../")
        call_scorer_script(dev)
        os.chdir("src")

    def get_samples(self, probs, doc_len, n, gold_antecedents):
        samples = np.zeros([doc_len, n])

        # get the best one
        best = [0]
        log_likelihood = 0
        for i in range(1, doc_len):
            best.append(np.argmax(probs[i]))
            log_likelihood += np.log(np.sum(probs[i][np.nonzero(gold_antecedents[i])]))
            print(probs[i])
            print(gold_antecedents[i])
            print('---------------------')
        print("=================================")

        samples[:, 0] = best
        scores = np.zeros(n)

        # sampling
        for i in range(1, doc_len):
            samples[i, 1:] = np.random.choice(i + 1, n - 1, p=probs[i][0:i + 1])

            if gold_antecedents[0] is not None:
                scores = scores + gold_antecedents[i][samples[i, :].astype(int)]

        samples = samples.astype(int)

        return samples, scores, log_likelihood

    def predict(self, model_path):
        if model_path is not None:
            print("loading model from file", args.model_path)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

        p_m_link_antecedents, p_m_in_clusters, _ = model.build_core_graph(False)

        data = Corpus(args.dev_na_prefix + "feats.h5",
                      args.dev_na_lex,
                      args.dev_pw_prefix + "feats.h5",
                      args.dev_na_prefix + "offsets.h5",
                      args.dev_pw_prefix + "offsets.h5",
                      args.dev_oracle_cluster)

        fout = open(args.eval_output, "w")
        total_loglikilihood = 0

        for doc_id in range(data.n_documents):
            if doc_id > 5:
                break

            flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                flat_gold_antecedents, flat_gold_cluster_ids, \
                flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                ment_words, ment_words_offsets, doc_len = \
                    data.get_doc(doc_id, self.voca_head, self.voca_gov, self.voca_ment, 20)

            # create feed_dict
            feed_dict = {
                x_phi_a: (flat_a_feats_indices, flat_a_feats_ids_val,
                          [flat_a_feats_indices[:, 0].max(), flat_a_feats_indices[:, 1].max()]),
                x_phi_a_offsets: a_feats_offsets,
                x_phi_p: (flat_pw_feats_indices, flat_pw_feats_ids_val,
                          [flat_pw_feats_indices[:, 0].max(), flat_pw_feats_indices[:, 1].max()]),
                x_phi_p_offsets: pw_feats_offsets,
                x_doc_len: doc_len,
                keep_prob: 1.
            }

            # create feed_dict
            if args.score_type == "link":
                score_func = p_m_link_antecedents
                gold = flat_gold_antecedents
            elif args.score_type == "cluster":
                score_func = p_m_in_clusters
                gold = flat_gold_cluster_ids
            else:
                raise Exception("invalid option")

            probs = sess.run(score_func, feed_dict=feed_dict)

            gold_antecedents = [None] * doc_len
            if gold is not None:
                start = 0
                for i in range(1, doc_len):
                    gold_antecedents[i] = gold[start:start + (i + 1)]
                    start += i + 1

            samples, oracle_scores, log_likelihood = self.get_samples(probs, doc_len, 1, gold_antecedents)
            best_id = np.argmax(oracle_scores)

            if log_likelihood > -9999999:
                total_loglikilihood += log_likelihood
            else:
                print('log 0')

            for link in samples[:, best_id]:
                fout.write(str(link) + " ")
            fout.write("\n")
            fout.flush()

            if doc_id % 10 == 0:
                print(str(doc_id) + "\r", end="")

        fout.close()
        print(total_loglikilihood)

############################## main ############################

if __name__ == "__main__":
    for arg,value in vars(args).items():
        print(arg, value)

    head_voca = Vocabulary()
    head_voca.load_from_file(args.voca + "head")

    gov_voca = Vocabulary()
    gov_voca.load_from_file(args.voca + "gov")

    ment_voca = Vocabulary()
    ment_voca.load_from_file(args.voca + "ment")

    print(head_voca.size())
    print(gov_voca.size())
    print(ment_voca.size())

    print("\n----------------------------------------\n")

    os.chdir(PROJECT_PATH + "/src")
    print("creating model")
    model = CoEn(head_voca, gov_voca, ment_voca, args.a_feat_dim, args.pw_feat_dim, args.cluster_emb_dim,
                 args.word_emb_dim)

    if args.mode == "train":
        print("training")
        model.train(args)

    elif args.mode == "eval":
        model.predict(model_path=args.model_path)

    else:
        raise Exception("undefined mode")

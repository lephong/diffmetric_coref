import argparse
from time import strftime
import math

from corpus import Corpus
from corpus import MENT_TYPE
from default import *
from lm.data_utils import Vocabulary
import tiktok as clock

from lm.language_model_whileloop import LM_whileloop as LM

#################### read arguments ##################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default="train")
parser.add_argument("--model_path", type=str,
                    help="path to model. If mode=train then continue training with the given model",
                    default=None)
parser.add_argument("--init_language_model_path", type=str,
                    help="path to language model",
                    default=None)
parser.add_argument("--init_gen_model_path", type=str,
                    help="path to pretrained generative model",
                    default=None)
parser.add_argument("--init_discr_model_path", type=str,
                    help="path to pretrained discriminative model",
                    default=None)

parser.add_argument("--eval_every_k_epochs", type=int,
                    help="evaluate after k epochs (-1 means not evaluating)",
                    default=-1)
parser.add_argument("--eval_output", type=str,
                    help="when mode=eval, output file",
                    default=None)

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
                    help="vocabulary file",
                    default=PROJECT_PATH + "./data/conll-2012/small_lex_full_hd5/")
parser.add_argument("--ment_map_file", type=str,
                    help="mention feature re-mapping files",
                    default=PROJECT_PATH + "./data/conll-2012/minimal_lex_hd5/anMapping.txt")
parser.add_argument("--pw_map_file", type=str,
                    help="mention feature re-mapping files",
                    default=PROJECT_PATH + "./data/conll-2012/minimal_lex_hd5/anMapping.txt")

parser.add_argument("--a_feat_dim", type=int,
                    help="dimensions of mention features",
                    default=12)
parser.add_argument("--pw_feat_dim", type=int,
                    help="dimensions of pair-wise features",
                    default=70)

parser.add_argument("--optimizer", type=str,
                    help="optimizer (AdaGrad/Adam)",
                    default="AdaGrad")

parser.add_argument("--layer_1_learning_rate", type=float,
                    help="learning rate for the first layer",
                    default=0.1)
parser.add_argument("--layer_2_learning_rate", type=float,
                    help="learning rate for the second layer",
                    default=0.002)
parser.add_argument("--gen_learning_rate", type=float,
                    help="learning rate for the decoder",
                    default=0.1)

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
if args.init_gen_model_path is not None:
    args.init_gen_model_path = os.path.abspath(args.init_gen_model_path)
if args.init_discr_model_path is not None:
    args.init_discr_model_path = os.path.abspath(args.init_discr_model_path)
if args.init_language_model_path is not None:
    args.init_language_model_path = os.path.abspath(args.init_language_model_path)

if args.eval_output is not None:
    args.eval_output = os.path.abspath(args.eval_output)
if args.voca is not None:
    args.voca = os.path.abspath(args.voca)
if args.ment_map_file is not None:
    args.ment_map_file = os.path.abspath(args.ment_map_file)

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
x_doc_len = tf.placeholder(shape=(), dtype=tf.int32)

# x_phi_a : 2-D containing a list of a_feature indices (row, feat_id)
x_phi_a = tf.sparse_placeholder(tf.int32)
x_phi_a_offsets = tf.placeholder(tf.int32)

# x_phi_p : 2-D containing a list of pairwise_feature indices (row, feat_id)
x_phi_p = tf.sparse_placeholder(tf.int32)
x_phi_p_offsets = tf.placeholder(tf.int32)

# x_gold_antecedents : 1-D containing
x_gold_class_antecedents = tf.placeholder(tf.float32)
x_lost_weight_link = tf.placeholder(tf.float32)
x_gold_class_antecedents_supgen = tf.placeholder(tf.float32)

x_gold_class_cluster_ids = tf.placeholder(tf.float32)
x_lost_weight_cluster = tf.placeholder(tf.float32)
x_gold_class_cluster_ids_supgen = tf.placeholder(tf.float32)

x_ment_full_words = tf.placeholder(dtype=tf.int32, shape=[None, None])
x_ment_full_words_lengths = tf.placeholder(tf.int32, shape=[None])
x_ment_len_ranges = tf.placeholder(tf.int32)
x_sentences = tf.placeholder(dtype=tf.int32, shape=[None, None])
x_sentence_lengths = tf.placeholder(tf.int32, shape=[None])
x_ment_properties = tf.placeholder(tf.int32) # sent_id, start_id, end_id, head_id, type, _, number, gender
x_ment_word_weights = tf.placeholder(tf.float32)

x_disc_weight = tf.placeholder(tf.float32)
x_gen_weight = tf.placeholder(tf.float32)
x_unsup_weight = tf.placeholder(tf.float32)

x_encoder_constraints = tf.placeholder(tf.float32)
x_decoder_gender_constraints = tf.placeholder(tf.float32)
x_decoder_number_constraints = tf.placeholder(tf.float32)

x_gold_anadet_class = tf.placeholder(tf.float32)

x_i_within_i = tf.placeholder(tf.int32)

############################# Coreference resolution class ######################


class CoEn:
    def __init__(self, voca, a_feat_dim, pw_feat_dim):
        self.a_feat_dim = a_feat_dim
        self.pw_feat_dim = pw_feat_dim

        self.voca = voca

        self.n_types = N_MENT_TYPES
        self.n_genders = N_MENT_GENDERS
        self.n_numbers = N_MENT_NUMBERS

        self.layer1_params = []
        self.layer2_params = []
        self.gen_params = []

        # compute link scores: mean and var
        self.link_Wa = tf.get_variable("link_Wa.layer_1", shape=[N_MENTION_FEATS, a_feat_dim],
                                       initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_ba = tf.get_variable("link_ba.layer_1", shape=[a_feat_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_Wp_pair = tf.get_variable("link_Wp_pair.layer_1", [N_PAIRWISE_FEATS, pw_feat_dim],
                                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_Wp_prev_ment = tf.get_variable("link_Wp_prev_ment.layer_1", [N_MENTION_FEATS, pw_feat_dim],
                                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_Wp_cur_ment = tf.get_variable("link_Wp_cur_ment.layer_1", [N_MENTION_FEATS, pw_feat_dim],
                                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_bp = tf.get_variable("link_bp.layer_1", [pw_feat_dim],
                                       initializer=tf.constant_initializer(0.),
                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # mean
        self.link_mean_ua = tf.get_variable("link_mean_ua.layer_2", [a_feat_dim, 1],
                                            initializer=tf.uniform_unit_scaling_initializer(1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_mean_up = tf.get_variable("link_mean_up.layer_2", [pw_feat_dim, 1],
                                            initializer=tf.uniform_unit_scaling_initializer(1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_mean_u0 = tf.get_variable("link_mean_u0.layer_2", [1],
                                            initializer=tf.constant_initializer(0.),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.link_mean_v = tf.get_variable("link_mean_va.layer_2", [a_feat_dim, 1],
                                           initializer=tf.uniform_unit_scaling_initializer(1),
                                           regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_mean_v0 = tf.get_variable("link_mean_v0.layer_2", [1],
                                            initializer=tf.constant_initializer(0),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # var
        self.link_var_ua = tf.get_variable("link_var_ua.layer_2", [a_feat_dim, 1],
                                            initializer=tf.uniform_unit_scaling_initializer(1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_var_up = tf.get_variable("link_var_up.layer_2", [pw_feat_dim, 1],
                                            initializer=tf.uniform_unit_scaling_initializer(1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_var_u0 = tf.get_variable("link_var_u0.layer_2", [1],
                                            initializer=tf.constant_initializer(0.),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.link_var_v = tf.get_variable("link_var_va.layer_2", [a_feat_dim, 1],
                                           initializer=tf.uniform_unit_scaling_initializer(1),
                                           regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.link_var_v0 = tf.get_variable("link_var_v0.layer_2", [1],
                                            initializer=tf.constant_initializer(0),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # language model
        with tf.variable_scope("model"):
            hps = LM.get_default_hparams()
            hps.num_sampled = 0
            hps.keep_prob = 1.0
            self.LM = LM(hps, "eval", "/cpu:0")

        # compute link scores without knowing the next mention
        self.proj_ment_ctx_W = tf.get_variable("proj_ment_ctx_W.gen",
                                                [a_feat_dim + 3 * hps.projected_size, hps.projected_size],
                                                initializer=tf.uniform_unit_scaling_initializer(1.15),
                                                regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.proj_ment_ctx_b = tf.get_variable("proj_ment_ctx_b.gen",
                                                [1, hps.projected_size],
                                                initializer=tf.constant_initializer(0),
                                                regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.proj_ctx_W = tf.get_variable("proj_ctx_W.gen",
                                          [2 * hps.projected_size, hps.projected_size],
                                          initializer=tf.uniform_unit_scaling_initializer(1.15),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.proj_ctx_b = tf.get_variable("proj_ctx_b.gen",
                                          [1, hps.projected_size],
                                          initializer=tf.constant_initializer(0),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        self.prior_score_prev_ment_W = tf.get_variable("prior_score_prev_ment_W.gen", [2 * hps.emb_size, 2 * hps.emb_size],
                                                       initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.prior_score_self_W = tf.get_variable("prior_score_self_W.gen", [1 * hps.emb_size, 1],
                                                       initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01),
                                                       regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        with tf.variable_scope('decoder_mention'):
            self.ment_decoder = tf.nn.rnn_cell.GRUCell(hps.emb_size)
        with tf.variable_scope('encode_all_mentions'):
            self.all_ment_encoder = tf.nn.rnn_cell.LSTMCell(hps.projected_size)

        with tf.variable_scope('encode_all_sentences'):
            self.all_sent_encoder = tf.nn.rnn_cell.LSTMCell(hps.projected_size)

        self.ctx_2_gen_ment_W = tf.get_variable("ctx_2_gen_ment_W.gen",
                                                [5 * hps.projected_size, hps.projected_size],
                                                initializer=tf.uniform_unit_scaling_initializer(1),
                                                regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.ctx_2_gen_ment_b = tf.get_variable("ctx_2_gen_ment_b.gen",
                                                [1, hps.projected_size],
                                                initializer=tf.constant_initializer(0),
                                                regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # generate gender
        self.gen_gender_W = tf.get_variable("gen_gender_W.gen", shape=[hps.projected_size, N_MENT_GENDERS],
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.gen_gender_b = tf.get_variable("gen_gender_b.gen", shape=[N_MENT_GENDERS],
                                          initializer=tf.constant_initializer(0.),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # generate number
        self.gen_number_W = tf.get_variable("gen_number_W.gen", shape=[hps.projected_size, N_MENT_NUMBERS],
                                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.gen_number_b = tf.get_variable("gen_number_b.gen", shape=[N_MENT_NUMBERS],
                                            initializer=tf.constant_initializer(0.),
                                            regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # detect pronouns
        self.gen_pronoun_W = tf.get_variable("gen_pronoun_W.gen", shape=[hps.projected_size, 1],
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.gen_pronoun_b = tf.get_variable("gen_pronoun_b.gen", shape=[1],
                                          initializer=tf.constant_initializer(0.),
                                          regularizer=tf.contrib.layers.l1_regularizer(1e-6))

        # detect non anaphoricity
        self.non_anaph_transform_W = tf.get_variable("non_anaph_transform_W", [2 * hps.projected_size, hps.projected_size],
                                                     initializer=tf.uniform_unit_scaling_initializer(1.15),
                                                     regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.non_anaph_transform_b = tf.get_variable("non_anaph_transform_b", [1, hps.projected_size],
                                                     initializer=tf.constant_initializer(0.),
                                                     regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.non_anaph_detect_W = tf.get_variable("non_anaph_detect_W", [2 * hps.projected_size, 1],
                                                  initializer=tf.uniform_unit_scaling_initializer(1),
                                                  regularizer=tf.contrib.layers.l1_regularizer(1e-6))
        self.non_anaph_detect_b = tf.get_variable("non_anaph_detect_b", [1],
                                                  initializer=tf.constant_initializer(0.),
                                                  regularizer=tf.contrib.layers.l1_regularizer(1e-6))

    def encoder(self, eval=False):
        # prepare for computing p(m_j | m_i)
        ha = tf.nn.embedding_lookup_sparse(self.link_Wa, x_phi_a, None, combiner="sum")
        ha = tf.tanh(ha + self.link_ba)
        mean_score_m_link_eps = tf.matmul(ha, self.link_mean_v) + self.link_mean_v0
        log_var_score_m_link_eps = tf.matmul(ha, self.link_var_v) + self.link_var_v0

        hp_pair = tf.nn.embedding_lookup_sparse(self.link_Wp_pair, x_phi_p, None, combiner="sum")
        hp_prev_ment = tf.gather(tf.nn.embedding_lookup_sparse(self.link_Wp_prev_ment, x_phi_a, None, combiner="sum"),
                                 tf.slice(DUP_PRE_MENT_ID, [0], [tf.shape(hp_pair)[0]]))
        hp_cur_ment = tf.gather(tf.nn.embedding_lookup_sparse(self.link_Wp_cur_ment, x_phi_a, None, combiner="sum"),
                                tf.slice(DUP_CUR_MENT_ID, [0], [tf.shape(hp_pair)[0]]))
        hp = tf.tanh(hp_pair + hp_cur_ment + hp_prev_ment + self.link_bp)

        mean_ga = tf.matmul(ha, self.link_mean_ua)
        mean_ga = tf.reshape(mean_ga, [1, x_doc_len])
        mean_gp = tf.matmul(hp, self.link_mean_up)
        log_var_ga = tf.matmul(ha, self.link_var_ua)
        log_var_ga = tf.reshape(log_var_ga, [1, x_doc_len])
        log_var_gp = tf.matmul(hp, self.link_var_up)

        # init while loop
        init_i = tf.constant(1)
        mean_score_m_link_antecendents = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False).\
            write(0, tf.pad([[0.]], [[0, 0], [0, x_doc_len - 1]], mode='CONSTANT'))
        var_score_m_link_antecendents = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False).\
            write(0, tf.pad([[1.]], [[0, 0], [0, x_doc_len - 1]], mode='CONSTANT'))
        log_p_m_link_antecendents = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False).\
            write(0, tf.pad([[0.]], [[0, 0], [0, x_doc_len - 1]], mode='CONSTANT'))

        if not eval:
            noise = tf.random_normal([x_doc_len, x_doc_len])
        else:
            noise = tf.zeros([x_doc_len, x_doc_len])

        def _time_step(i, mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents):
            # compute mean link score
            mean_ga_i = tf.slice(mean_ga, [0, i], [1, 1])
            mean_ga_i = tf.tile(mean_ga_i, [1, i])

            mean_gp_i = tf.slice(mean_gp, [tf.to_int32((i - 1) * i / 2), 0], [i, -1])
            mean_gp_i = tf.reshape(mean_gp_i, [1, i])

            mean_score_m_i_link_not_eps = mean_gp_i + mean_ga_i + self.link_mean_u0
            mean_score_m_i_link_eps = tf.reshape(tf.gather(mean_score_m_link_eps, [i]), [1, 1])

            mean_score_m_i_link_antecedents = tf.concat(1, [mean_score_m_i_link_not_eps, mean_score_m_i_link_eps])
            # set iwi = -inf
            iwi_mask = tf.to_float(x_i_within_i[i:i + 1, :i + 1])
            mean_score_m_i_link_antecedents = mean_score_m_i_link_antecedents * (1 - iwi_mask) + iwi_mask * (-50)
            mean_score_m_i_link_antecedents_padded = tf.pad(mean_score_m_i_link_antecedents,
                                                            [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            mean_score_m_link_antecedents = mean_score_m_link_antecedents.write(i, mean_score_m_i_link_antecedents_padded)

            # compute var link score
            log_var_ga_i = tf.slice(log_var_ga, [0, i], [1, 1])
            log_var_ga_i = tf.tile(log_var_ga_i, [1, i])

            log_var_gp_i = tf.slice(log_var_gp, [tf.to_int32((i - 1) * i / 2), 0], [i, -1])
            log_var_gp_i = tf.reshape(log_var_gp_i, [1, i])

            log_var_score_m_i_link_not_eps = log_var_gp_i + log_var_ga_i + self.link_var_u0
            log_var_score_m_i_link_eps = tf.reshape(tf.gather(log_var_score_m_link_eps, [i]), [1, 1])

            var_score_m_i_link_antecedents = tf.exp(tf.concat(1, [log_var_score_m_i_link_not_eps,
                                                                  log_var_score_m_i_link_eps]))
            var_score_m_i_link_antecedents_padded = tf.pad(var_score_m_i_link_antecedents,
                                                           [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            var_score_m_link_antecedents = var_score_m_link_antecedents.write(i, var_score_m_i_link_antecedents_padded)

            # p
            score_m_i_link_antecedents = mean_score_m_i_link_antecedents + \
                                         tf.sqrt(var_score_m_i_link_antecedents) * noise[i:i + 1, 0:i + 1]
            log_p_m_i_link_antecedents = tf.nn.log_softmax(score_m_i_link_antecedents)
            log_p_m_i_link_antecedents_padded = tf.pad(log_p_m_i_link_antecedents,
                                                   [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            log_p_m_link_antecedents = log_p_m_link_antecedents.write(i, log_p_m_i_link_antecedents_padded)

            return i + 1, mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents

        _, mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents = \
            tf.while_loop(cond=lambda i, *_: tf.less(i, x_doc_len),
                          body=_time_step,
                          loop_vars=(init_i,
                                     mean_score_m_link_antecendents,
                                     var_score_m_link_antecendents,
                                     log_p_m_link_antecendents))

        mean_score_m_link_antecedents = mean_score_m_link_antecedents.concat()
        var_score_m_link_antecedents = var_score_m_link_antecedents.concat()
        log_p_m_link_antecedents = log_p_m_link_antecedents.concat()
        log_p_m_link_antecedents = log_p_m_link_antecedents + \
                                   tf.slice(np.triu(MIN_FLOAT32 * np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS],
                                                                          dtype=np.float32),
                                                    1),
                                            [0, 0], [x_doc_len, x_doc_len])

        return mean_score_m_link_antecedents, var_score_m_link_antecedents, \
               log_p_m_link_antecedents

    def supervised_cost(self, mean_score_m_link_antecedents):
        init_i = tf.constant(1)
        init_link_cost = tf.constant(0.)
        init_offset = tf.constant(0)

        def body(i, link_cost, offset):
            n_cols = i + 1

            # compute link cost
            gold_class_i = tf.slice(x_gold_class_antecedents, [offset], [n_cols])
            lost_weight_i = tf.slice(x_lost_weight_link, [offset], [n_cols])
            p_m_i = tf.nn.softmax(tf.slice(mean_score_m_link_antecedents, [i, 0], [1, n_cols]))

            gold_class_i = tf.reshape(gold_class_i, [1, n_cols])
            lost_weight_i = tf.reshape(lost_weight_i, [1, n_cols])
            p_m_i = tf.reshape(p_m_i, [1, n_cols])
            p_m_i = p_m_i * tf.exp(lost_weight_i)
            p_m_i = p_m_i / tf.reduce_sum(p_m_i)

            before_log = tf.reduce_sum(p_m_i * gold_class_i, 1)
            link_cost_i = - tf.reduce_sum(tf.log(before_log))

            return i + 1, link_cost + link_cost_i, offset + i + 1

        def condition(i, *args):
            return tf.less(i, x_doc_len)

        _, link_cost, _ = tf.while_loop(condition, body, [init_i, init_link_cost, init_offset])

        return link_cost

    def _encode_ment_bucket(self, ment_embs, len_range, mask):
        n_time_steps = tf.minimum(len_range, tf.shape(x_ment_full_words)[1])
        selected_ids = tf.reshape(tf.to_int32(tf.where(mask)), [-1])

        selected_ments = tf.gather(x_ment_full_words, selected_ids)[:, :n_time_steps]
        batch_size = tf.shape(selected_ments)[0]

        outputs = self.LM._forward(0, selected_ments, None, None, compute_loss=False)

        outputs = tf.reshape(outputs, [-1, tf.shape(outputs)[-1]])
        positions = tf.gather(x_ment_full_words_lengths - 1, selected_ids) + \
                    n_time_steps * tf.range(0, limit=batch_size)  # we ignore the last token, i.e <S>
        outputs = tf.expand_dims(tf.gather(outputs, positions), 1)

        ment_embs = ment_embs.scatter(selected_ids, outputs)
        return ment_embs

    def _gen_ment_bucket(self, ante_embs, log_p_m_link_prev_kbest_m_self, mask_link_kbest_m_self,
                         len_range, mask, log_p_idv_chain):
        n_ante_to_gen = tf.minimum(x_doc_len - 1, N_ANTE_TO_GEN)

        selected_ids = tf.reshape(tf.to_int32(tf.where(mask)), [-1])
        n_ments = tf.size(selected_ids)

        selected_ments = tf.reshape(tf.tile(tf.gather(x_ment_full_words, selected_ids), [1, n_ante_to_gen + 1]),
                                    [n_ments * (n_ante_to_gen + 1), -1])
        selected_ments = selected_ments[:, :tf.minimum(tf.shape(x_ment_full_words)[1], len_range)]

        selected_ments_word_weights = tf.reshape(tf.tile(tf.gather(x_ment_word_weights, selected_ids), [1, n_ante_to_gen + 1]),
                                    [n_ments * (n_ante_to_gen + 1), -1])
        selected_ments_word_weights = selected_ments_word_weights[:, :tf.minimum(tf.shape(x_ment_full_words)[1], len_range)]

        selected_ments_is_pronoun = tf.gather(x_ment_properties[:, 4], selected_ids)
        selected_ments_is_pronoun = tf.to_float(tf.equal(selected_ments_is_pronoun, tf.constant(MENT_TYPE["PRONOMINAL"])))
        selected_ments_number = tf.reshape(
            tf.tile(tf.gather(x_ment_properties[:, 5:6], selected_ids), [1, n_ante_to_gen + 1]),
            [n_ments * (n_ante_to_gen + 1), -1])
        selected_ments_gender = tf.reshape(
            tf.tile(tf.gather(x_ment_properties[:, 6:7], selected_ids), [1, n_ante_to_gen + 1]),
            [n_ments * (n_ante_to_gen + 1), -1])

        batch_size = tf.shape(selected_ments)[0]

        # each mention starts with <S> but don't care the <S> at the end
        input_words_i = selected_ments[:, :-1]
        n_time_steps = tf.shape(input_words_i)[1]
        word_embs = tf.reshape(tf.nn.embedding_lookup(self.LM.emb_vars,
                                                      tf.reshape(input_words_i, [-1])),
                               [batch_size, n_time_steps, self.LM.hps.emb_size])
        selected_ante_embs = tf.reshape(tf.gather(ante_embs, selected_ids),
                                        [batch_size, self.ment_decoder.output_size])

        with tf.variable_scope("decode_mention"):
            output_i, _ = tf.nn.dynamic_rnn(self.ment_decoder, word_embs,
                                            initial_state=selected_ante_embs, dtype=tf.float32)

        output_i = tf.reshape(output_i, [batch_size * n_time_steps, -1])
        log_p_gen_ment_indiv_words_i = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.matmul(output_i, self.LM.softmax_w) + self.LM.softmax_b,
            tf.reshape(selected_ments[:, 1:], [-1]))
        log_p_gen_ment_indiv_words_i = tf.reshape(log_p_gen_ment_indiv_words_i, [n_ments, n_ante_to_gen + 1, n_time_steps]) * \
                                       tf.reshape(selected_ments_word_weights, [n_ments, n_ante_to_gen + 1, n_time_steps])

        # for pronouns
        p_gen_ment_pronouns = tf.nn.sigmoid(tf.matmul(selected_ante_embs, self.gen_pronoun_W) + self.gen_pronoun_b)
        p_gen_ment_pronouns = tf.reshape(p_gen_ment_pronouns, [n_ments, n_ante_to_gen + 1])
        log_p_gen_ment_gender = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.matmul(selected_ante_embs, self.gen_gender_W) + self.gen_gender_b,
            tf.reshape(selected_ments_gender, [-1]))
        log_p_gen_ment_gender = tf.reshape(log_p_gen_ment_gender, [n_ments, n_ante_to_gen + 1])
        log_p_gen_ment_number = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.matmul(selected_ante_embs, self.gen_number_W) + self.gen_number_b,
            tf.reshape(selected_ments_number, [-1]))
        log_p_gen_ment_number = tf.reshape(log_p_gen_ment_number, [n_ments, n_ante_to_gen + 1])

        init_i = tf.constant(0)
        sum_log_p = tf.constant(0.)

        def _time_step_sum_log(i, sum_log_p, log_p_idv_chain):
            id = selected_ids[i]
            is_pronoun = selected_ments_is_pronoun[i]
            log_p_idv_words_i = log_p_gen_ment_indiv_words_i[i, :, :x_ment_full_words_lengths[id] - 1]
            log_p_idv_chain_i = is_pronoun * (tf.log(p_gen_ment_pronouns[i, :]) +
                                              log_p_gen_ment_number[i, :] +
                                              log_p_gen_ment_gender[i, :]) + \
                                (1 - is_pronoun) * (tf.log(1 - p_gen_ment_pronouns[i, :]) +
                                                    tf.reduce_sum(log_p_idv_words_i, 1))
            log_p_idv_chain = log_p_idv_chain.write(id, log_p_idv_chain_i)

            log_ps_i = tf.boolean_mask(log_p_m_link_prev_kbest_m_self[id, :],
                                       tf.reshape(mask_link_kbest_m_self[id, :], [x_doc_len]))
            n_cases = tf.size(log_ps_i)
            log_p_idv_chain_i = log_p_idv_chain_i[:n_cases] + log_ps_i

            sum_log_p = sum_log_p + tf.reshape(tf.reduce_logsumexp(log_p_idv_chain_i), ())
            return i + 1, sum_log_p, log_p_idv_chain

        _, sum_log_p, log_p_idv_chain = tf.while_loop(cond=lambda i, *_: tf.less(i, n_ments),
                                                      body=_time_step_sum_log,
                                                      loop_vars=(init_i, sum_log_p, log_p_idv_chain))

        return sum_log_p, log_p_idv_chain

    def compute_ment_embs(self):
        ment_embs = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False)

        def _encode_ment(i, ment_embs):
            mask = (x_ment_full_words_lengths >= x_ment_len_ranges[i - 1]) & (x_ment_full_words_lengths < x_ment_len_ranges[i])
            ment_embs = self._encode_ment_bucket(ment_embs, x_ment_len_ranges[i], mask)
            return i + 1, ment_embs

        _, ment_embs = tf.while_loop(cond=lambda i, *_: tf.shape(x_ment_full_words)[1] >= x_ment_len_ranges[i - 1],
                                     body=_encode_ment,
                                     loop_vars=(tf.constant(1), ment_embs))

        ment_embs = ment_embs.concat()
        input = tf.reshape(ment_embs, [1, x_doc_len, self.LM.hps.projected_size])

        # using an lstm to capture all mentions
        with tf.variable_scope("encode_all_mentions"):
            output, _ = tf.nn.dynamic_rnn(self.all_ment_encoder, input, dtype=tf.float32)

        all_prev_ment_embs = tf.reshape(output, [-1, self.all_ment_encoder.output_size])
        all_prev_ment_embs = tf.concat(0, [tf.zeros([1, tf.shape(all_prev_ment_embs)[-1]]), all_prev_ment_embs[:-1, :]])

        return ment_embs, all_prev_ment_embs

    def compute_sent_embs(self):
        n_sents = tf.shape(x_sentences)[0]
        n_time_steps = tf.shape(x_sentences)[1]

        with tf.variable_scope("model", reuse=True):
            sent_embs = self.LM._forward(0, x_sentences, None, None, compute_loss=False)

        # compute concat of all prev sentences
        input = tf.reshape(sent_embs, [n_sents * n_time_steps, -1])
        idx = tf.range(n_sents) * n_time_steps + x_sentence_lengths - 1 # ignore the last token <S>
        input = tf.gather(input, idx)
        input = tf.reshape(input, [1, n_sents, self.LM.hps.projected_size])

        with tf.variable_scope("encode_all_sentences"):
            output, _ = tf.nn.dynamic_rnn(self.all_sent_encoder, input, dtype=tf.float32)

        discourse_embs = tf.reshape(output, [n_sents, -1])
        discourse_embs = tf.concat(0, [tf.zeros([1, self.all_sent_encoder.output_size]), discourse_embs])

        return sent_embs, discourse_embs

    def decoder(self, mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents,
                prior_weight=0.01):

        # compute embeddings for mentions
        ment_embs, all_prev_ment_embs = self.compute_ment_embs()

        # compute text context for each mention
        sent_embs, all_prev_sent_embs = self.compute_sent_embs()

        n_sents = tf.shape(sent_embs)[0]
        sen_len = tf.shape(sent_embs)[1]
        emb_dim = tf.shape(sent_embs)[2]

        all_prev_sent_embs_ctx = tf.gather(all_prev_sent_embs, x_ment_properties[:, 0])
        all_prev_words_in_sent_embs_ctx = tf.concat(1, [tf.zeros([n_sents, 1, emb_dim]), sent_embs])
        all_prev_words_in_sent_embs_ctx = tf.gather(
            tf.reshape(all_prev_words_in_sent_embs_ctx, [n_sents * (sen_len + 1), emb_dim]),
            tf.gather(tf.range(n_sents) * (sen_len + 1), x_ment_properties[:, 0]) + x_ment_properties[:, 1])
        text_ctx = tf.concat(1, [all_prev_sent_embs_ctx, all_prev_words_in_sent_embs_ctx])

        # split p for prev ments and p for self
        pair_filter = tf.slice(np.tril(np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), 0),
                                     [0, 0], [x_doc_len, x_doc_len])
        prev_ments_filter = tf.slice(np.tril(np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), -1),
                                     [0, 0], [x_doc_len, x_doc_len])
        log_p_m_link_prev_m = log_p_m_link_antecedents * prev_ments_filter + \
                              tf.slice(np.triu(MIN_FLOAT32 * np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), 0),
                                     [0, 0], [x_doc_len, x_doc_len])

        # select only N_ANTE_TO_GEN
        n_ante_to_gen = tf.minimum(N_ANTE_TO_GEN, x_doc_len - 1)
        _, idx = tf.nn.top_k(log_p_m_link_prev_m, k=n_ante_to_gen)
        idx, _ = tf.nn.top_k(idx, k=n_ante_to_gen, sorted=True) # dirty trick to sort idx
        idx = tf.reverse(idx, [False, True])
        rows = tf.reshape(tf.tile(tf.reshape(tf.range(x_doc_len), [-1, 1]), [1, n_ante_to_gen]), [-1, 1])
        idx = tf.concat(1, [rows, tf.reshape(idx, [-1, 1])])
        mask_link_prev_kbest_m = (tf.sparse_to_dense(idx, [x_doc_len, x_doc_len], 1.) * prev_ments_filter) > 0
        mask_link_kbest_m_self = (tf.to_float(mask_link_prev_kbest_m) + (pair_filter - prev_ments_filter)) > 0
        log_p_m_link_prev_kbest_m_self = log_p_m_link_antecedents * tf.to_float(mask_link_kbest_m_self) + \
                                         (1 - tf.to_float(mask_link_kbest_m_self)) * MIN_FLOAT32

        # generate full mentions
        # mention i-the (i < N_ANTE_TO_GEN) is genearated by j > i with 0 probability
        tmp = tf.slice(np.triu(np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS])),
                       [0, 0], [n_ante_to_gen, n_ante_to_gen])
        tmp = tf.pad(tmp, [[0, x_doc_len - n_ante_to_gen], [x_doc_len - n_ante_to_gen, 0]]) > 0
        tmp_mask = mask_link_kbest_m_self | tmp
        selected_ids = tf.to_int32(tf.where(tmp_mask)[:, 1])
        ante_embs = tf.concat(
            2, [tf.reshape(tf.gather(ment_embs, selected_ids), [x_doc_len, n_ante_to_gen + 1, self.LM.hps.projected_size]),
                tf.reshape(tf.gather(text_ctx, selected_ids), [x_doc_len, n_ante_to_gen + 1, tf.shape(text_ctx)[-1]])])

        self.selected_id = tf.reshape(selected_ids, [x_doc_len, n_ante_to_gen + 1])

        # replace all self-link ante_embs by zeros vectors
        replace_mask = tf.concat(0, [tf.slice(np.identity(N_ANTE_TO_GEN + 1, dtype=np.float32),
                                              [0, 0], [n_ante_to_gen + 1, n_ante_to_gen + 1]),
                                     tf.pad(tf.ones([x_doc_len - n_ante_to_gen - 1, 1]), [[0, 0], [n_ante_to_gen, 0]])])
        replace_mask = tf.tile(tf.expand_dims(replace_mask, 2), [1, 1, tf.shape(ante_embs)[-1]])
        ante_embs = tf.concat(
            2, [ante_embs * (1 - replace_mask),
                tf.tile(tf.expand_dims(text_ctx, 1), [1, n_ante_to_gen + 1, 1])])
        ante_embs = tf.reshape(tf.tanh(tf.matmul(tf.reshape(ante_embs, [x_doc_len * (n_ante_to_gen + 1), -1]),
                                                 self.ctx_2_gen_ment_W) +
                                       self.ctx_2_gen_ment_b),
                               [x_doc_len, n_ante_to_gen + 1, -1])

        log_p_idv_chain = tf.TensorArray(dtype=tf.float32, size=x_doc_len)

        def _gen_ment(i, sum_log_p, log_p_idv_chain):
            mask = (x_ment_full_words_lengths >= x_ment_len_ranges[i - 1]) & \
                   (x_ment_full_words_lengths < x_ment_len_ranges[i])
            sum_log_p_i, log_p_idv_chain = self._gen_ment_bucket(ante_embs, log_p_m_link_prev_kbest_m_self,
                                                                 mask_link_kbest_m_self, x_ment_len_ranges[i], mask,
                                                                 log_p_idv_chain)
            sum_log_p = sum_log_p + sum_log_p_i
            return i + 1, tf.reshape(sum_log_p, ()), log_p_idv_chain

        _, sum_log_p, log_p_idv_chain = tf.while_loop(
            cond=lambda i, *_: tf.shape(x_ment_full_words)[1] >= x_ment_len_ranges[i - 1],
            body=_gen_ment,
            loop_vars=(tf.constant(1), tf.constant(0.), log_p_idv_chain))

        cost = - sum_log_p

        self.log_p_idv_chain = log_p_idv_chain.pack()

        # D_KL
        prior_mean_m_link = tf.slice(PRIOR_MEAN_LINK, [0, 0], [x_doc_len, x_doc_len])
        cost_prior = prior_weight * (-0.5) * tf.reduce_sum((1 + tf.log(var_score_m_link_antecedents + 1e-6) -
                                      tf.pow(mean_score_m_link_antecedents - prior_mean_m_link, 2) -
                                      var_score_m_link_antecedents) * pair_filter)

        return cost + cost_prior

    def sup_gen(self):
        noise = tf.random_uniform([x_doc_len, x_doc_len], minval=0, maxval=0.1)
        p_m_link_antecedents = (x_gold_class_antecedents_supgen + noise) / \
                               tf.reduce_sum(x_gold_class_antecedents_supgen, 1, keep_dims=True)
        log_p_m_link_antecedents = tf.log(p_m_link_antecedents)

        return self.decoder(x_gold_class_antecedents_supgen,
                            tf.zeros(tf.shape(x_gold_class_antecedents_supgen)),
                            log_p_m_link_antecedents, prior_weight=0.)

    def coref_without_current_mention(self):
        ment_embs, _ = self.compute_ment_embs()
        sent_embs, discourse_embs = self.compute_sent_embs()

        n_sents = tf.shape(sent_embs)[0]
        sen_len = tf.shape(sent_embs)[1]
        emb_dim = tf.shape(sent_embs)[2]

        all_prev_sent_embs_ctx = tf.gather(discourse_embs, x_ment_properties[:, 0])
        all_prev_words_in_sent_embs_ctx = tf.gather(
            tf.reshape(sent_embs, [n_sents * sen_len, emb_dim]),
            tf.gather(tf.range(n_sents) * sen_len, x_ment_properties[:, 0]) + x_ment_properties[:, 1] - 1) # get the output at the time step right before the mention
        ctx_embs = tf.concat(1, [all_prev_words_in_sent_embs_ctx, all_prev_sent_embs_ctx])

        cost, score = self.compute_probs_coref_without_current_mention(ment_embs, ctx_embs)

        return score, cost

    def compute_probs_coref_without_current_mention(self, ment_embs, ctx_embs):
        ha = tf.nn.embedding_lookup_sparse(self.link_Wa, x_phi_a, None, combiner="sum")
        ha = tf.tanh(ha + self.link_ba)

        ment_ctx_embs = tf.tanh(tf.matmul(tf.concat(1, [ha, ment_embs, ctx_embs]), self.proj_ment_ctx_W) +
                                self.proj_ment_ctx_b)
        ctx_embs = tf.tanh(tf.matmul(ctx_embs, self.proj_ctx_W) + self.proj_ctx_b)

        score_m_link_prev_ments = tf.matmul(ctx_embs, #tf.matmul(ment_ctx_embs, self.prior_score_prev_ment_W),
                                            ment_ctx_embs,
                                            transpose_b=True)
        score_m_link_self = tf.matmul(ctx_embs, self.prior_score_self_W)

        p_m_link_ante = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False)
        p_m_link_ante = p_m_link_ante.write(0, tf.zeros([1, x_doc_len]))

        init_i = tf.constant(1)
        init_link_cost = tf.constant(0.)
        init_offset = tf.constant(0)

        def _each_mention(i, link_cost, offset, p_m_link_ante):
            n_cols = i + 1

            # compute link cost
            gold_class_i = tf.slice(x_gold_class_antecedents, [offset], [n_cols])
            lost_weight_i = tf.slice(x_lost_weight_link, [offset], [n_cols])
            score = tf.concat(1, [score_m_link_prev_ments[i:i + 1, :i], score_m_link_self[i:i + 1, :]])
            p_m_i = tf.nn.softmax(score)
            # p_m_i = tf.Print(p_m_i, [p_m_i], summarize=1000)
            p_m_link_ante = p_m_link_ante.write(i, tf.pad(p_m_i, [[0, 0], [0, x_doc_len - i - 1]]))

            gold_class_i = tf.reshape(gold_class_i, [1, n_cols])
            lost_weight_i = tf.reshape(lost_weight_i, [1, n_cols])
            p_m_i = tf.reshape(p_m_i, [1, n_cols])
            p_m_i = p_m_i * tf.exp(lost_weight_i)
            p_m_i = p_m_i / tf.reduce_sum(p_m_i)

            before_log = tf.reduce_sum(p_m_i * gold_class_i, 1)
            link_cost_i = - tf.reduce_sum(tf.log(before_log))

            return i + 1, link_cost + link_cost_i, offset + i + 1, p_m_link_ante

        _, link_cost, _, p_m_link_ante = tf.while_loop(cond=lambda i, *_: i < x_doc_len,
                                                           body=_each_mention,
                                                           loop_vars=[init_i, init_link_cost, init_offset,
                                                                      p_m_link_ante])

        link_cost = link_cost / tf.to_float(x_doc_len)

        return link_cost, p_m_link_ante.pack()

    def train(self, args):
        mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents = self.encoder()
        disc_link_cost_func = self.supervised_cost(mean_score_m_link_antecedents)

        # gen_cost_func = self.sup_gen()
        # cost_func = x_disc_weight * disc_link_cost_func + x_gen_weight * gen_cost_func

        # unsup_cost_func = self.decoder(mean_score_m_link_antecedents, var_score_m_link_antecedents,
        #                                log_p_m_link_antecedents, prior_weight=0.01)
        # cost_func = x_disc_weight * disc_link_cost_func + x_unsup_weight * unsup_cost_func

        # _, cost_func = self.detect_anaphoricity()

        # gen_cost_func = self.sup_gen()
        # cost_func = x_gen_weight * gen_cost_func

        cost_func = disc_link_cost_func

        # _, cost = self.coref_without_current_mention()
        # cost_func = cost

        # don't udpate the language model
        variables = [w for w in tf.contrib.framework.get_variables()
                     if w not in tf.contrib.framework.get_variables(scope="model")]

        optimizer = tf.contrib.layers.optimize_loss(cost_func,
                                                    tf.contrib.framework.get_global_step(),
                                                    optimizer='Adagrad',
                                                    # variables=variables,
                                                    learning_rate=args.layer_1_learning_rate,
                                                    clip_gradients=10.0)

        # for saving parameters
        saver = tf.train.Saver(max_to_keep=None, write_version=1)

        print("init variables")
        sess.run(tf.initialize_all_variables())

        # training
        if args.init_language_model_path is not None:
            print("load language model from", args.init_language_model_path)
            params = [w for w in tf.contrib.framework.get_variables(scope="model")
                      if w not in tf.contrib.framework.get_variables_by_suffix("/Adagrad")]
            ckpt = tf.train.get_checkpoint_state(args.init_language_model_path)
            tf.train.Saver(params).restore(sess, ckpt.model_checkpoint_path)

        if args.init_gen_model_path is not None:
            print("init params from pretrained generative model from", args.init_gen_model_path)
            all_params = tf.contrib.framework.get_variables_to_restore()
            not_used_params = tf.contrib.framework.get_variables_by_suffix("/Adagrad") + \
                              tf.contrib.framework.get_variables_by_suffix(".layer_1") + \
                              tf.contrib.framework.get_variables_by_suffix(".layer_2")
            params = [w for w in all_params if w not in not_used_params]
            tf.train.Saver(params).restore(sess, args.init_gen_model_path)

        if args.init_discr_model_path is not None:
            print("init params from pretrained discriminative model from", args.init_discr_model_path)
            all_params = tf.contrib.framework.get_variables_to_restore()
            not_used_params = tf.contrib.framework.get_variables_by_suffix("/Adagrad") + \
                              tf.contrib.framework.get_variables_by_suffix(".gen") + \
                              tf.contrib.framework.get_variables(scope="encode_mention") + \
                              tf.contrib.framework.get_variables(scope="decode_mention") + \
                              tf.contrib.framework.get_variables(scope="encode_all_mentions") + \
                              tf.contrib.framework.get_variables(scope="encode_sentence") + \
                              tf.contrib.framework.get_variables(scope="encode_all_sentences")
            params = [w for w in all_params if w not in not_used_params]
            tf.train.Saver(params).restore(sess, args.init_discr_model_path)

        if args.model_path is not None:
            print("loading model from file", args.model_path)
            saver.restore(sess, args.model_path)

        if args.model_path is None:
            args.model_path = args.experiment_dir + "/coref.model"

        sess.run(tf.initialize_local_variables())

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
                disc_weight = 0.
                unsup_weight = 1.
                gen_weight = 0.

                # if doc_id > 10:
                #     break

                if doc_id % 1 == 0:
                    disc_weight = 1.
                    gen_weight = 0.
                    unsup_weight = 1.

                clock.tik("data")
                flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                    flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                    flat_gold_antecedents, flat_gold_cluster_ids, \
                    flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                    ment_full_word, ment_full_word_lengths, ment_properties, \
                    sentences, sentence_lengths, \
                    doc_len, antecedents_weights = train_data.get_doc(doc_id, self.voca, args.max_length)

                ante_matrix = Corpus.convert_flat_cluster_idx_to_matrix(flat_gold_antecedents, doc_len)
                clus_ids_matrix = Corpus.convert_flat_cluster_idx_to_matrix(flat_gold_cluster_ids, doc_len)
                iwi_matrix = Corpus.i_within_i(ment_properties)
                ment_word_weights = Corpus.get_word_weight(ment_properties, ment_full_word)

                # clock.tik("constraints")
                # encoder_constrs, decoder_gender_constrs, decoder_number_constrs = \
                #     train_data.get_constraints(ment_genders, ment_numbs, ment_is_not_prp)
                # clock.tok("constraints")

                k = min(10, doc_len - 1)
                # print([self.voca_head.id2word[ment_head_words[i]] for i in range(k)])
                # print(ment_is_not_prp[:k])
                # print(decoder_gender_constrs[:k, :k])
                # print(decoder_number_constrs[:k, :k])
                # print(encoder_constrs[:k, :k])

                # print(doc_len, ment_full_word.shape, np.sum(ment_full_word_lengths))
                # print(ment_full_word_lengths)

                # print(doc_len, flat_pw_feats_indices.shape)
                # print(ment_properties)
                # print(ment_full_word[:k, :])
                # print(ment_word_weights[:k, :])


                # create feed_dict
                feed_dict = {
                    x_phi_a: (flat_a_feats_indices, flat_a_feats_ids_val,
                              [flat_a_feats_indices[:, 0].max(), flat_a_feats_indices[:, 1].max()]),
                    x_phi_a_offsets: a_feats_offsets,
                    x_phi_p: (flat_pw_feats_indices, flat_pw_feats_ids_val,
                              [flat_pw_feats_indices[:, 0].max(), flat_pw_feats_indices[:, 1].max()]),
                    x_phi_p_offsets: pw_feats_offsets,
                    x_ment_full_words: ment_full_word,
                    x_ment_full_words_lengths: ment_full_word_lengths,
                    x_sentences: sentences,
                    x_sentence_lengths: sentence_lengths,
                    x_gold_class_antecedents: flat_gold_antecedents,
                    x_lost_weight_link: flat_lost_weight_antecedent,
                    x_gold_class_antecedents_supgen: ante_matrix,
                    x_gold_class_cluster_ids: flat_gold_cluster_ids,
                    x_lost_weight_cluster: flat_lost_weight_cluster,
                    x_gold_class_cluster_ids_supgen: clus_ids_matrix,
                    x_doc_len: doc_len,
                    x_disc_weight: disc_weight,
                    x_gen_weight: gen_weight,
                    x_unsup_weight: unsup_weight,
                    x_ment_len_ranges: Corpus.split_length_ranges(ment_full_word_lengths),
                    x_gold_anadet_class: flat_gold_anadet_class,
                    x_ment_properties: ment_properties,
                    x_i_within_i: iwi_matrix,
                    x_ment_word_weights: ment_word_weights
                }

                clock.tok("data")

                clock.tik("process")
                #print("unsup")
                _, c = sess.run([optimizer, cost_func], feed_dict=feed_dict)

                if math.isnan(c):
                    raise Exception("The cost is NaN. Move to the next training portion data.")

                # print(hs.shape, ps.shape, log_p.shape)
                # print(ps[0:100])
                # print(group_words)
                # print(group_ments_offsets)
                # print(mean[:k, :k]    )
                # # print(Corpus.convert_flat_cluster_idx_to_matrix(flat_gold_antecedents, doc_len)[:10, :10])
                # print(var[:k, :k])
                # print(p1[:k, :k])

                total_cost += c
                clock.tok("process")
                clock.reset()

                print("\t\t", strftime("%Y-%m-%d %H:%M:%S"), doc_len, "\tcost", c)

                complete = float(doc_id) / train_data.n_documents * 100
                if int(complete) > next_complete:
                    print("\t", strftime("%Y-%m-%d %H:%M:%S"), int(complete), "%\tcost", total_cost)
                    next_complete += 10

            print("\t", strftime("%Y-%m-%d %H:%M:%S"), "epoch", epoch, "\tcost", total_cost)

            save_path = saver.save(sess, args.model_path, global_step=epoch, write_meta_graph=False, write_state=False)
            print("\tModel saved in file: ", save_path, "\n")

    def get_samples(self, log_probs, doc_len, n, gold_clusters):
        samples = np.zeros([doc_len, n])

        # get the best one
        best = [0]
        log_likelihood = 0
        # print("***********************************************")
        for i in range(1, doc_len):
            log_probs_i = log_probs[i, :i + 1]
            gold_i = gold_clusters[i, :i + 1]
            ante_id = np.argmax(log_probs_i)
            # if ante_id < i:
            #     probs[:, i] = -1e10 # eliminate the case that cluster_i appears
            # if ante_id == -1:
            #     ante_id = i
            best.append(ante_id)
            log_likelihood += 0 #np.log(np.sum(probs_i[np.nonzero(gold_i)]))
        #     print(probs[i])
        #     print(gold_antecedents[i])
        #     print('---------------------')
        # print("=================================")

        samples[:, 0] = best
        scores = np.zeros(n)

        # sampling
        # for i in range(1, doc_len):
        #     samples[i, 1:] = np.random.choice(i + 1, n - 1, p=probs[i][0:i + 1])
        #
        #     if gold_antecedents[0] is not None:
        #         scores = scores + gold_antecedents[i][samples[i, :].astype(int)]

        samples = samples.astype(int)

        return samples, scores, log_likelihood

    def predict(self, model_path):
        mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents = self.encoder(eval=True)
        score = log_p_m_link_antecedents

        # global N_ANTE_TO_GEN
        # N_ANTE_TO_GEN = 30
        # mean_score_m_link_antecedents, var_score_m_link_antecedents, log_p_m_link_antecedents = self.encoder()
        # unsup_cost_func = self.decoder(mean_score_m_link_antecedents, var_score_m_link_antecedents,
        #                                log_p_m_link_antecedents, prior_weight=0.)

        # score, cost = self.coref_without_current_mention()

        if model_path is not None:
            print("loading model from file", args.model_path)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

        sess.run(tf.initialize_local_variables())

        data = Corpus(args.dev_na_prefix + "feats.h5",
                      args.dev_na_lex,
                      args.dev_pw_prefix + "feats.h5",
                      args.dev_na_prefix + "offsets.h5",
                      args.dev_pw_prefix + "offsets.h5",
                      args.dev_oracle_cluster)

        fout = open(args.eval_output, "w")
        total_loglikilihood = 0

        total_type_count = np.zeros([4])
        error_type_count = np.zeros([4])

        total_type_count_not_self = np.zeros([4])
        error_type_count_not_self = np.zeros([4])

        total_type_count_self = np.zeros([4])
        error_type_count_self = np.zeros([4])
        wrong_self_score = [[], [], [], []]

        total_smaller_than_self_score = np.zeros([4])

        all_log_p = []

        for doc_id in range(data.n_documents):
            flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                flat_gold_antecedents, flat_gold_cluster_ids, \
                flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                ment_full_word, ment_full_word_lengths, ment_properties, \
                sentences, sentence_lengths, \
                doc_len, antecedents_weights = data.get_doc(doc_id, self.voca)

            ante_matrix = Corpus.convert_flat_cluster_idx_to_matrix(flat_gold_antecedents, doc_len)
            iwi_matrix = Corpus.i_within_i(ment_properties)
            ment_word_weights = Corpus.get_word_weight(ment_properties, ment_full_word)

            # create feed_dict
            feed_dict = {
                x_phi_a: (flat_a_feats_indices, flat_a_feats_ids_val,
                          [flat_a_feats_indices[:, 0].max(), flat_a_feats_indices[:, 1].max()]),
                x_phi_a_offsets: a_feats_offsets,
                x_phi_p: (flat_pw_feats_indices, flat_pw_feats_ids_val,
                          [flat_pw_feats_indices[:, 0].max(), flat_pw_feats_indices[:, 1].max()]),
                x_phi_p_offsets: pw_feats_offsets,
                x_ment_full_words: ment_full_word,
                x_ment_full_words_lengths: ment_full_word_lengths,
                x_sentences: sentences,
                x_sentence_lengths: sentence_lengths,
                x_gold_class_antecedents: flat_gold_antecedents,
                x_lost_weight_link: flat_lost_weight_antecedent,
                x_gold_class_antecedents_supgen: ante_matrix,
                x_gold_class_cluster_ids: flat_gold_cluster_ids,
                x_lost_weight_cluster: flat_lost_weight_cluster,
                x_doc_len: doc_len,
                x_ment_len_ranges: Corpus.split_length_ranges(ment_full_word_lengths),
                x_gold_anadet_class: flat_gold_anadet_class,
                x_ment_properties: ment_properties,
                x_i_within_i: iwi_matrix,
                x_ment_word_weights: ment_word_weights
            }

            # # create feed_dict
            # if doc_id > 100:
            #     break
            # log_p_idv_chain, selected_id = sess.run([self.log_p_idv_chain, self.selected_id],
            #                                         feed_dict=feed_dict)
            #
            # # print ment generation
            # ments = [ list(map(lambda w: self.voca.id2word[w],
            #                    ment_full_word[i, :ment_full_word_lengths[i]]))
            #           for i in range(doc_len) ]
            #
            # for i in range(doc_len):
            #     print(i, ments[i])
            #     for j in range(min(selected_id.shape[1], i + 1)):
            #         print("\t", "\t", log_p_idv_chain[i, j], ments[selected_id[i, j]],
            #               "*" if ante_matrix[i, selected_id[i, j]] == 1 else "")


            # if doc_id > 100:
            #     break
            # self.log_p_idv_chain = tf.zeros(tf.shape(log_p_m_link_antecedents))
            # self.selected_id = tf.ones([x_doc_len, 1], dtype=tf.int32)

            # log_probs, log_p_idv_chain, selected_id = sess.run([log_p_m_link_antecedents, self.log_p_idv_chain,
            #                                                     self.selected_id],
            #                                                    feed_dict=feed_dict)
            #
            # # print ment generation
            # ments = [ list(map(lambda w: self.voca.id2word[w],
            #                    ment_full_word[i, :ment_full_word_lengths[i]]))
            #           for i in range(doc_len) ]
            #
            # for i in range(doc_len):
            #     amax_lp = np.argmax(log_probs[i, :i + 1])
            #     ment_type = ment_properties[i, 4]
            #     total_type_count[ment_type] += 1
            #     if ante_matrix[i, amax_lp] != 1:
            #         print(i, ments[i], ment_properties[i, 4])
            #         error_type_count[ment_type] += 1
            #         print(total_type_count, error_type_count, error_type_count / total_type_count * 100, doc_id)
            #         for j in range(min(selected_id.shape[1], i + 1)):
            #             print("\t", "\t", log_probs[i, selected_id[i, j]], log_p_idv_chain[i, j], ments[selected_id[i, j]],
            #                   "*" if ante_matrix[i, selected_id[i, j]] == 1 else "")


            # if doc_id > 100:
            #     break

            log_probs = sess.run(score, feed_dict=feed_dict)

            all_log_p.append(log_probs)

            for i in range(doc_len):
                amax_lp = np.argmax(log_probs[i, :i + 1])
                ment_type = ment_properties[i, 4]

                total_type_count[ment_type] += 1
                if ante_matrix[i, i] == 1:
                    total_type_count_self[ment_type] += 1
                else:
                    total_type_count_not_self[ment_type] += 1

                if ante_matrix[i, amax_lp] != 1:
                    error_type_count[ment_type] += 1
                    if ante_matrix[i, i] != 1:
                        error_type_count_not_self[ment_type] += 1
                    else:
                        error_type_count_self[ment_type] += 1

                if amax_lp == i and ante_matrix[i, i] != 1:
                    wrong_self_score[ment_type].append(log_probs[i, amax_lp])

                if log_probs[i, i] < -0.02288834:
                    total_smaller_than_self_score[ment_type] += 1


            # select the best
            samples, oracle_scores, log_likelihood = self.get_samples(log_probs, doc_len, 1, ante_matrix)
            best_id = np.argmax(oracle_scores)

            total_loglikilihood += log_likelihood

            for link in samples[:, best_id]:
                fout.write(str(link) + " ")
            fout.write("\n")
            fout.flush()

            if doc_id % 10 == 0:
                print(str(doc_id) + "\r", end="")

        print(total_type_count, error_type_count, error_type_count / total_type_count * 100)
        print(total_type_count_not_self, error_type_count_not_self, error_type_count_not_self / total_type_count_not_self * 100)
        print(total_type_count_self, error_type_count_self, error_type_count_self / total_type_count_self * 100)
        print(total_smaller_than_self_score)

        np.save(args.dev_oracle_cluster + "_log_probs.npy", np.array(all_log_p, dtype=object))

        fout.close()
        print(total_loglikilihood)

    def predict_anaphoric_detect(self, model_path):
        true_pos = 0.
        false_pos = 0.
        true_neg = 0.
        false_neg = 0.

        score_func, _ = self.detect_anaphoricity()

        print("loading model from file", model_path)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        data = Corpus(args.dev_na_prefix + "feats.h5",
                      args.dev_na_lex,
                      args.dev_pw_prefix + "feats.h5",
                      args.dev_na_prefix + "offsets.h5",
                      args.dev_pw_prefix + "offsets.h5",
                      args.dev_oracle_cluster)

        for doc_id in range(data.n_documents):

            # if doc_id > 50:
            #     break

            clock.tik("data")
            flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                flat_gold_antecedents, flat_gold_cluster_ids, \
                flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                ment_full_word, ment_full_word_lengths, ment_positions, \
                sentences, sentence_lengths, \
                ment_types, ment_genders, ment_numbs, \
                doc_len = data.get_doc(doc_id, self.voca)

            # create feed_dict
            feed_dict = {
                x_phi_a: (flat_a_feats_indices, flat_a_feats_ids_val,
                          [flat_a_feats_indices[:, 0].max(), flat_a_feats_indices[:, 1].max()]),
                x_phi_a_offsets: a_feats_offsets,
                x_phi_p: (flat_pw_feats_indices, flat_pw_feats_ids_val,
                          [flat_pw_feats_indices[:, 0].max(), flat_pw_feats_indices[:, 1].max()]),
                x_phi_p_offsets: pw_feats_offsets,
                x_ment_full_words: ment_full_word,
                x_ment_full_words_lengths: ment_full_word_lengths,
                x_sentences: sentences,
                x_sentence_lengths: sentence_lengths,
                x_ment_properties: ment_positions,
                x_ment_types: ment_types,
                x_ment_genders: ment_genders,
                x_ment_numbers: ment_numbs,
                x_gold_class_antecedents: flat_gold_antecedents,
                x_lost_weight_link: flat_lost_weight_antecedent,
                x_gold_class_cluster_ids: flat_gold_cluster_ids,
                x_lost_weight_cluster: flat_lost_weight_cluster,
                x_doc_len: doc_len,
                x_ment_len_ranges: Corpus.split_length_ranges(ment_full_word_lengths),
                x_gold_anadet_class: flat_gold_anadet_class
            }

            # create feed_dict
            score = sess.run(score_func, feed_dict=feed_dict)

            # find links
            links = "0 "
            for i in range(0, doc_len - 1):
                if flat_gold_anadet_class[i] == 1:
                    if score[i] > 0:
                        true_pos += 1
                    else:
                        false_neg += 1
                else:
                    if score[i] > 0:
                        false_pos += 1
                    else:
                        true_neg += 1

            if doc_id % 10 == 0:
                print(str(doc_id) + "\r", end="")

        print(true_pos, true_neg, false_pos, false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print("\nscores:", precision, recall, 2 * precision * recall / (precision + recall))


############################## main ############################

if __name__ == "__main__":
    for arg,value in vars(args).items():
        print(arg, value)

    # voca = Vocabulary()
    # voca.load_from_file(args.voca + "/words.lst")
    voca = Vocabulary.from_file(args.voca)

    # head_embs, gov_embs, ment_embs, ctx_embs = (np.loadtxt(args.voca + "/head.embs", dtype=np.float32),
    #                                             None, None, None)

    word_embs = None #np.loadtxt(args.voca + "/words.embs")

    # head_embs, gov_embs, ment_embs, ctx_embs = \
    #     (np.loadtxt(args.voca + "/head.embs"),
    #      np.loadtxt(args.voca + "/gov.embs"),
    #      np.loadtxt(args.voca + "/ment.embs"),
    #      np.loadtxt(args.voca + "/ctx.embs"))  if args.mode == "train" else (None, None, None, None)
    #
    # print(head_voca.size())
    # print(gov_voca.size())
    # print(ment_voca.size())
    # print(ctx_voca.size())
    #
    # if head_embs is not None:
    #     print("make sure that the following vectors are identical")
    #     print(head_embs[head_voca.get_id("he"), :5])
    #     print(gov_embs[gov_voca.get_id("he"), :5])
    #     print(ment_embs[ment_voca.get_id("he"), :5])
    #     print(ctx_embs[ctx_voca.get_id("he"), :5])

    print("\n----------------------------------------\n")

    os.chdir(PROJECT_PATH + "/src")
    Corpus.load_remap(args.ment_map_file, args.pw_map_file)
    print("creating model")
    model = CoEn(voca, args.a_feat_dim, args.pw_feat_dim)

    if args.mode == "train" or args.mode == "train_cont":
        print("training")
        model.train(args)

    elif args.mode == "eval":
        model.predict(model_path=args.model_path)
        # model.predict_anaphoric_detect(model_path=args.model_path)

    else:
        raise Exception("undefined mode")

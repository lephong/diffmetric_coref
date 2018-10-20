import h5py
import random
import codecs
from functools import reduce
import pickle as marshal

# from vocabulary import Vocabulary
from lm.data_utils import Vocabulary
import tiktok as clock
from default import *

MENT_TYPE = {"PROPER": 0, "NOMINAL": 1, "PRONOMINAL": 2, "DEMONSTRATIVE": 3}
GENDER = {"MALE": 0, "FEMALE": 1, "NEUTRAL": 2, "UNKNOWN": 3}
NUMBER = {"SINGULAR": 0, "PLURAL": 1, "UNKNOWN": 2}

# SNGend=#UNK# 29 43
# SNGend=FEMALE-SN=true 506 44
# SNGend=MALE-SN=true 276 45
# SNGend=NEUTRAL-SN=true 30 46
# SNGend=UNKNOWN-SN=true 57 47
GENDER_CONSTRAINTS = {
    GENDER["MALE"]: [0., 0, 1, 0.2, 1],
    GENDER["FEMALE"]: [0., 1, 0, 0.2, 1],
    GENDER["NEUTRAL"]: [0., 0.2, 0.2, 1, 1],
    GENDER["UNKNOWN"]: [0., 1, 1, 1, 1]
}

# SNNumber=#UNK# 31 12230
# SNNumber=PLU-SN=true 259 12231
# SNNumber=PLURAL-SN=true 58 12232
# SNNumber=SING-SN=true 301 12233
# SNNumber=SINGULAR-SN=true 32 12234
# SNNumber=UNKNOWN-SN=true 566 12235
NUMBER_CONSTRAINTS = {
    NUMBER["SINGULAR"]: [0., 0, 0, 1, 1, 1],
    NUMBER["PLURAL"]: [0., 1, 1, 0, 0, 1],
    NUMBER["UNKNOWN"]: [0., 1, 1, 1, 1, 1]
}

class Corpus:
    an_remap = {}
    an_group_offsets = []
    pw_remap = {}
    pw_group_offsets = []
    an_remap_func = np.frompyfunc(lambda x: Corpus.an_remap[x], 1, 1)
    pw_remap_func = np.frompyfunc(lambda x: Corpus.pw_remap[x], 1, 1)
    gender_feat_offset = 0
    number_feat_offset = 0

    @staticmethod
    def load_remap(anaph_re_mapping_path, pw_re_mapping_path):

        with open(anaph_re_mapping_path, "r") as f:
            prev_feat_name = ""

            for line in f:
                feat_name, org_id, id = line.strip().split()
                feat_name = feat_name.split('=')[0]

                if feat_name != prev_feat_name:
                    Corpus.an_group_offsets.append(int(id))
                    if feat_name == "SNNumber":
                        Corpus.number_feat_offset = int(id)
                    if feat_name == "SNGend":
                        Corpus.gender_feat_offset = int(id)

                Corpus.an_remap[int(org_id)] = int(id)
                prev_feat_name = feat_name

            print(Corpus.an_group_offsets)

        with open(pw_re_mapping_path, "r") as f:
            prev_feat_name = ""
            offset = 0

            for line in f:
                feat_name, org_id, id = line.strip().split()
                feat_name = feat_name.split('=')[0]

                if feat_name != prev_feat_name:
                    Corpus.pw_group_offsets.append(int(id) + offset) # the first feature in a group is always for self-link
                    offset += 1

                Corpus.pw_remap[int(org_id)] = int(id) + offset
                prev_feat_name = feat_name

            print(Corpus.pw_group_offsets)

    def __init__(self, a_feats_path, a_lex_path, pw_feats_path, a_offset_path, pw_offset_path, opc_path):
        print("loading files:", a_feats_path, a_lex_path, pw_feats_path, a_offset_path, pw_offset_path, opc_path)

        f = h5py.File(a_feats_path, "r")
        self.a_feats = f["feats"][:]
        f.close()

        f = h5py.File(pw_feats_path, "r")
        self.pw_feats = f["feats"][:]
        f.close()

        f = h5py.File(a_offset_path, "r")
        self.a_ment_starts = f["ment_starts"][:]
        self.a_doc_starts = f["doc_starts"][:]
        f.close()

        f = h5py.File(pw_offset_path, "r")
        self.pw_ment_starts = f["ment_starts"][:]
        self.pw_doc_starts = f["doc_starts"][:]
        f.close()

        try:
            self.antecedent_weights = np.load(opc_path + "_log_probs.npy")
            print(opc_path + "_log_probs.npy : loaded")
        except:
            self.antecedent_weights = None
            print(opc_path + "_log_probs.npy : not exist")

        if len(self.a_doc_starts) != len(self.pw_doc_starts):
            raise Exception("Corpus.__init__: size not match")

        self.n_documents = len(self.a_doc_starts) - 1
        self.a_doc_lens = [self.a_doc_starts[i + 1] - self.a_doc_starts[i] for i in range(self.n_documents)]
        self.pw_doc_lens = [self.pw_doc_starts[i + 1] - self.pw_doc_starts[i] for i in range(self.n_documents)]

        self.indices = list(range(self.n_documents))

        # reading sentences and position of mentions
        self.raw_docs = []

        f = codecs.open(a_lex_path, "r",encoding='utf-8', errors='ignore')
        for count in range(len(self.a_doc_lens)):
            sentences = []
            ment_words = []
            ment_properties = []
            prev_is_ment = False

            while True:
                line = f.readline().strip()
                if line == "":
                    break

                comps = line.split()

                if len(comps) == 8 and comps[0].isdigit() and comps[1].isdigit() and \
                        comps[2].isdigit() and comps[3].isdigit():
                    sent_id = int(comps[0])
                    start_id = int(comps[1]) + 1 # we add <S> to the begin of each sentence
                    end_id = int(comps[2]) + 1
                    head_id = int(comps[3]) + 1
                    type = MENT_TYPE[comps[4]]
                    number = NUMBER[comps[6]]
                    gender = GENDER[comps[7]]

                    ment_words.append(["<S>"] + sentences[sent_id][start_id:end_id] + ["<S>"])
                    ment_properties.append((sent_id, start_id, end_id, head_id, type, number, gender))

                    prev_is_ment = True
                else:
                    if prev_is_ment:
                        raise Exception("we are reading mentions")
                    sentences.append(["<S>"] + comps + ["<S>"])

            self.raw_docs.append({"sentences": sentences,
                                 "mentions": ment_words,
                                  "ment_properties": ment_properties})
            if len(self.raw_docs[-1]["mentions"]) != self.a_doc_lens[len(self.raw_docs) - 1]:
                raise Exception("size not match")

        f.close()

        # read gold clusters
        self.gold_antecedents = None
        self.gold_clusters = None
        self.gold_is_anaphoric = None
        self.gold_cluster_ids = None

        # if opc_path is None:
        #     return

        from_file = True
        try:
            f = open(opc_path, "r")
        except:
            from_file = False

        self.gold_antecedents = []
        self.gold_cluster_ids = []
        self.gold_clusters = []
        self.gold_is_anaphoric = []
        for i in range(self.n_documents):
            if from_file:
                line = f.readline()
            else:
                line = reduce(lambda x, y: str(x) + ' ' + str(y), range(self.a_doc_lens[i]))

            if line == "":
                raise Exception("Corpus.__init__: numbers of mentions not match")

            n_mentions = self.a_doc_lens[i]
            ga = [0] * (n_mentions)
            gc_id = [0] * (n_mentions)
            gia = [1] * (n_mentions)
            clusters = line.split("|")
            gc = []
            for i in range(len(clusters)):
                id_strs = clusters[i].split(" ")
                cluster = [int(id_strs[j]) for j in range(len(id_strs))]
                gc.append(cluster)

                # the first mention in a  is not anaphoric
                temp = [0] * cluster[0] + [1]
                ga[cluster[0]] = np.asarray(temp)
                gc_id[cluster[0]] = np.asarray(temp)
                gia[cluster[0]] = -1

                for j in range(1, len(cluster)):
                    atd = np.asarray([0] * (cluster[j] + 1))
                    atd[cluster[0]] = 1
                    gc_id[cluster[j]] = atd

                    atd = np.asarray([0] * (cluster[j] + 1))
                    atd[cluster[0:j]] = 1
                    ga[cluster[j]] = atd

            self.gold_antecedents.append(ga)
            self.gold_cluster_ids.append(gc_id)
            self.gold_clusters.append(gc)
            self.gold_is_anaphoric.append(gia)

    def shuffle(self):
        random.shuffle(self.indices)

    def get_doc_id(self, id):
        return self.indices[id]

    def get_doc(self, id, dict, cutoff=-1):
        doc_id = self.indices[id]
        doc_len = self.a_doc_lens[doc_id]

        if self.antecedent_weights is not None:
            antecedent_weights = self.antecedent_weights[doc_id]
            if antecedent_weights.shape[0] != doc_len:
                raise Exception("size not match", antecedent_weights.shape, doc_len)
        else:
            antecedent_weights = np.zeros([doc_len, doc_len])

        if doc_len > cutoff > 0:
            antecedent_weights = antecedent_weights[:cutoff, :cutoff]

        gold_antecedents = []
        gold_cluster_ids = []
        lost_weight_antecedent = []
        lost_weight_cluster = []
        gold_anadet_class = []

        # embeddings
        clock.tik("emb")
        raw_doc = self.raw_docs[doc_id]

        # eom = dict.get_id(Vocabulary.eom_token)
        ment_full_word = [ment[:min(len(ment), MAX_MENT_LEN)] for ment in raw_doc["mentions"]]
        ment_full_word = [np.array([dict.get_id(w) for w in ment])
                          for ment in ment_full_word]
        ment_full_word_lengths = np.array([ment.size for ment in ment_full_word])
        ment_properties = np.array(raw_doc["ment_properties"])

        sentences = [np.array([dict.get_id(w) for w in sent])
                     for sent in raw_doc["sentences"]]
        sentence_lengths = np.array([len(sent) for sent in sentences])

        if doc_len > cutoff > 0:
            ment_full_word = ment_full_word[:cutoff]
            ment_full_word_lengths = ment_full_word_lengths[:cutoff]
            ment_properties = ment_properties[:cutoff]

        ment_max_len = np.max(ment_full_word_lengths)
        ment_full_word = [np.pad(ment_full_word[i], (0, ment_max_len - ment_full_word_lengths[i]),
                                 'constant', constant_values=(0, dict.s_id))
                          for i in range(cutoff if doc_len > cutoff > 0 else doc_len)]
        ment_full_word = np.vstack(ment_full_word)

        sent_max_len = np.max(sentence_lengths)
        sentences = [np.pad(sentences[i], (0, sent_max_len - sentence_lengths[i]), 'constant', constant_values=(0, dict.s_id))
                     for i in range(len(sentences))]
        sentences = np.vstack(sentences)

        clock.tok("emb")

        # read mention features
        clock.tik("ment")
        begin = self.a_ment_starts[self.a_doc_starts[doc_id]]
        end = self.a_ment_starts[self.a_doc_starts[doc_id + 1]]

        a_feats_ids_val = Corpus.an_remap_func(self.a_feats[begin:end])
        a_feats_indices = np.zeros([end - begin, 2], dtype=np.int)
        a_feats_indices[:, 1] = range(end - begin)
        a_feats_offsets = []

        cur_id = self.a_doc_starts[doc_id]
        offset = self.a_ment_starts[cur_id]
        for i in range(self.a_doc_lens[doc_id]):
            b = self.a_ment_starts[cur_id] - offset
            e = self.a_ment_starts[cur_id + 1] - offset
            a_feats_indices[b:e, 0] = i
            a_feats_offsets.append(b)
            cur_id += 1

        a_feats_offsets.append(e)

        if doc_len > cutoff > 0:
            a_feats_indices = a_feats_indices[:a_feats_offsets[cutoff], :]
            a_feats_ids_val = a_feats_ids_val[:a_feats_offsets[cutoff]]
            a_feats_offsets = a_feats_offsets[:cutoff + 1]

        clock.tok("ment")

        # pairwise features
        clock.tik("pw")
        clock.tik("1")
        begin = self.pw_ment_starts[self.pw_doc_starts[doc_id]]
        end = self.pw_ment_starts[self.pw_doc_starts[doc_id + 1]]
        clock.tok("1")

        clock.tik("2")
        pw_feats_indices = np.concatenate((np.zeros([end - begin, 1], dtype=int),
                                           np.reshape(np.arange(end - begin, dtype=int), [-1, 1])),
                                          axis=1)
        pw_feats_offsets = [0]
        clock.tok("2")
        clock.tik("3")
        pw_feats = self.pw_feats[begin:end]
        pw_feats_ids_val = []
        clock.tok("3")

        clock.tik("4")
        cur_id = self.pw_doc_starts[doc_id]
        offset = self.pw_ment_starts[cur_id]
        residual = 0
        for i in range(doc_len - 1):
            b0 = self.pw_ment_starts[cur_id] - offset
            for j in range(i + 1):
                b = self.pw_ment_starts[cur_id] - offset
                e = self.pw_ment_starts[cur_id + 1] - offset
                pw_feats_indices[b:e, 0] = cur_id - self.pw_doc_starts[doc_id] # + i - residual
                cur_id += 1
            pw_feats_ids_val.extend(self.pw_remap_func(pw_feats[b0:e]))

            pw_feats_offsets.append(len(pw_feats_ids_val))
            residual = cur_id - self.pw_doc_starts[doc_id] + i + 1

        if doc_len > cutoff > 0:
            pw_feats_indices = pw_feats_indices[:pw_feats_offsets[cutoff - 1], :]
            pw_feats_ids_val = pw_feats_ids_val[:pw_feats_offsets[cutoff - 1]]
            pw_feats_offsets = pw_feats_offsets[:cutoff]

        clock.tok("4")
        clock.tok("pw")

        clock.tik("gold")
        if self.gold_antecedents is not None:
            for i in range(doc_len):
                ga = self.gold_antecedents[doc_id][i]
                gold_antecedents.append(ga.flatten())

                gc_id = self.gold_cluster_ids[doc_id][i]
                gold_cluster_ids.append(gc_id.flatten())

                gia = self.gold_is_anaphoric[doc_id][i]
                gold_anadet_class.append(gia)

                lw = np.zeros(ga.shape)
                if gia == -1:
                    lw[:-1] = LINK_DELTA_FALSE_LINK
                else:
                    lw[-1] = LINK_DELTA_FALSE_NEW
                    lw[:-1] = LINK_DELTA_WRONG_LINK
                    lw[:-1] = lw[:-1] - ga[:-1] * LINK_DELTA_WRONG_LINK
                lost_weight_antecedent.append(lw.flatten())

                lw = np.zeros(ga.shape)
                if gia == -1:
                    lw[:-1] = CLUSTER_DELTA_FALSE_LINK
                else:
                    lw[-1] = CLUSTER_DELTA_FALSE_NEW
                    lw[:-1] = CLUSTER_DELTA_WRONG_LINK
                    lw[:-1] = lw[:-1] - gc_id[:-1] * CLUSTER_DELTA_WRONG_LINK
                lost_weight_cluster.append(lw.flatten())
        clock.tok("gold")

        if doc_len > cutoff > 0:
            doc_len = cutoff
            if self.gold_antecedents is not None:
                gold_antecedents = gold_antecedents[:cutoff]
                gold_cluster_ids = gold_cluster_ids[:cutoff]
                gold_anadet_class = gold_anadet_class[:cutoff]
                lost_weight_antecedent = lost_weight_antecedent[:cutoff]
                lost_weight_cluster = lost_weight_cluster[:cutoff]

        # convert to flat format
        flat_a_feats_indices = a_feats_indices
        flat_a_feats_ids_val = a_feats_ids_val[:len(a_feats_indices)]
        flat_pw_feats_indices = pw_feats_indices
        flat_pw_feats_ids_val = np.array(pw_feats_ids_val[:len(pw_feats_indices)])

        flat_gold_antecedents = np.concatenate(tuple(gold_antecedents))[1:] if self.gold_antecedents is not None else []
        flat_gold_cluster_ids = np.concatenate(tuple(gold_cluster_ids))[1:] if self.gold_cluster_ids is not None else []
        flat_gold_anadet_class = gold_anadet_class if self.gold_antecedents is not None else []
        flat_lost_weight_antecedent = np.concatenate(tuple(lost_weight_antecedent))[1:] if self.gold_antecedents is not None else []
        flat_lost_weight_cluster = np.concatenate(tuple(lost_weight_cluster))[1:] if self.gold_antecedents is not None else []

        return flat_a_feats_indices, flat_a_feats_ids_val, np.array(a_feats_offsets), \
               flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
               flat_gold_antecedents, flat_gold_cluster_ids, \
               flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
               ment_full_word, ment_full_word_lengths, ment_properties, \
               sentences, sentence_lengths, \
               doc_len, antecedent_weights

    @staticmethod
    def i_within_i(ment_properties):
        doc_len = ment_properties.shape[0]
        iwi = np.zeros([doc_len, doc_len], dtype=np.int32)
        for i in range(doc_len):
            sent_i = ment_properties[i, 0]
            start_i = ment_properties[i, 1]
            end_i = ment_properties[i, 2]
            for j in range(max(0, i - 10), i):
                if sent_i == ment_properties[j, 0]:
                    if start_i <= ment_properties[j, 1] and end_i >= ment_properties[j, 2] \
                            or start_i >= ment_properties[j, 1] and end_i <= ment_properties[j, 2]:
                        iwi[i, j] = 1

        return iwi

    @staticmethod
    def convert_flat_antecedent_to_matrix(flat_gold, doc_len):
        split_idx = np.cumsum(range(2, doc_len))
        gold_antecedents = np.split(flat_gold, split_idx)
        gold_antecedents = [np.zeros(doc_len)] + gold_antecedents #[np.roll(a, 1) for a in gold_antecedents] # epsilon is always at the beginning
        gold_antecedents = np.vstack(
            [np.pad(a, (0, doc_len - a.size), mode="constant") for a in gold_antecedents])

        return gold_antecedents

    @staticmethod
    def convert_flat_cluster_idx_to_matrix(flat_gold, doc_len):
        split_idx = np.cumsum(range(2, doc_len))
        gold_cluster_ids = np.split(flat_gold, split_idx)
        gold_cluster_ids_0 = np.zeros(doc_len)
        gold_cluster_ids_0[0] = 1
        gold_cluster_ids = [gold_cluster_ids_0] + [np.pad(a, (0, doc_len - a.size), mode="constant") for a in gold_cluster_ids]
        gold_cluster_ids = np.vstack(gold_cluster_ids)
        return gold_cluster_ids

    @staticmethod
    def split_length_ranges(ment_lengths):
        # ranges = [0, 3, 5, 8, 12, 20, 1000]
        ranges = [0, 10000]
        ret_ranges = [0]
        for i in range(1, len(ranges)):
            if np.any(np.logical_and(ment_lengths >= ret_ranges[-1], ment_lengths < ranges[i])):
                ret_ranges.append(ranges[i])
        return ret_ranges

    @staticmethod
    def get_gender_number_constraints(ment_properties):
        gender_constr = []
        number_constr = []
        for prop in ment_properties:
            # (sent_id, start_id, end_id, head_id, type, number, gender)
            gender_constr.append(GENDER_CONSTRAINTS[prop[6]])
            number_constr.append(NUMBER_CONSTRAINTS[prop[5]])

        return gender_constr, number_constr


    # def get_constraints(self, ment_genders, ment_numbs, ment_is_not_prp):
    #     doc_len = np.shape(ment_genders)[0]
    #     genders = self.ment_gender_map[2][ment_genders.nonzero()[1]]
    #     numbers = self.ment_num_map[2][ment_numbs.nonzero()[1]]
    #
    #     # constraints for encoder
    #     encoder_constraints = [[(genders[i] & genders[j]) * (numbers[i] & numbers[j]) * ment_is_not_prp[j]
    #                             for j in range(i)] for i in range(doc_len)]
    #     encoder_constraints = [np.pad(a, (0, doc_len - len(a)), mode="constant") for a in encoder_constraints]
    #     encoder_constraints = np.vstack(encoder_constraints)
    #     encoder_constraints = (encoder_constraints > 0).astype(np.float32) + np.identity(doc_len)
    #
    #     # constraints for decoder
    #     decoder_constr_gender = self.ment_gender_map[3][ment_genders.nonzero()[1]]
    #     decoder_constr_number = self.ment_num_map[3][ment_numbs.nonzero()[1]]
    #
    #     return encoder_constraints, decoder_constr_gender, decoder_constr_number

    # @staticmethod
    # def convert_to_binary(na_prefix, na_lex, pw_prefix, oracle_cluster, word_emb_path, output):
    #     voca, embs = Vocabulary.load_word_embeddings(word_emb_path)
    #
    #     data = Corpus(na_prefix + "feats.h5",
    #                   na_lex,
    #                   pw_prefix + "feats.h5",
    #                   na_prefix + "offsets.h5",
    #                   pw_prefix + "offsets.h5",
    #                   oracle_cluster)
    #     docs = []
    #
    #     for doc_id in range(data.n_documents):
    #         docs.append(data.get_doc(doc_id, voca, embs))
    #         if doc_id % 50 == 0:
    #             print(doc_id)
    #
    #     print("here")
    #     marshal.dump(docs, open(output, "wb"))

    @staticmethod
    def get_word_weight(ment_properties, ment_full_words):
        word_weight = np.ones(ment_full_words.shape)
        head_id_in_ment = ment_properties[:, 3] - ment_properties[:, 1] + 1 # because we already added "<S>"
        for i in range(ment_full_words.shape[0]):
            if head_id_in_ment[i] < word_weight.shape[1]:
                word_weight[i, head_id_in_ment[i]] = HEAD_WEIGHT
            else:
                print("head not found")

        return word_weight / HEAD_WEIGHT

    @staticmethod
    def test():
        path = "../data/toy/"
        a_feats_path = path + "train-na-feats.h5"
        pw_feats_path = path + "train-pw-feats.h5"
        a_offset_path = path + "train-na-offsets.h5"
        pw_offset_path = path + "train-pw-offsets.h5"
        opc_path = path + "gold_clusters.txt"
        a_lex_path = path + "train-na-lex.txt"
        voca_path = path + "words.lst"
        an_re_mapping_path = path + "anaphReMapping.txt"
        pw_re_mapping_path = path + "pwReMapping.txt"

        voca = Vocabulary.from_file(voca_path)

        Corpus.load_remap(an_re_mapping_path, pw_re_mapping_path)
        corpus = Corpus(a_feats_path, a_lex_path, pw_feats_path, a_offset_path, pw_offset_path, opc_path)

        flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
            flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
            flat_gold_antecedents, flat_gold_cluster_ids, \
            flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
            ment_full_word, ment_full_word_lengths, ment_properties, \
            sentences, sentence_lengths, \
            doc_len, antecedent_weights = corpus.get_doc(1, voca)

        print(flat_a_feats_indices)
        print(flat_a_feats_ids_val)
        print(a_feats_offsets)
        print("----------------------------")
        print(flat_pw_feats_indices)
        print(flat_pw_feats_ids_val)
        print(pw_feats_offsets)
        print("----------------------------")
        print(flat_gold_antecedents)
        # print(flat_gold_cluster_ids)
        # print(flat_gold_anadet_class)
        print(flat_lost_weight_antecedent)
        # print(flat_lost_weight_cluster)
        # print("----------------------------")
        # print(ment_full_word)
        # print(ment_full_word_lengths)
        # print(sentences)
        # print(sentence_lengths)
        # print(ment_properties)
        #
        # print("----------------------------")
        # ante_matrix = Corpus.convert_flat_antecedent_to_matrix(flat_gold_antecedents, doc_len)
        #
        # print(Corpus.convert_flat_antecedent_to_matrix(flat_gold_antecedents, doc_len))
        # print(Corpus.convert_flat_cluster_idx_to_matrix(flat_gold_cluster_ids, doc_len))
        # print(flat_gold_cluster_ids)
        #
        # idx = np.where(ante_matrix == 1)


if __name__ == "__main__":
    Corpus.test()

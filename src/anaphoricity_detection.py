from entity_coref import *

class AnaDet(CoEn):

    def encoder(self):
        ha = tf.nn.embedding_lookup_sparse(self.link_Wa, x_phi_a, None, combiner="sum")
        ha = tf.tanh(ha + self.link_ba)
        score_m_link_eps = tf.matmul(ha, self.link_v) + self.link_v0

        return score_m_link_eps, tf.constant(0.)

    def supervised_encoder(self, score_m_link_eps):
        class_weights = (1., 1.35)

        # for class +1
        mask = tf.equal(x_gold_anadet_class, 1),
        cost_i_plus = class_weights[1] * \
                      tf.reduce_sum(tf.maximum(0., tf.sub(1., tf.boolean_mask(score_m_link_eps, mask))))

        # for class -1
        mask = tf.equal(x_gold_anadet_class, -1)
        cost_i_minus = class_weights[0] * \
                       tf.reduce_sum(tf.maximum(0., tf.sub(1., -tf.boolean_mask(score_m_link_eps, mask))))

        cost = cost_i_plus + cost_i_minus
        return cost

    def get_cost_function(self):
        scores, _ = self.encoder()
        return self.supervised_encoder(scores)

    def get_optimizer(self, cost_func):
        optimizer = tf.contrib.layers.optimize_loss(cost_func,
                                                    tf.contrib.framework.get_global_step(),
                                                    optimizer='Adagrad',
                                                    learning_rate=args.layer_1_learning_rate,
                                                    clip_gradients=100.0)
        return optimizer

    def predict(self, model_path):
        true_pos = 0.
        false_pos = 0.
        true_neg = 0.
        false_neg = 0.

        score_func, _ = self.encoder()

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

        for doc_id in range(data.n_documents):
            flat_a_feats_indices, flat_a_feats_ids_val, a_feats_offsets, \
                flat_pw_feats_indices, flat_pw_feats_ids_val, pw_feats_offsets, \
                flat_gold_antecedents, flat_gold_cluster_ids, \
                flat_gold_anadet_class, flat_lost_weight_antecedent, flat_lost_weight_cluster, \
                ment_full_word, ment_full_word_lengths, ment_properties, \
                sentences, sentence_lengths, \
                doc_len, antecedents_weights = data.get_doc(doc_id, self.voca)

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
                x_gold_class_antecedents: flat_gold_antecedents * 0, # never use annotations
                x_lost_weight_link: flat_lost_weight_antecedent,
                x_gold_class_cluster_ids: flat_gold_cluster_ids * 0,
                x_lost_weight_cluster: flat_lost_weight_cluster,
                x_doc_len: doc_len,
                x_ment_len_ranges: Corpus.split_length_ranges(ment_full_word_lengths),
                x_gold_anadet_class: np.ones(len(flat_gold_anadet_class)),
                x_ment_properties: ment_properties,
                x_antecedents_weights: antecedents_weights,
                keep_prob: 1.
            }

            # create feed_dict
            score = sess.run(score_func, feed_dict=feed_dict)

            # find links
            links = "0 "
            for i in range(1, doc_len):
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


if __name__ == "__main__":
    for arg,value in vars(args).items():
        print(arg, value)

    voca = Vocabulary.from_file(args.voca)

    word_embs = None #np.loadtxt(args.voca + "/words.embs")

    print("\n----------------------------------------\n")

    os.chdir(PROJECT_PATH + "/src")
    Corpus.load_remap(args.ment_map_file, args.pw_map_file)
    print("creating model")
    model = AnaDet(voca, args.a_feat_dim, args.pw_feat_dim)

    if args.mode == "train" or args.mode == "train_cont":
        model.eval = False
        print("training")
        model.train(args)

    elif args.mode == "eval":
        model.eval = True
        model.predict(model_path=args.model_path)

    else:
        raise Exception("undefined mode")

from entity_coref import *

class AnteRank(CoEn):

    def encoder(self):
        # prepare for computing p(m_j | m_i)
        hp_pair = tf.nn.embedding_lookup_sparse(self.link_Wp_pair, x_phi_p, None, combiner="sum")
        hp_prev_ment = tf.gather(tf.nn.embedding_lookup_sparse(self.link_Wp_prev_ment, x_phi_a, None, combiner="sum"),
                                 tf.slice(DUP_PRE_MENT_ID, [0], [tf.shape(hp_pair)[0]]))
        hp_cur_ment = tf.gather(tf.nn.embedding_lookup_sparse(self.link_Wp_cur_ment, x_phi_a, None, combiner="sum"),
                                tf.slice(DUP_CUR_MENT_ID, [0], [tf.shape(hp_pair)[0]]))
        hp = tf.tanh(hp_pair + hp_cur_ment + hp_prev_ment + self.link_bp)
        hp = tf.nn.dropout(hp, keep_prob)
        gp = tf.matmul(hp, self.link_up)

        # init while loop
        init_i = tf.constant(1)
        scores = tf.TensorArray(dtype=tf.float32, size=x_doc_len, infer_shape=False).\
            write(0, tf.pad([[0.]], [[0, 0], [0, x_doc_len - 1]], mode='CONSTANT'))

        def _time_step(i, scores):
            # compute mean link score
            gp_i = tf.slice(gp, [tf.to_int32((i - 1) * i / 2), 0], [i, -1])
            gp_i = tf.reshape(gp_i, [1, i])

            score_m_i_link_not_eps = gp_i + self.link_u0
            score_m_i_link_not_eps = tf.pad(score_m_i_link_not_eps,
                                                       [[0, 0], [0, x_doc_len - i]], mode='CONSTANT')
            scores = scores.write(i, score_m_i_link_not_eps)

            return i + 1, scores

        _, scores = tf.while_loop(cond=lambda i, *_: tf.less(i, x_doc_len),
                                  body=_time_step,
                                  loop_vars=(init_i, scores))

        scores = scores.concat()
        scores += tf.slice(np.triu(MIN_FLOAT32 * np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), 1),
                           [0, 0], [x_doc_len, x_doc_len])

        scores = tf.reshape(scores, [x_doc_len, x_doc_len])

        return scores, tf.constant(0.)

    def supervised_encoder(self, score_m_link_antecedents):
        def _each_ment(i, link_cost, offset):
            # compute link cost
            gold_class_i = tf.slice(x_gold_class_antecedents, [offset], [i])

            def _body():
                p_m_i = tf.nn.softmax(score_m_link_antecedents[i, :i])
                before_log = tf.reduce_sum(p_m_i * gold_class_i)

                link_cost_i = -tf.reduce_sum(tf.log(before_log))
                return link_cost_i

            link_cost_i = tf.cond(pred=tf.reduce_sum(gold_class_i) > 0,
                                  fn1=lambda: _body(),
                                  fn2=lambda: tf.constant(0.))

            return i + 1, link_cost + link_cost_i, offset + i + 1

        _, link_cost, _ = tf.while_loop(cond=lambda i, *_: tf.less(i, x_doc_len),
                                        body=_each_ment,
                                        loop_vars=(tf.constant(1), tf.constant(0.), tf.constant(0)))

        return link_cost

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

    def initialize(self, saver):
        sess.run(tf.initialize_all_variables())

        # training
        if args.init_model_path is not None:
            print("load init model from", args.init_model_path)
            params = [w for w in tf.contrib.framework.get_variables()
                      if w not in (tf.contrib.framework.get_variables_by_suffix("/Adagrad") +
                                   tf.contrib.framework.get_variables_by_suffix("global_step"))]
            tf.train.Saver(params).restore(sess, args.init_model_path)

        if args.model_path is not None:
            print("loading model from file", args.model_path)
            saver.restore(sess, args.model_path)

        if args.model_path is None:
            args.model_path = args.experiment_dir + "/coref.model"

        sess.run(tf.initialize_local_variables())

    def predict(self, model_path):
        total = 0.
        correct = 0.

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
            offset = 0
            for i in range(1, doc_len):
                if flat_gold_anadet_class[i] == 1:
                    total += 1
                    score_i = score[i, :i]
                    ante_i = flat_gold_antecedents[offset:offset + i]
                    pred = np.argmax(score_i)
                    # print(score_i, ante_i, pred)
                    if ante_i[pred] == 1:
                        correct += 1
                offset += i + 1

            if doc_id % 10 == 0:
                print(str(doc_id) + "\r", end="")

        print("\naccuracy:", correct / total)


if __name__ == "__main__":
    for arg,value in vars(args).items():
        print(arg, value)

    voca = Vocabulary.from_file(args.voca)

    word_embs = None #np.loadtxt(args.voca + "/words.embs")

    print("\n----------------------------------------\n")

    os.chdir(PROJECT_PATH + "/src")
    Corpus.load_remap(args.ment_map_file, args.pw_map_file)
    print("creating model")
    model = AnteRank(voca, args.a_feat_dim, args.pw_feat_dim)

    if args.mode == "train" or args.mode == "train_cont":
        model.eval = False
        print("training")
        model.train(args)

    elif args.mode == "eval":
        model.eval = True
        model.predict(model_path=args.model_path)

    else:
        raise Exception("undefined mode")

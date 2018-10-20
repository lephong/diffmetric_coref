Optimizing Differentiable Relaxations of Coreference Evaluation Metrics
========

This repository contains code for training and running the neural coreference models decribed in the paper:

  Phong Le and Ivan Titov (2017). [Optimizing Differentiable Relaxations of Coreference Evaluation Metrics](https://arxiv.org/pdf/1704.04451). CoNLL.

  Written and maintained by Phong Le (p.le [at] uva.nl)

### Requirements 

You have to have: Python 3.5 or 3.6, Tensorflow 0.7

### Run

Firstly, you have to generate data, following `https://github.com/swiseman/nn_coref` but use our own `modifiedBCS`. Next, run 

	./train_supgen_coref.sh

### Quick use of the idea

If you have a mention-ranking-based system (i.e., a system computes `p(m_i links to m_j)`), you can quickly try our idea using
the following functions. Note that, these functions work on 1-document batches.

Compute `p(m_i \in E_u)` (section 3 of the paper) where `p_m_link` means `p(m_i links to m_j)`


```python

    def compute_p_m_entity(self, p_m_link):
        p_m_entity = tf.concat(1, [[[1.]], tf.zeros([1, x_doc_len - 1])])

        def _time_step(i, p_m_entity):
            p_m_e = p_m_entity[:, :i]
            p_m_link_i = p_m_link[i:i + 1, :i]
            p_m_e_i = tf.matmul(p_m_link_i, p_m_e)
            p_m_e_i = tf.concat(1, [p_m_e_i, p_m_link[i:i + 1, i:i + 1]])
            p_m_e_i = tf.pad(p_m_e_i, [[0, 0], [0, x_doc_len - i - 1]], mode='CONSTANT')
            p_m_entity = tf.concat(0, [p_m_entity, p_m_e_i])
            return i + 1, p_m_entity

        _, p_m_entity = tf.while_loop(cond=lambda i, *_: tf.less(i, x_doc_len),
                                      body=_time_step,
                                      loop_vars=(tf.constant(1), p_m_entity),
                                      shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None])))

        # p_m_entity = tf.Print(p_m_entity, [tf.reduce_sum(p_m_entity, 1)], summarize=1000)

        return p_m_entity

```


Compute losses using B3 or Lea metrics (section 4 of the paper) where `x_gold_class_cluster_ids_supgen` 
is a vector of the gold cluster indices of mentions.


```python

    def compute_b3_lost(self, p_m_entity, beta=2.0):
        # remove singleton entities
        gold_entities = tf.reduce_sum(x_gold_class_cluster_ids_supgen, 0) > 1.2

        sys_m_e = tf.one_hot(tf.argmax(p_m_entity, 1), x_doc_len)
        sys_entities = tf.reduce_sum(sys_m_e, 0) > 1.2

        gold_entity_filter = tf.reshape(tf.where(gold_entities), [-1])
        gold_cluster = tf.gather(tf.transpose(x_gold_class_cluster_ids_supgen), gold_entity_filter)

        sys_entity_filter, merge = tf.cond(pred=tf.reduce_any(sys_entities & gold_entities),
                                           fn1=lambda: (tf.reshape(tf.where(sys_entities), [-1]), tf.constant(0)),
                                           fn2=lambda: (tf.reshape(tf.where(sys_entities | gold_entities), [-1]), tf.constant(1)))
        system_cluster = tf.gather(tf.transpose(p_m_entity), sys_entity_filter)

        # compute intersections
        gold_sys_intersect = tf.pow(tf.matmul(gold_cluster, system_cluster, transpose_b=True), 2)
        r_num = tf.reduce_sum(tf.reduce_sum(gold_sys_intersect, 1) / tf.reduce_sum(gold_cluster, 1))
        r_den = tf.reduce_sum(gold_cluster)
        recall = tf.reshape(r_num / r_den, [])

        sys_gold_intersection = tf.transpose(gold_sys_intersect)
        p_num = tf.reduce_sum(tf.reduce_sum(sys_gold_intersection, 1) / tf.reduce_sum(system_cluster, 1))
        p_den = tf.reduce_sum(system_cluster)
        prec = tf.reshape(p_num / p_den, [])

        beta_2 = beta ** 2
        f_beta = (1 + beta_2) * prec * recall / (beta_2 *  prec + recall)

        lost = -f_beta
        lost = tf.Print(lost, [merge,
                               r_num, r_den, p_num, p_den,
                               gold_entity_filter, sys_entity_filter, #tf.reduce_sum(p_m_entity, 0),
                               beta, recall, prec, f_beta], summarize=1000)

        return tf.cond(pred=tf.reduce_all([r_num > .1, p_num > .1, r_den > .1, p_den > .1]),
                       fn1=lambda: lost,
                       fn2=lambda: tf.stop_gradient(tf.constant(0.)))

    def compute_lea_lost(self, p_m_entity, beta=2.0):
        # remove singleton entities
        gold_entities = tf.reduce_sum(x_gold_class_cluster_ids_supgen, 0) > 1.2

        sys_m_e = tf.one_hot(tf.argmax(p_m_entity, 1), x_doc_len)
        sys_entities = tf.reduce_sum(sys_m_e, 0) > 1.2

        gold_entity_filter = tf.reshape(tf.where(gold_entities), [-1])
        gold_cluster = tf.gather(tf.transpose(x_gold_class_cluster_ids_supgen), gold_entity_filter)

        sys_entity_filter, merge = tf.cond(pred=tf.reduce_any(sys_entities & gold_entities),
                                           fn1=lambda: (tf.reshape(tf.where(sys_entities), [-1]), tf.constant(0)),
                                           fn2=lambda: (tf.reshape(tf.where(sys_entities | gold_entities), [-1]), tf.constant(1)))
        system_cluster = tf.gather(tf.transpose(p_m_entity), sys_entity_filter)

        # compute p(m_i \in sys_v, m_j \in sys_v) = p(m_i \in sys_v) p(m_j \in sys_v)
        p1 = tf.expand_dims(system_cluster, 2)
        p2 = tf.expand_dims(system_cluster, 1)
        p_e_m_m = tf.batch_matmul(p1, p2)

        # compute link(gold_u \intersect sys_v)
        link_gold_sys = tf.TensorArray(dtype=tf.float32, size=tf.shape(gold_cluster)[0], infer_shape=False)
        m_m_not_self = x_gold_class_antecedents_supgen * \
                       tf.slice(np.tril(np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), -1),
                                [0, 0], [x_doc_len, x_doc_len])
        p_e_m_m_not_self = p_e_m_m * \
                           tf.expand_dims(tf.slice(np.tril(np.ones([MAX_N_MENTIONS, MAX_N_MENTIONS], dtype=np.float32), -1),
                                                   [0, 0], [x_doc_len, x_doc_len]), 0)

        def _each_gold_entity(i, link_gold_sys):
            mentions_in_gold_entity_i = tf.reshape(tf.where(gold_cluster[i, :] > 0), [-1])

            m_m_gold_entity_i = tf.gather(m_m_not_self, mentions_in_gold_entity_i)
            m_m_gold_entity_i = tf.transpose(tf.gather(tf.transpose(m_m_gold_entity_i), mentions_in_gold_entity_i))

            e_m_m_sys = tf.gather(tf.transpose(p_e_m_m_not_self, [1, 2, 0]), mentions_in_gold_entity_i)
            e_m_m_sys = tf.gather(tf.transpose(e_m_m_sys, [1, 0, 2]), mentions_in_gold_entity_i)
            e_m_m_sys = tf.transpose(e_m_m_sys, [2, 1, 0])

            link_gold_i_sys = tf.reduce_sum(e_m_m_sys * tf.expand_dims(m_m_gold_entity_i, 0), [1, 2])
            link_gold_sys = link_gold_sys.write(i, link_gold_i_sys)
            return i + 1, link_gold_sys

        _, link_gold_sys = tf.while_loop(cond=lambda i, *_: i < tf.shape(gold_cluster)[0],
                                         body=_each_gold_entity,
                                         loop_vars=(tf.constant(0), link_gold_sys))

        link_gold_sys = link_gold_sys.pack()

        # link(gold_u), link(sys_v)
        gold_cluster_size = tf.reduce_sum(gold_cluster, 1)
        link_gold_cluster = gold_cluster_size * (gold_cluster_size - 1) / 2

        sys_cluster_size = tf.reduce_sum(system_cluster, 1)
        link_sys_cluster = tf.reduce_sum(p_e_m_m_not_self, [1, 2])

        # compute recall, prec, fscore
        r_num = tf.reduce_sum(gold_cluster_size * tf.reduce_sum(link_gold_sys, 1) / link_gold_cluster)
        r_den = tf.reduce_sum(gold_cluster_size)
        recall = tf.reshape(r_num / r_den, [])

        p_num = tf.reduce_sum(sys_cluster_size * tf.reduce_sum(tf.transpose(link_gold_sys), 1) / link_sys_cluster)
        p_den = tf.reduce_sum(sys_cluster_size)
        prec = tf.reshape(p_num / p_den, [])

        beta_2 = beta ** 2
        f_beta = (1 + beta_2) * prec * recall / (beta_2 *  prec + recall)

        lost = -f_beta
        lost = tf.Print(lost, [merge,
                               r_num, r_den, p_num, p_den,
                               gold_entity_filter, sys_entity_filter, #tf.reduce_sum(p_m_entity, 0),
                               recall, prec, f_beta], summarize=1000)

        return tf.cond(pred=tf.reduce_all([r_num > .1, p_num > .1, r_den > .1, p_den > .1]),
                       fn1=lambda: lost,
                       fn2=lambda: tf.stop_gradient(tf.constant(0.)))
```
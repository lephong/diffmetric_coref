import tensorflow as tf

from lm.model_utils import sharded_variable, LSTMCell
from lm.common import assign_to_gpu, average_grads, find_trainable_variables
from lm.hparams import HParams


class LM_whileloop(object):
    def __init__(self, hps, mode="train", ps_device="/gpu:0"):
        self.hps = hps
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.w = tf.placeholder(tf.int32, [None, None])

        losses = []
        tower_grads = []
        xs = tf.split(0, hps.num_gpus, self.x)
        ys = tf.split(0, hps.num_gpus, self.y)
        ws = tf.split(0, hps.num_gpus, self.w)
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i, ps_device)), tf.variable_scope(tf.get_variable_scope(),
                                                                           reuse=True if i > 0 else None):
                loss = self._forward(i, xs[i], ys[i], ws[i])
                losses += [loss]
                if mode == "train":
                    cur_grads = self._backward(loss, summaries=(i == hps.num_gpus - 1))
                    tower_grads += [cur_grads]

        self.loss = tf.add_n(losses) / len(losses)
        tf.scalar_summary("model/loss", self.loss)

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                           trainable=False)

        if mode == "train":
            grads = average_grads(tower_grads)
            optimizer = tf.train.AdagradOptimizer(hps.learning_rate, initial_accumulator_value=1.0)
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
            self.summary_op = tf.merge_all_summaries()
        else:
            self.train_op = tf.no_op()

        if mode in ["train", "eval"] and hps.average_params:
            with tf.name_scope(None):  # This is needed due to EMA implementation silliness.
                # Keep track of moving average of LSTM variables.
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                variables_to_average = find_trainable_variables("LSTM")
                self.train_op = tf.group(*[self.train_op, ema.apply(variables_to_average)])
                self.avg_dict = ema.variables_to_restore(variables_to_average)

    def _forward(self, gpu, x, y, w, compute_loss=True, eval=False):
        hps = self.hps
        batch_size = tf.shape(x)[0]

        self.initial_states = []
        for i in range(hps.num_layers):
            with tf.device("/gpu:%d" % gpu):
                v = tf.get_variable(shape=[1000, hps.state_size + hps.projected_size], trainable=False,
                                    initializer=tf.constant_initializer(0),
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES], name="state_%d_%d" % (gpu, i))
                self.initial_states += [v]

        emb_vars = sharded_variable("emb", [hps.vocab_size, hps.emb_size], hps.num_shards)
        self.emb_vars = emb_vars

        x = tf.nn.embedding_lookup(emb_vars, x)  # [bs, steps, emb_size]
        if hps.keep_prob < 1.0:
            x = tf.nn.dropout(x, hps.keep_prob)

        num_steps = tf.shape(x)[1]

        x = tf.transpose(x, [1, 0, 2]) # [steps, bs, emb_size]
        inputs = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)
        inputs = inputs.unpack(x)

        for i in range(hps.num_layers):
            with tf.variable_scope("lstm_%d" % i):
                cell = LSTMCell(hps.state_size, hps.emb_size, num_proj=hps.projected_size)

            state = self.initial_states[i][:batch_size, :]
            outputs = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

            def _time_step(t, inputs, outputs, state):
                output, state = cell(inputs.read(t), state)
                if hps.keep_prob < 1.0:
                    output = tf.nn.dropout(output, hps.keep_prob)
                outputs = outputs.write(t, output)
                return t + 1, inputs, outputs, state

            _, _, outputs, state = tf.while_loop(cond=lambda t, *_: t < num_steps,
                                                 body=_time_step,
                                                 loop_vars=(tf.constant(0), inputs, outputs, state),
                                                 back_prop=not eval,
                                                 shape_invariants=(tf.constant(0).get_shape(),
                                                                   tf.TensorShape(None),
                                                                   tf.TensorShape(None),
                                                                   tf.TensorShape(None)))

            inputs = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)
            outputs = outputs.pack()
            inputs = inputs.unpack(outputs[:-1, :, :])

            state = tf.pad(state, [[0, 1000 - batch_size], [0, 0]])
            with tf.control_dependencies([tf.assign(self.initial_states[i], state)]):
                inputs = inputs.write(num_steps - 1, tf.identity(outputs[-1, :, :]))

        inputs = inputs.pack()
        inputs = tf.transpose(inputs, [1, 0, 2]) # [bs, steps, dim]

        softmax_w = sharded_variable("softmax_w", [hps.vocab_size, hps.projected_size], hps.num_shards)
        softmax_b = tf.get_variable("softmax_b", [hps.vocab_size])

        full_softmax_w = tf.reshape(tf.concat(1, softmax_w), [-1, hps.projected_size])
        full_softmax_w = full_softmax_w[:hps.vocab_size, :]
        self.softmax_w = tf.transpose(full_softmax_w)
        self.softmax_b = softmax_b

        if compute_loss:
            w = tf.to_float(w)
            inputs = tf.reshape(inputs, [-1, hps.projected_size])

            # Initialization ignores the fact that softmax_w is transposed. That worked slightly better.
            if hps.num_sampled == 0:
                full_softmax_w = tf.reshape(tf.concat(1, softmax_w), [-1, hps.projected_size])
                full_softmax_w = full_softmax_w[:hps.vocab_size, :]

                logits = tf.matmul(inputs, full_softmax_w, transpose_b=True) + softmax_b
                # targets = tf.reshape(tf.transpose(self.y), [-1])
                targets = tf.reshape(y, [-1])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            else:
                targets = tf.reshape(y, [-1, 1])
                loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, tf.to_float(inputs),
                                                  targets, hps.num_sampled, hps.vocab_size)

            loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))
            return loss
        else:
            return inputs

    def _backward(self, loss, summaries=False):
        hps = self.hps

        loss = loss * hps.num_steps

        emb_vars = find_trainable_variables("emb")
        lstm_vars = find_trainable_variables("LSTM")
        softmax_vars = find_trainable_variables("softmax")

        all_vars = emb_vars + lstm_vars + softmax_vars
        grads = tf.gradients(loss, all_vars)
        orig_grads = grads[:]
        emb_grads = grads[:len(emb_vars)]
        grads = grads[len(emb_vars):]
        for i in range(len(emb_grads)):
            assert isinstance(emb_grads[i], tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(emb_grads[i].values * hps.batch_size, emb_grads[i].indices,
                                            emb_grads[i].dense_shape)

        lstm_grads = grads[:len(lstm_vars)]
        softmax_grads = grads[len(lstm_vars):]

        lstm_grads, lstm_norm = tf.clip_by_global_norm(lstm_grads, hps.max_grad_norm)
        clipped_grads = emb_grads + lstm_grads + softmax_grads
        assert len(clipped_grads) == len(orig_grads)

        if summaries:
            tf.scalar_summary("model/lstm_grad_norm", lstm_norm)
            tf.scalar_summary("model/lstm_grad_scale", tf.minimum(hps.max_grad_norm / lstm_norm, 1.0))
            tf.scalar_summary("model/lstm_weight_norm", tf.global_norm(lstm_vars))
            # for v, g, cg in zip(all_vars, orig_grads, clipped_grads):
            #     name = v.name.lstrip("model/")
            #     tf.histogram_summary(name + "/var", v)
            #     tf.histogram_summary(name + "/grad", g)
            #     tf.histogram_summary(name + "/clipped_grad", cg)

        return list(zip(clipped_grads, all_vars))

    @staticmethod
    def get_default_hparams():
        # return HParams(
        #     batch_size=128,
        #     num_steps=20,
        #     num_shards=1,
        #     num_layers=1,
        #     learning_rate=0.2,
        #     max_grad_norm=10.0,
        #     num_delayed_steps=150,
        #     keep_prob=0.9,
        #
        #     vocab_size=42466,
        #     emb_size=512,
        #     state_size=1024,
        #     projected_size=512,
        #     num_sampled=1000,
        #     num_gpus=1,
        #
        #     average_params=True,
        #     run_profiler=False,
        # )

        return HParams(
            batch_size=128,
            num_steps=20,
            num_shards=1,
            num_layers=1,
            learning_rate=0.2,
            max_grad_norm=10.0,
            num_delayed_steps=150,
            keep_prob=0.9,

            vocab_size=42466,
            emb_size=100,
            state_size=100,
            projected_size=100,
            num_sampled=2000,
            num_gpus=1,

            average_params=True,
            run_profiler=False,
        )

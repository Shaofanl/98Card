import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, layers


class DQL(object):
    def __init__(self,
                 input_len,
                 nb_state_token,
                 nb_action_token,
                 nb_hand):
        self.input_len = input_len
        self.nb_state_token = nb_state_token
        self.nb_action_token = nb_action_token
        self.nb_hand = nb_hand

        self.built = False

    def build(self,
              cells=[64, 64],
              basic_cell=rnn.LSTMCell,
              state_emb_dim=50,
              action_emb_dim=50,
              batch_size=1,
              learning_rate=1e-4,
              clip_norm=1.,
              gamma=1.0):
        # state
        single_state = tf.placeholder('int32', [1, self.input_len],
                                      name='single_state')
        batch_state = tf.placeholder('int32', [batch_size, self.input_len],
                                     name='batch_state')
        # state_embW = self.get_preset_state_emb(self.nb_state_token)
        state_embW = tf.Variable(
            tf.random_uniform([self.nb_state_token, state_emb_dim], -1.0, 1.0),
            name='state_embW')
        single_state_emb = tf.nn.embedding_lookup(state_embW, single_state)
        batch_state_emb = tf.nn.embedding_lookup(state_embW, batch_state)

        encoder = rnn.MultiRNNCell([basic_cell(c) for c in cells])
        single_encoder_outputs, _ = \
            tf.nn.dynamic_rnn(encoder, single_state_emb, dtype=tf.float32)
        batch_encoder_outputs, _ = \
            tf.nn.dynamic_rnn(encoder, batch_state_emb, dtype=tf.float32)
        single_last_output = single_encoder_outputs[:, -1, :]
        batch_last_output = batch_encoder_outputs[:, -1, :]

        # action
        action = tf.placeholder('int32', [batch_size], name='action')
        action_embW = tf.Variable(
            tf.random_uniform([self.nb_action_token,
                               action_emb_dim], -1.0, 1.0), name='action_embW')
        action_emb = tf.nn.embedding_lookup(action_embW, action)
        # (bs, a_emb_dim)

        # joint
        joint = tf.concat([batch_last_output, action_emb], axis=-1)

        def decision_fnn(input, reuse):
            with tf.variable_scope('decision', reuse=reuse):
                x = layers.linear(input, 512, scope='d1')
                x = tf.nn.relu(x)
                x = layers.linear(x, 256, scope='d2')
                x = tf.nn.relu(x)
                x = layers.linear(x, 1, scope='qv')
                return x
        qvalue = decision_fnn(joint, reuse=False)

        # expected_qv
        expected_qv = tf.placeholder('float32', [batch_size],
                                     name='expected_qv')
        loss = expected_qv-qvalue[:, 0]
        loss = tf.where(tf.abs(loss) < 1.0,
                        0.5 * tf.square(loss),
                        tf.abs(loss) - 0.5)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=clip_norm)

        # enumerate every actions
        joints = tf.concat([tf.tile(single_last_output,
                                    (self.nb_action_token, 1)),
                            action_embW], axis=-1)
        qvalues = decision_fnn(joints, reuse=True)

        max_qv = tf.reduce_max(qvalues)
        max_qv_ind = tf.argmax(qvalues[:, 0])

        # calculate Q(s, a)
        self.single_state = single_state
        self.batch_state = batch_state
        self.qvalue = qvalue
        self.action = action

        # training
        self.expected_qv = expected_qv
        self.train_op = train_op
        self.loss = loss
        self.batch_size = batch_size
        self.gamma = gamma

        # finding max_{a'} Q(s', a')
        self.qvalues = qvalues
        self.max_qv = max_qv
        self.max_qv_ind = max_qv_ind

        self.built = True

    def train(self,
              env,
              continued=False,
              epsilon=0.1,
              epsilon_decay=0.99,
              epsilon_min=0.1,
              iterations=1000,
              rng=np.random,
              max_exp=200):
        assert (self.built)

        with tf.Session() as sess:
            R = tf.Variable(0.)
            R_summary = tf.summary.scalar('R', R)

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('tensorboard', sess.graph)
            saver = tf.train.Saver()
            if continued:
                saver.restore(sess, 'session/sess.ckpt')

            exp = []
            global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
            i = begin = global_step.eval()
            for i in range(begin, begin+iterations):
                env.reset()
                sess.run(R.assign(0))
                print '[{}, {}]'.format(i, epsilon)
                while True:
                    s = env.state
                    if rng.rand() < epsilon:
                        a = rng.choice(self.nb_action_token)
                    else:
                        a = sess.run(self.max_qv_ind,
                                     feed_dict={self.single_state: [s]})
                    r = env.play(a)
                    s_next = env.state
                    exp.append((s, a, r, s_next))
                    # print s
                    sess.run(R.assign_add(r))
                    if r <= 0:
                        break

                    # sample and update
                    batch_s = []
                    batch_a = []
                    batch_q = []
                    batch_ind = np.random.choice(len(exp),
                                                 size=(self.batch_size,))
                    for exp_ind in batch_ind:
                        s, a, r, s_next = exp[exp_ind]
                        if r > 0:
                            exp_pv = sess.run(self.max_qv,
                                              feed_dict={self.single_state:
                                                         [s_next]})
                            exp_pv = r + self.gamma*exp_pv
                        else:
                            exp_pv = r
                        batch_s.append(s)
                        batch_a.append(a)
                        batch_q.append(exp_pv)
                    feed_dict = {self.batch_state: batch_s,
                                 self.action: batch_a,
                                 self.expected_qv: batch_q}
                    qv, loss, _ = sess.run([self.qvalue,
                                            self.loss,
                                            self.train_op],
                                           feed_dict=feed_dict)
                    # print loss, zip(qv[:, 0], batch_q)
                epsilon = max(epsilon*epsilon_decay, epsilon_min)

                if len(exp) > max_exp:
                    np.random.shuffle(exp)
                    exp = exp[:int(max_exp*0.2)]
                    print 'shuffle'

                train_writer.add_summary(R_summary.eval(), i)

                if i % 100 == 0:
                    saver.save(sess, 'session/sess.ckpt')

    def get_preset_state_emb(self, n):
        emb = np.ones((n+1, n+1), dtype='float32') * (-1)
        emb[0] = 0
        for i in range(1, n+1):
            for j in range(i):
                emb[i, j] = +1
        return tf.Variable(emb, name='state_embW', trainable=False)


if __name__ == '__main__':
    pass

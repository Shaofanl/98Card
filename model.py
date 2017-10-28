import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, layers


class DQL(object):
    # not for general purpose
    def __init__(self,
                 input_len,
                 nb_state_token,
                 nb_action_token,
                 nb_hand,
                 nb_pile,
                 nb_card):
        self.input_len = input_len
        self.nb_state_token = nb_state_token
        self.nb_action_token = nb_action_token
        self.nb_hand = nb_hand
        self.nb_pile = nb_pile
        self.nb_card = nb_card

        self.built = False

    def build(self,
              cells=[256, 256],
              basic_cell=rnn.LSTMCell,
              card_emb_dim=50,
              batch_size=1,
              learning_rate=1e-4,
              clip_norm=1.,
              gamma=1.0):
        # cards (-1: piled, 0: not seen, 1: on hand)
        state1 = tf.placeholder('float32', [None, self.nb_card], name='state1')
        # piles
        state2 = tf.placeholder('int32', [None, self.nb_pile], name='state2')
        # hand
        state3 = tf.placeholder('int32', [None, self.nb_hand], name='state3')
        # action replace (from A[0] to A[1])
        action = tf.placeholder('int32', [None], name='action')
        action_ind_one_hot = tf.one_hot(action,
                                        depth=self.nb_pile*self.nb_hand)

        # embeddings
        # card_embW = self.get_preset_state_emb(self.nb_card, name='card_embW')
        card_embW = tf.Variable(
            tf.random_uniform([self.nb_card, card_emb_dim], -1.0, 1.0),
            name='card_embW')
        pile_emb = tf.nn.embedding_lookup(card_embW, state2)
        hand_emb = tf.nn.embedding_lookup(card_embW, state3)

        encoder = rnn.MultiRNNCell([basic_cell(c) for c in cells])
        pile_emb, _ = \
            tf.nn.dynamic_rnn(encoder, pile_emb, dtype=tf.float32)
        hand_emb, _ = \
            tf.nn.dynamic_rnn(encoder, hand_emb, dtype=tf.float32)

        joint = tf.concat([state1,
                           pile_emb[:, -1, :],
                           hand_emb[:, -1, :]], axis=-1)
        print joint
        with tf.variable_scope('decision'):
            x = layers.linear(joint, 512)
            x = tf.nn.relu(x)
            x = layers.linear(x, 512)
            x = tf.nn.relu(x)
            x = layers.linear(x, self.nb_hand*self.nb_pile, scope='qvs')
        qvalues = x
        qvalue = tf.reduce_sum(x*action_ind_one_hot, 1)

        # expected_qv
        expected_qv = tf.placeholder('float32', [batch_size],
                                     name='expected_qv')
        loss = expected_qv-qvalue
        loss = tf.where(tf.abs(loss) < 1.0,
                        0.5 * tf.square(loss),
                        tf.abs(loss) - 0.5)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=clip_norm)

        # enumerate every actions
        max_qv = tf.reduce_max(qvalues[0])
        max_qv_ind = tf.argmax(qvalues[0])

        # calculate Q(s, a)
        self.state1 = state1
        self.state2 = state2
        self.state3 = state3
        self.action = action
        self.qvalue = qvalue
        self.qvalues = qvalues
        self.action = action

        # training
        self.expected_qv = expected_qv
        self.train_op = train_op
        self.loss = loss
        self.batch_size = batch_size
        self.gamma = gamma

        # finding max_{a'} Q(s', a')
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
            pos_acts = env.all_actions
            for i in range(begin, begin+iterations):
                env.reset()
                sess.run(R.assign(0))
                print '=========[{}, {}]========'.format(i, epsilon)
                while True:
                    s = env.state
                    # print env.piles, env.hand
                    if rng.rand() < epsilon:
                        a = pos_acts[rng.choice(len(pos_acts))]
                    else:
                        feed_dict = {self.state1: [s[0]],
                                     self.state2: [s[1]],
                                     self.state3: [s[2]]}
                        aind = sess.run(self.max_qv_ind,
                                        feed_dict=feed_dict)
                        a = pos_acts[aind]
                    r = env.play(a)
                    s_next = env.state
                    exp.append((s, a, r, s_next))
                    # print s
                    sess.run(R.assign_add(r))
                    if r <= 0:
                        break

                    # sample and update
                    batch_s1 = []
                    batch_s2 = []
                    batch_s3 = []
                    batch_a = []
                    batch_q = []
                    batch_ind = np.random.choice(len(exp),
                                                 size=(self.batch_size,))
                    for exp_ind in batch_ind:
                        s, a, r, s_next = exp[exp_ind]
                        if r > 0:
                            feed_dict = {self.state1: [s[0]],
                                         self.state2: [s[1]],
                                         self.state3: [s[2]]}
                            exp_pv = sess.run(self.max_qv,
                                              feed_dict=feed_dict)
                            exp_pv = r + self.gamma*exp_pv
                        else:
                            exp_pv = r
                        batch_s1.append(s[0])
                        batch_s2.append(s[1])
                        batch_s3.append(s[2])
                        batch_a.append(a)
                        batch_q.append(exp_pv)
                    feed_dict = {self.state1: batch_s1,
                                 self.state2: batch_s2,
                                 self.state3: batch_s3,
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

    def get_preset_state_emb(self, n, name):
        emb = np.ones((n+1, n+1), dtype='float32') * (-1)
        emb[0] = 0
        for i in range(1, n+1):
            for j in range(i):
                emb[i, j] = +1
        return tf.Variable(emb, name=name, trainable=False)


if __name__ == '__main__':
    pass

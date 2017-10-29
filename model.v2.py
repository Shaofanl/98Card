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
        # expected_qv
        expected_qv = tf.placeholder('float32', [None],
                                     name='expected_qv')

        # embeddings
        # card_embW = self.get_preset_state_emb(self.nb_card, name='card_embW')
        card_embW = tf.Variable(
            tf.random_uniform([self.nb_card, card_emb_dim], -1.0, 1.0),
            name='card_embW')
        pile_emb = tf.nn.embedding_lookup(card_embW, state2)
        hand_emb = tf.nn.embedding_lookup(card_embW, state3)

        def DQN(pile_emb, hand_emb,
                action_ind_one_hot,
                expected_qv,
                scope):
            with tf.variable_scope(scope):
                encoder = rnn.MultiRNNCell([basic_cell(c) for c in cells])
                pile_emb, _ = \
                    tf.nn.dynamic_rnn(encoder, pile_emb, dtype=tf.float32)
                hand_emb, _ = \
                    tf.nn.dynamic_rnn(encoder, hand_emb, dtype=tf.float32)

                joint = tf.concat([state1,
                                   pile_emb[:, -1, :],
                                   hand_emb[:, -1, :]], axis=-1)
                x = layers.linear(joint, 1024)
                x = tf.nn.relu(x)
                x = layers.linear(x, 512)
                x = tf.nn.relu(x)
                x = layers.linear(x, 512)
                x = tf.nn.relu(x)
                x = layers.linear(x, self.nb_hand*self.nb_pile, scope='qvs')
                qvalues = x
                qvalue = tf.reduce_sum(x*action_ind_one_hot, 1)

                loss = expected_qv-qvalue
                loss = tf.where(tf.abs(loss) < 1.0,
                                0.5 * tf.square(loss),
                                tf.abs(loss) - 0.5)
                loss = tf.reduce_mean(loss)

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                train_op = tf.contrib.slim.learning.create_train_op(
                    loss, optimizer, clip_gradient_norm=clip_norm)
            return qvalue, qvalues, train_op, loss
        dqn1 = DQN(pile_emb, hand_emb, action_ind_one_hot, expected_qv, 'dqn1')
        dqn2 = DQN(pile_emb, hand_emb, action_ind_one_hot, expected_qv, 'dqn2')

        # enumerate every actions
        # max_qv = tf.reduce_max(qvalues, 1)
        # max_qv_ind = tf.argmax(qvalues, 1)

        # inputs
        self.state1 = state1
        self.state2 = state2
        self.state3 = state3
        self.action = action
        self.expected_qv = expected_qv

        self.qvalue, self.qvalues, self.train_op, self.loss = \
            zip(dqn1, dqn2)

        self.batch_size = batch_size
        self.gamma = gamma

        # finding max_{a'} Q(s', a')
        # self.max_qv = max_qv
        # self.max_qv_ind = max_qv_ind

        self.built = True

    @property
    def rand_dqn(self):
        return np.random.randint(2)

    @staticmethod
    def findmax(all_qvs, fes_a):
        fes_qvs = all_qvs[fes_a]
        fes_rank = fes_qvs.argsort()[::-1]
        return fes_a[fes_rank[0]]

    def train(self,
              env,
              continued=False,
              epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.1,
              iterations=1000,
              rng=np.random,
              max_exp=200):
        assert (self.built)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('tensorboard', sess.graph)
            saver = tf.train.Saver()
            if continued:
                saver.restore(sess, 'session/sess.ckpt')

            exp = []
            global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
            i = begin = global_step.eval()
            # pos_acts = env.all_actions
            for i in range(begin, begin+iterations):
                env.reset()
                R = 0
                print '=========[{}, {}]========'.format(i, epsilon)
                while True:
                    s = env.state
                    fes_a = env._next_fes
                    # print env.piles, env.hand
                    if rng.rand() < epsilon:
                        a = fes_a[rng.choice(len(fes_a))]
                        # a = pos_acts[rng.choice(len(pos_acts))]
                    else:
                        feed_dict = {self.state1: [s[0]],
                                     self.state2: [s[1]],
                                     self.state3: [s[2]]}
                        all_qv = sess.run(self.qvalues[self.rand_dqn],
                                          feed_dict=feed_dict)
                        a = self.findmax(all_qv[0], fes_a)
                    r = env.play(a)
                    s_next = env.state
                    exp.append((s, a, r, s_next, len(env._next_fes) == 0))
                    # print s
                    R += r
                    # print env.hand, '\t', env.piles
                    if len(env._next_fes) == 0:
                        print R
                        break

                    # sample and update
                    batch_a = []
                    batch_q = []
                    batch_ind = np.random.choice(len(exp),
                                                 size=(self.batch_size,))
                    batch_s1 = [exp[j][3][0] for j in batch_ind]
                    batch_s2 = [exp[j][3][1] for j in batch_ind]
                    batch_s3 = [exp[j][3][2] for j in batch_ind]
                    estimate_dqn_ind = self.rand_dqn
                    trained_dqn_ind = 1-estimate_dqn_ind
                    batch_all_qv = sess.run(self.qvalues[estimate_dqn_ind],
                                            feed_dict={self.state1: batch_s1,
                                                       self.state2: batch_s2,
                                                       self.state3: batch_s3})
                    batch_s1 = []
                    batch_s2 = []
                    batch_s3 = []
                    for ind, exp_ind in enumerate(batch_ind):
                        s, a, r, s_next, teriminated = exp[exp_ind]
                        if teriminated:
                            exp_pv = r
                        else:
                            fes_a = env.fesible_actions(s_next)
                            a = self.findmax(batch_all_qv[ind], fes_a)
                            exp_pv = r + self.gamma*batch_all_qv[ind, a]
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
                    sess.run(self.train_op[trained_dqn_ind],
                             feed_dict=feed_dict)
                    # print loss, zip(qv[:, 0], batch_q)
                epsilon = max(epsilon*epsilon_decay, epsilon_min)

                if len(exp) > max_exp:
                    np.random.shuffle(exp)
                    exp = exp[:int(max_exp*0.2)]
                    print 'shuffle'

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="R", simple_value=R),
                ])
                train_writer.add_summary(summary, i)

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

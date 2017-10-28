import numpy as np
from itertools import product
from copy import deepcopy


class Environment(object):
    def __init__(self,
                 # game parameters
                 cards_range=(2, 98),
                 deal_after=2,
                 nb_hand=8,
                 nb_asc_pile=2,
                 nb_des_pile=2,
                 reverse_scale=10,
                 # rewards
                 punishment=-10,
                 rewardcard=+1,):
        self.cards_range = cards_range
        self.deal_after = deal_after
        self.nb_hand = nb_hand
        self.nb_asc_pile = nb_asc_pile
        self.nb_des_pile = nb_des_pile
        self.nb_pile = nb_asc_pile+nb_des_pile
        self.reverse_scale = reverse_scale

        self.punishment = punishment
        self.rewardcard = rewardcard
        self.nb_action_token = nb_hand*self.nb_pile
        self.nb_state_token = self.nb_card = cards_range[1]+1
        self.nb_state_len = nb_hand+self.nb_pile

        self.all_actions = range(self.nb_pile*self.nb_hand)

        self.reset()

    def reset(self, rng=np.random):
        self.deck = np.arange(self.cards_range[0],
                              self.cards_range[1]+1)
        rng.shuffle(self.deck)
        self.nb_dealt = 0
        self.piled = np.zeros((self.nb_card))
        self.nb_empty = self.nb_hand

        self.hand = [0 for i in range(self.nb_hand)]
        self.piles = [0 for i in range(self.nb_asc_pile+self.nb_des_pile)]

        self.deal()

    def deal(self):
        # all cards are dealt
        if self.nb_dealt == len(self.deck):
            return self.hand

        if self.nb_empty >= self.deal_after:
            while self.nb_dealt < len(self.deck) and \
                  self.nb_empty > 0:
                self.hand[self.hand.index(0)] = self.deck[self.nb_dealt]
                self.piled[self.deck[self.nb_dealt]] = 1.0
                self.nb_empty -= 1
                self.nb_dealt += 1
        return self.hand

    def play(self, action):
        '''
            Action is in the form of (index_of_card,
                                      index_of pile (asc first))
            For instance (0, 3) is
                `pile the first (0) card in hand to the fourth (3) pile.`

            return value <= 0 means gameover:
                =0: All cards are served
                <0: Illegal actions
        '''
        if isinstance(action, int):
            action = (action / self.nb_hand, action % self.nb_hand)

        # print action
        assert(0 <= action[0] < self.nb_pile)
        assert(0 <= action[1] < self.nb_hand)

        card = self.hand[action[1]]
        top = self.piles[action[0]]
        if card == 0:
            return self.punishment

        if action[0] < self.nb_asc_pile:
            # asc pile
            if top == 0 or card > top or card == top-self.reverse_scale:
                self.piles[action[0]] = card
            else:
                return self.punishment
        else:
            if top == 0 or card < top or card == top+self.reverse_scale:
                self.piles[action[0]] = card
            else:
                return self.punishment
        self.piled[card] = -1.0
        self.hand[action[1]] = 0
        self.nb_empty += 1
        self.deal()
        return self.rewardcard

    def possible_actions(self, s):
        return self.all_actions

    @property
    def state(self):
        return self.piled, self.piles, self.hand

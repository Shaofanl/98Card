from env import Environment
from model import DQL


if __name__ == '__main__':
    env = Environment(cards_range=(2, 30),
                      punishment=-1,
                      nb_hand=4,
                      nb_asc_pile=1,
                      nb_des_pile=1)
    model = DQL(input_len=env.nb_state_len,
                nb_state_token=env.nb_state_token,
                nb_action_token=env.nb_action_token,
                nb_hand=env.nb_hand,
                nb_pile=env.nb_pile,
                nb_card=env.nb_card)
    model.build(
                cells=[64, 128],
                batch_size=64,
                learning_rate=1e-4,
                card_emb_dim=50,
                gamma=1.0)
    model.train(env,
                iterations=500000,
                epsilon=1.00,
                epsilon_decay=1-1e-4,
                epsilon_min=0.1,
                max_exp=100000)

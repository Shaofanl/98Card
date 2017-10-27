from env import Environment
from model import DQL


if __name__ == '__main__':
    env = Environment(cards_range=(2, 30),
                      punishment=-1)
    model = DQL(input_len=env.nb_state_len,
                nb_state_token=env.nb_state_token,
                nb_action_token=env.nb_action_token,
                nb_hand=env.nb_hand)
    model.build(
                cells=[256, 256],
                batch_size=32,
                learning_rate=1e-3,
                state_emb_dim=128,
                action_emb_dim=128,
                gamma=0.95)
    model.train(env,
                iterations=500000,
                epsilon=0.9,
                epsilon_decay=1-1e-3,
                epsilon_min=0.2,
                max_exp=500)

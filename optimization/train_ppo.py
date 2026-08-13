"""Train PPO: python -m optimization.train_ppo --template cross_basic --route <rou.xml>"""
from optimization.train_common import build_arg_parser, train

if __name__ == "__main__":
    train("ppo", build_arg_parser("ppo").parse_args())

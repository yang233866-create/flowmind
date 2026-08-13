"""Train DQN: python -m optimization.train_dqn --template cross_basic --route <rou.xml>"""
from optimization.train_common import build_arg_parser, train

if __name__ == "__main__":
    train("dqn", build_arg_parser("dqn").parse_args())

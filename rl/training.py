from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from RL_environment import RL_setup


def build_environment():
    env = RL_setup()
    return env

def training(env, total_timesteps=500_000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def save(model, filename="ppo_reactor"):
    model.save(filename)

if __name__ == "__main__":
    env = build_environment()
    model = training(env)
    save(model)

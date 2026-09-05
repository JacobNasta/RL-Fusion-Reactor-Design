from scipy.stats import qmc
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from particle import Particle
from reactor import reactor
from reactor import power
from reactor import stability


param_limits = {
    "major_radius": (0, 8),
    "minor_radius": (0.05, 8),
    "elongation": (0.5, 2.5),
    "triangularity": (-0.8, 0.8),
    "squareness": (-0.5, 0.5),
    "B": (0.1, 20),
    "n": (1e18, 1e21),
    "T": (1e7, 3e8),
}

actions_general = {
    0: ("major_radius", 0.25), 1: ("major_radius", -0.25),
    2: ("minor_radius", 0.25), 3: ("minor_radius", -0.25),
    4: ("elongation", 0.1), 5: ("elongation", -0.1),
    6: ("triangularity", 0.05), 7: ("triangularity", -0.05),
    8: ("squareness", 0.05), 9: ("squareness", -0.05),
    10: ("B", 0.5), 11: ("B", -0.5),
    14: ("T", 5e6), 15: ("T", -5e6),
}

actions_fuel = {16, 17, 18, 19, 20, 21, 22, 23}


class RL_setup(gym.Env): # turns the parameters into an environment with actions, rewards, etc for an agent to train off of 

    def __init__(self):
        super().__init__()
        self.reactor = reactor()
        self.power = power()
        self.stability = stability()
        self.brem_scaling = 1e-38
        self.synch_scaling = 1e-34
        self.surface_scaling = 1e5
        self.parameters = None
        self.current_step = 0
        self.max_steps = 100

        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

    # sample the initial parameters 
    def build_parameters_from_sample(self, row):
        fuel1 = Particle(int(round(row[0])), int(round(row[1])))
        fuel2 = Particle(int(round(row[2])), int(round(row[3])))

        no_hole = row[4] < 0.2
        size = row[5]
        radius_fraction = row[6]

        if no_hole:
            major_radius = 0
            minor_radius = radius_fraction * size
        else:
            major_radius = size
            minor_radius = radius_fraction * major_radius

        return {
            "fuel1": fuel1,
            "fuel2": fuel2,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "elongation": row[7],
            "triangularity": row[8],
            "squareness": row[9],
            "B": row[10],
            "n": row[11],
            "T": row[12],
        }

    def initial_environment(self, number):
        min_values = np.array([1, 0, 1, 0, 0, 1, 0.05, 0.5, -0.8, -0.5, 0.1, 1e18, 1e7])
        max_values = np.array([3, 4, 3, 4, 1, 8, 0.95, 2.5, 0.8, 0.5, 20, 1e21, 3e8])

        sample = qmc.LatinHypercube(d=13, seed=self.np_random).random(number)
        sample = qmc.scale(sample, min_values, max_values)

        environments = []
        for row in sample:
            parameters = self.build_parameters_from_sample(row)
            environments.append(self.environment_update(parameters))
        return environments

    def consequence(self, parameters):
        V = self.reactor.geometry.volume(
            parameters["major_radius"], parameters["minor_radius"],
            parameters["elongation"], parameters["triangularity"], parameters["squareness"]
        )
        surface_area = self.reactor.geometry.surface_area(
            parameters["major_radius"], parameters["minor_radius"],
            parameters["elongation"], parameters["triangularity"], parameters["squareness"]
        )
        cross_section = self.reactor.geometry.cross_section(
            parameters["minor_radius"], parameters["elongation"],
            parameters["triangularity"], parameters["squareness"]
        )
        volume_to_surface = self.reactor.geometry.volume_to_surface(
            parameters["major_radius"], parameters["minor_radius"],
            parameters["elongation"], parameters["triangularity"], parameters["squareness"]
        )
        beta = self.stability.beta(parameters["B"], parameters["n"], parameters["T"])
        confinement_time = self.stability.confinement_time(
            parameters["fuel1"].Z + parameters["fuel2"].Z, parameters["n"], parameters["T"],
            parameters["B"], V, surface_area, self.brem_scaling, self.synch_scaling, self.surface_scaling
        )
        total_power = self.power.total_power(
            parameters["fuel1"], parameters["fuel2"], parameters["n"], V, parameters["T"],
            parameters["B"], surface_area, self.brem_scaling, self.synch_scaling, self.surface_scaling
        )

        return {
            "volume": V,
            "surface_area": surface_area,
            "cross_section": cross_section,
            "volume_to_surface": volume_to_surface,
            "beta": beta,
            "confinement_time": confinement_time,
            "total_power": total_power,
        }

    # rewards for the agent 
    def power_reward(self, total_power):
        return np.sign(total_power) * np.log1p(abs(total_power) / 1e6)

    def beta_penalty(self, beta, target_beta=0.05):
        penalty = abs(beta - target_beta) / target_beta
        return min(penalty, 20)

    def confinement_reward(self, confinement_time):
        return np.log1p(max(0, confinement_time))

    def reward(self, parameters):
        return (
            self.power_reward(parameters["total_power"])
            + self.confinement_reward(parameters["confinement_time"])
            - self.beta_penalty(parameters["beta"])
        )

    def environment_update(self, parameters):
        consequences = self.consequence(parameters)
        parameters = {**parameters, **consequences}
        parameters["reward"] = self.reward(parameters)
        return parameters

    # build the state 
    def get_state(self):
        p = self.parameters
        return np.array([
            p["fuel1"].Z / 3, p["fuel1"].N / 4,
            p["fuel2"].Z / 3, p["fuel2"].N / 4,
            p["major_radius"] / 8, p["minor_radius"] / 8,
            p["elongation"] / 2.5, p["triangularity"] / 0.8, p["squareness"] / 0.5,
            p["B"] / 20, (np.log10(p["n"]) - 18) / 3, p["T"] / 3e8,
            p["volume"] / 1000, p["cross_section"] / 500, p["volume_to_surface"] / 10,
            p["beta"],
            np.sign(p["total_power"]) * np.log1p(abs(p["total_power"]) / 1e6),
        ], dtype=np.float32)

    # actions for the agent to explore different states 
    def apply_fuel_action(self, action):
        fuel_key = "fuel1" if action < 20 else "fuel2"
        Z, N = self.parameters[fuel_key].Z, self.parameters[fuel_key].N
        if action in (16, 20):
            Z = min(Z + 1, 3)
        elif action in (17, 21):
            Z = max(Z - 1, 1)
        elif action in (18, 22):
            N = min(N + 1, 4)
        elif action in (19, 23):
            N = max(N - 1, 0)
        self.parameters[fuel_key] = Particle(Z, N)

    def apply_action(self, action):
        if action in actions_general:
            param, delta = actions_general[action]
            self.parameters[param] += delta
        elif action == 12:
            self.parameters["n"] *= 1.1
        elif action == 13:
            self.parameters["n"] /= 1.1
        elif action in actions_fuel:
            self.apply_fuel_action(action)

    def clip_parameters(self):
        for name, (low, high) in param_limits.items():
            self.parameters[name] = np.clip(self.parameters[name], low, high)
        if self.parameters["major_radius"] > 0:
            self.parameters["minor_radius"] = min(self.parameters["minor_radius"], self.parameters["major_radius"])

    # using gym to carry out heavy lifting
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.parameters = self.initial_environment(1)[0]
        return self.get_state(), {}

    def step(self, action):
        self.current_step += 1

        self.apply_action(action)
        self.clip_parameters()
        self.parameters = self.environment_update(self.parameters)

        reward = self.parameters["reward"]
        terminated = False
        if not np.isfinite(reward):
            reward = -1000
            terminated = True

        truncated = self.current_step >= self.max_steps
        state = np.nan_to_num(self.get_state(), nan=0, posinf=10, neginf=-10)

        return state, reward, terminated, truncated, {}

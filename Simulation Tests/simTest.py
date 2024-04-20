from datetime import timedelta, datetime
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.env import T1DSimEnv
import numpy as np
import gym
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
env = gym.make('InsulinEnv-v1')

action = Action(basal=0.0, bolus=0.0)

env.step(action)
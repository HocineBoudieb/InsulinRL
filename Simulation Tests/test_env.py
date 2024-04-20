import gym
import time
from RL.InsulinEnv import InsulinEnv
from simglucose.controller.base import Action

gym.register(id='InsulinEnv-v0', entry_point=InsulinEnv)

env = gym.make('InsulinEnv-v0')
obs = env.reset()

#env.render()
done = False
#input("Press Enter to continue...")
action = Action(basal=0.0, bolus=0.0)

#env.step(action)
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
    #wait 3 seconds
    #time.sleep(10)
    input("Press Enter to continue...")
print("Simulation done")
    
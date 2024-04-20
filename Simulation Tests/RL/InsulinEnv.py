import gym
from gym import spaces
import numpy as np
import functions.PAmodel as GIM
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output

class InsulinEnv(gym.Env):
    def __init__(self):
        self.time = 0
        self.action_space = spaces.Discrete(40)  # assuming insulin can be 0-40 muU/min
        self.observation_space = spaces.Box(low=0, high=400, shape=(1441,))  # assuming glucose level is observed

        #initialize subplots
        self.glucose = np.zeros(1441)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        plt.ion()
        # Initialize state variables
        self.Gb = 120
        self.Ib = 12
        self.weight = 70
        self.intensity = 4317
        self.duration = 60
        self.pa_start = 240
        self.t_meal = [60,360,720]
        self.D = [4e4]
        self.t_sim = 1441
        self.t = np.arange(self.t_sim)
        self.meal = np.zeros(self.t_sim)
        self.meal[self.t_meal] = self.D
        self.u = np.zeros(self.t_sim)
        self.AC = np.zeros(self.t_sim)
        self.AC[self.pa_start:self.pa_start+self.duration] = self.intensity
        self.params = pd.read_csv('parameters/params_standard.csv', header=None,
                                  dtype={0: str}, delimiter=';').set_index(0).squeeze().to_dict()

    def step(self, action):
        self.u[self.time] = action
        model = GIM.compute_glucose_simulation(self.params, [self.Gb, self.Ib], self.weight, self.AC, self.meal, u=self.u,
                                               bolus_calc=True)
        glucose = model['G']
        self.glucose = glucose
        self.time += 30
        done = self.time >= self.t_sim-1
        if self.time <1441-120:
            glucose_future_ = self.glucose[self.time+60:self.time+120]
        else:
            glucose_future_ = self.glucose[self.time:]

        #get mean on glucose_future
        glucose_future = np.mean(glucose_future_)
        # reward function
            
        
        if (glucose_future > 70):
            if (glucose_future < 350):
                reward = 1
            else:
                if (glucose_future < 1000):
                    reward = 0
                else:
                    reward = -100
        else:
            if (glucose_future > 10):
                reward = 0
            else:
                reward = -100
        print("reward:", reward)
        print("time:", self.time)
        return glucose, reward, done, {}  # assuming a reward of negative absolute difference from 120

    def reset(self):
        self.time = 0
        self.u = np.zeros(self.t_sim)
        return self.step(0)[0]  # return initial observation
    
    def render(self):
        # Plot the glucose level
        clear_output(wait=True)
        self.ax.clear()
        self.ax.plot(self.glucose)
        self.ax.set_xlabel('Time (min) = '+str(self.time))
        self.ax.set_ylabel('Glucose level (mg/dl)')
        self.ax.set_title('Glucose level over time')
        self.ax.grid(True)
        # Display the plot
        plt.show()
        
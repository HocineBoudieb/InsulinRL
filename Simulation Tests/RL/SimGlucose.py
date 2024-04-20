from simglucose.simulation.user_interface import simulate
from datetime import timedelta, datetime
from simglucose.controller.base import Controller, Action
#import T1DPatient
from simglucose.patient.t1dpatient import T1DPatient
#import SimObj
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
#import CGM
from simglucose.sensor.cgm import CGMSensor
#import InsulinPump
from simglucose.actuator.pump import InsulinPump
#import T1DSimEnv
from simglucose.simulation.env import T1DSimEnv
#import gym
import gym
#import plot
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#import stable baselines PPO
#from stable_baselines3 import PPO

from simglucose.simulation.scenario_gen import RandomScenario


#Actor Critic Network
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow.keras as keras


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='relu')#relu parce qu'on a des actions continues positives

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi

#Controller for RL
class RlController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
        #actor critic network
        self.gamma = 0.99
        self.n_actions = 2 #basal and bolus
        self.action_space = [0, 10]
        self.actor_critic = ActorCriticNetwork(self.n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=0.001))
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        _, probs = self.actor_critic(state)
        bolus_prob = tfp.distributions.Normal(probs[0], 1)
        basal_prob = tfp.distributions.Normal(probs[1], 1)
        action_bolus = bolus_prob.sample()
        action_basal = basal_prob.sample()
        self.action = [action_bolus, action_basal]
        return [action_basal, action_bolus]

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        self.state = observation
        action = self.choose_action(observation)
        action = Action(basal=action[1], bolus=action[0])
        return action
    
    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            bolus_prob = tfp.distributions.Normal(probs[0], 1)
            basal_prob = tfp.distributions.Normal(probs[1], 1)
            log_bolus_prob = bolus_prob.log_prob(self.action[0])
            log_basal_prob = basal_prob.log_prob(self.action[1])

            delta = reward + \
                self.gamma*state_value_*(1-int(done)) - state_value
            actor_bolus_loss = -log_bolus_prob*delta
            actor_basal_loss = -log_basal_prob*delta
            actor_loss = actor_bolus_loss + actor_basal_loss
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        params = self.actor_critic.trainable_variables
        grads = tape.gradient(total_loss, params)
        self.actor_critic.optimizer.apply_gradients(zip(grads, params))

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state


ctrller = RlController(0)

#custom reward function

def custom_reward(BG_last_hour):
    glucose_future = BG_last_hour[-1]
    if glucose_future > 70:
        if glucose_future < 350:
            return 1
        else:
            if glucose_future < 1000:
                return 0
            else:
                return -100
    else:
        if glucose_future > 10:
            return 0
        else:
            return -100


#Env for RL

"""class SimGlEnv(gym.Wrapper):
    def __init__(self, sim_instance, controller):
        super(SimGlEnv, self).__init__(sim_instance)
        self.controller = controller
        self.glucose = []

    def step(self, action):
        #action = self.controller.policy()
        self.time = self.env.time
        self.glucose = self.env.CGM
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
    
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
    """

RANDOM_SEED = 25
sim_start_time = datetime.now()
sim_run_time = timedelta(hours=24)
sim_scenario = RandomScenario(start_time = sim_start_time, seed = RANDOM_SEED)

patientID = 12
patient = T1DPatient.withID(12)

sim_sensor = CGMSensor.withName('Dexcom')

sim_pump = InsulinPump.withName('Insulet')

environment = T1DSimEnv(patient, sim_sensor, sim_pump, sim_scenario)

results_path = './results/'
simulator = SimObj(
environment,
ctrller,
sim_run_time,
animate=False,
path = results_path
)
num_episodes = 10
max_steps_per_episode = 1441
gym.register(id='InsulinEnv-v1', entry_point='simglucose.envs:T1DSimEnv', kwargs={'patient_name':'adolescent#002','reward_fun': custom_reward})
env = gym.make('InsulinEnv-v1')
obs = env.reset()
done = False
for i_episode in range(num_episodes):
    observation = env.reset()
    for t in range(max_steps_per_episode):
        env.render()
        action = ctrller.policy(observation, reward, done, **info)
        observation, reward, done, info = env.step(action)
        ctrller.learn(observation, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

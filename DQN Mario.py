import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import random
import tqdm
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-8-3-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

inx, iny, inc = env.observation_space.shape
inx = int(inx / 8)
iny = int(iny / 8)
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 200  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 64  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'Mario'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
AGGREGATE_STEPS_EVERY = 50  # episodes
SHOW_PREVIEW = True

# Exploration settings
EPSILON = 1  # not a constant, going to be decayed

class QNetwork(nn.Module):
    def __init__(self):
		super(QNetwork, self).__init__()
		self.hidden_layer_1 = nn.Linear(inx*iny, 256)
		self.hidden_layer_2 = nn.Linear(256, 128)
		self.output_layer = nn.Linear(64, env.action_space.n)
		
		self.dropout_layer = nn.Dropout(0.2)
		
	def forward(self, x):
		x = self.hidden_layer_1(x)
		x = F.relu(x)
		x = self.dropout_layer(0.2)
		x = self.hidden_layer_2(x)
		x = F.relu(x)
		x = self.output_layer(x)
		return x

class DQNAgent:
    def __init__(self):
        # gets trained every step
        self.policy_model = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
		self.target_net.load_state_dict(policy_model.state_dict())
		self.target_net.eval() #Retire la propa des gradients.
		
		self.optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_states = current_states.reshape(-1,inx*iny)
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = new_current_states.reshape(-1, inx * iny)
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []

        for index, (state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(state)
            y.append(current_qs)
		X = torch.from_numpy(np.array(X)).float()
		y = torch.from_numpy(np.array(y)).long()
		
		predicted_value = self.policy_model(X)
		loss = mse_loss(predicted_value, y)
		self.optimizer.zero_grad()
		self.loss.backward()
		
		for param in self.policy_model.parameters():
			param.grad.data.clamp_(-1, 1)
			
		self.optimizer.step()
        
		if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_net.load_state_dict(self.policy_model.state_dict())
            self.target_update_counter = 0


agent = DQNAgent()
for episode in tqdm.tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    current_state = cv2.resize(current_state, (inx, iny))
    current_state = cv2.cvtColor(current_state, cv2.COLOR_BGR2GRAY)
    current_state = np.reshape(current_state, (inx*iny, -1))
    done = False
    ep_rewards = []
    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, info = env.step(action)
        new_state = cv2.resize(new_state, (inx, iny))
        new_state = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
        new_state = np.reshape(new_state, (inx*iny, -1))
        episode_reward += reward
        env.render()
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    print(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        # Save model, but only when min reward is greater or equal a set value
        #         if average_reward >= MIN_REWARD:
        #             agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}')
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


from env import TradingEnv
from memory import Transition, ReplayMemory
from model import DQN, QuantileLoss
from utilis import save_data_structure, plot_loss_portfolio, logarithmic_epsilon_decay
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class DQN_agent:
	def __init__(self, 
				T=96, 
				gamma=0.99, 
				episodes=1000,
				batch_size=64, 
				memory_train_capacity=10000, 
				memory_val_capacity=10000,
				patience=30,
				soft_update_interval=5,
				epochs = 1000,
				early_saturation_percentage=1,
				lr=0.00025,
				is_tunning=False):
		
		self.train_env = TradingEnv(T=T)
		self.val_env = TradingEnv(T=T)
		
		self.state_size = self.train_env.data.state_size + 3
		self.action_size = 3
		self.memory_train = ReplayMemory(memory_train_capacity)
		self.memory_val = ReplayMemory(memory_val_capacity)
		self.inventory = []
		self.T = T
		self.gamma = gamma
		self.max_epsilon = 1.0
		self.min_epsilon_min = 0.01
		self.epsilon_iterator = logarithmic_epsilon_decay(episodes, self.max_epsilon , self.min_epsilon_min)
		self.epsilon = next(self.epsilon_iterator)
		self.batch_size = batch_size
		self.patience = patience
		self.soft_update_interval = soft_update_interval
		self.epochs = epochs
		self.episodes = episodes
		self.prev_weights = None
		self.early_saturation_percentage = early_saturation_percentage
		self.quantiles = [0.95, 0.5, 0.05]
		self.lr = lr
		self.is_tunning = is_tunning

		if os.path.exists('models/target_model'):
			self.policy_net = torch.load('models/policy_model', map_location=device)
			self.target_net = torch.load('models/target_model', map_location=device)
		else:
			self.policy_net = DQN(self.state_size).to(device)
			self.target_net = DQN(self.state_size).to(device)

			for param_p in self.policy_net.parameters(): 
				weight_init.normal_(param_p)

		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
		
	def act(self, state):
		# Exploration
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size) - 1
		# Exploitation
		else:
			tensor = torch.FloatTensor(state).to(device)
			tensor = tensor.unsqueeze(0)
			options = self.target_net(tensor)
			return (np.argmax(options[-1].detach().cpu().numpy()) - 1)

	def store_train(self, state, actions, new_states, rewards, action, step):
		if step < self.batch_size * 20: # make sure its bigger than batchsize. Else rewrite
			for n in range(len(actions)):
				self.memory_train.push(state, actions[n], new_states[n], rewards[n])
		else:
			for n in range(len(actions)):
				if actions[n] == action:
					self.memory_train.push(state, actions[n], new_states[n], rewards[n])
					break

	def store_val(self, state, actions, new_states, rewards, action, step):
		if step < self.batch_size * 20: # make sure its bigger than batchsize. Else rewrite
			for n in range(len(actions)):
				self.memory_val.push(state, actions[n], new_states[n], rewards[n])
		else:
			for n in range(len(actions)):
				if actions[n] == action:
					self.memory_val.push(state, actions[n], new_states[n], rewards[n])
					break
	
	def gather_samples(self):
		state_train = self.train_env.reset() 
		state_val = self.val_env.reset()
		self.total_reward = []
		for step in range(len(self.train_env.data)):
			try:
				action_train = self.act(state_train) 
				actions_train, rewards_train, new_states_train, state_train, done_train = self.train_env.step(action_train)
				self.store_train(state_train, actions_train, new_states_train, rewards_train, action_train, step)
				self.total_reward += rewards_train[-1]

				action_val = self.act(state_val) 
				actions_val, rewards_val, new_states_val, state_val, done_val = self.val_env.step(action_val)
				self.store_val(state_val, actions_val, new_states_val, rewards_val, action_val, step)

				if done_train and done_val:
					break
			except:
				break 

	
	def optimize(self, step):

		done = False

		self.gather_samples()
		training_losses = []
		validation_losses = []
		best_val_loss = float('inf')
		patience_counter = 0

		for epoch in range(1, self.epochs+1):
			# Training mode
			transitions = self.memory_train.sample(self.batch_size)
			batch = Transition(*zip(*transitions))
			next_state = torch.FloatTensor(batch.next_state).to(device)
			non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))

			state_batch = torch.FloatTensor(batch.state).to(device)
			action_batch = torch.LongTensor(torch.add(torch.tensor(batch.action), torch.tensor(1))).to(device)
			reward_batch = torch.FloatTensor(batch.reward).to(device)

			l = self.policy_net(state_batch).size(0)
			state_action_values = self.policy_net(state_batch)[self.T-1:l:self.T].gather(1, action_batch.reshape((self.batch_size, 1)))
			state_action_values = state_action_values.squeeze(-1)
			next_state_values = torch.zeros(self.batch_size, device=device)
			next_state_values[non_final_mask] = self.target_net(next_state)[self.T-1:l:self.T].max(1)[0].detach()
			# Compute the expected Q values
			expected_state_action_values = (next_state_values * self.gamma) + reward_batch

			loss_fn = QuantileLoss(self.quantiles)
			loss = loss_fn(expected_state_action_values, state_action_values)
			loss.backward()

			for param in self.policy_net.parameters():
					param.grad.data.clamp_(-1, 1)
			loss_numpy = loss.cpu().detach().numpy()
			training_losses.append(loss_numpy.item())
			self.optimizer.step()

			# Valuation mode
			self.policy_net.eval()
			self.target_net.eval()
			with torch.no_grad():
				transitions_val = self.memory_val.sample(self.batch_size)
				batch_val = Transition(*zip(*transitions_val))
				next_state_val = torch.FloatTensor(batch_val.next_state).to(device)
				non_final_mask_val = torch.tensor(tuple(map(lambda s: s is not None, next_state_val)))
				state_batch_val = torch.FloatTensor(batch_val.state).to(device)
				action_batch_val = torch.LongTensor(torch.add(torch.tensor(batch_val.action), torch.tensor(1))).to(device)
				reward_batch_val = torch.FloatTensor(batch_val.reward).to(device)
				l_val = self.policy_net(state_batch_val).size(0)
				state_action_values_val = self.policy_net(state_batch_val)[self.T-1:l:self.T].gather(1, action_batch_val.reshape((self.batch_size, 1)))
				state_action_values_val = state_action_values_val.squeeze(-1)
				next_state_values_val = torch.zeros(self.batch_size, device=device)
				next_state_values_val[non_final_mask_val] = self.target_net(next_state_val)[self.T-1:l_val:self.T].max(1)[0].detach()
				# Compute the expected Q values
				expected_state_action_values_val = (next_state_values_val * self.gamma) + reward_batch_val
				loss_val =loss_fn(expected_state_action_values_val, state_action_values_val)
			loss_numpy_val = loss_val.cpu().detach().numpy()
			validation_losses.append(loss_numpy_val.item())

			# Manual soft update of QNetwork for stabilization
			if step % self.soft_update_interval == 0:
				gamma = 0.001
				target_update = copy.deepcopy(self.target_net.state_dict())
				for k in target_update.keys():
					target_update[k] = self.target_net.state_dict()[k] * (1 - gamma) + self.policy_net.state_dict()[k] * gamma
				self.target_net.load_state_dict(target_update)
				
			self.policy_net.train()
			self.target_net.train()

			# Early stoppping
			if loss_numpy_val < best_val_loss:
				best_val_loss = loss_numpy_val
				patience_counter = 0
			else:
				patience_counter += 1
			if patience_counter >= self.patience:
				print(f'Early stopping at epoch {epoch + 1}')
				break
		# if step % 100 == 0 or step == 1000 or step == 1:
		# 	plot_loss(training_losses, validation_losses)
		if self.is_tunning==False:
			save_data_structure('results/training_loss.json', step, training_losses)
			save_data_structure('results/validation_loss.json', step, validation_losses)
			plot_loss_portfolio(training_losses, validation_losses, self.train_env.portfolio)
		self.epsilon = next(self.epsilon_iterator)

		return -np.mean(self.total_reward) 
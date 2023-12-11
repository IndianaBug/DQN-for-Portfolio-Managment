from run_agent import RunAgent
from agent import Agent
from trading_env import TradingEnv

a = RunAgent(TradingEnv(), TradingEnv(), Agent())
state_train = a.env_train.reset() # initial_state
state_val = a.env_val.reset() # initial_state

for step in range(len(a.env_train.data)):
    try:
        action_train = a.agent.act(state_train) # select greedy action, exploration is done in step-method
        actions_train, rewards_train, new_states_train, state_train, done_train = a.env_train.step(action_train)
        a.agent.store_train(state_train, actions_train, rewards_train, new_states_train, action_train, step)
    except:
        break
print(a.agent.memory_train)

# Epslon greedily

    # When ϵ=1ϵ=1, the agent always chooses a random action, which results in total exploration. This is useful when the agent is initially exploring the environment to gather information about the rewards associated with different actions.

    # When ϵ=0ϵ=0, the agent always chooses the action with the highest estimated value (exploits), resulting in total exploitation. This is useful when the agent has learned a relatively good policy and wants to focus on exploiting the known information to maximize rewards.


import random

# class EpsilonGreedyAgent:
#     def __init__(self, epsilon):
#         self.epsilon = epsilon

#     def choose_action(self, q_values):
#         if random.uniform(0, 1) < self.epsilon:
#             # Exploration: Choose a random action
#             return random.choice(range(len(q_values)))
#         else:
#             # Exploitation: Choose the action with the highest Q-value
#             return max(range(len(q_values)), key=lambda a: q_values[a])

# # Example usage
# epsilon_greedy_agent = EpsilonGreedyAgent(epsilon=0.1)
# q_values = [0.5, 0.8, 0.3, 0.2]

# # Action selection
# chosen_action = epsilon_greedy_agent.choose_action(q_values)
# print(f"Chosen Action: {chosen_action}")

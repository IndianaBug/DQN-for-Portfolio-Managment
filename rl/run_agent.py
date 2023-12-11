from agent import Agent
from trading_env import TradingEnv

class RunAgent:
    def __init__(self, env_train=TradingEnv(),  env_val=TradingEnv(), agent=Agent()):
        self.env_train = env_train
        self.env_val = env_val
        self.agent = agent

    def run(self, episodes, args):
        # self.agent.initialize()
        state_train = self.env_train.reset() # initial_state
        state_val = self.env_val.reset()

        for step in range(episodes):
            # Training set
            action_train = self.agent.act(state_train) # select greedy action, exploration is done in step-method
            actions_train, rewards_train, new_states_train, state_train, done_train = self.env_train.step(action_train, step)
            self.agent.store_train(state_train, actions_train, rewards_train, new_states_train, action_train, step)
            # Val set
            action_val = self.agent.act(state_val)
            actions_val, rewards_val, new_states_val, state_val, done_val = self.env_val.step(action_val, step)
            self.agent.store_train(state_val, actions_val, rewards_val, new_states_val, action_val, step)
            # Stop 
            if done_train and done_val:
                break

            self.agent.optimize(step)

        self.env_train.print_stats(args)
        self.env_val.print_stats(args)
#this defines the trading MDP
from data_loader import Data
import numpy as np
from copy import deepcopy
from RL.utilis import save_data_structure
import json, codecs
import numpy as np
                  

def hot_encoding(a):
    a_ = np.zeros(3, dtype=np.float32)
    a_[a + 1] = 1.
    return a_

class TradingEnv:
    def __init__(self, initial_value=10000, T=96):
        self.initial_value = initial_value
        self.portfolio = [float(initial_value)]
        self.actions = []
        self.return_ = None
        self.data = Data(T=T)
        self.spread = 0.005
        self.commision = 0.001
        self.margin_maintanance = 0.0001
        self.trade_size = initial_value
        self.previous_position = "None"

    def merge_state_action(self, state, a_variable):
        """
            state - new state
            action_for_a_state - actions taken in a given state
            a_variable - If "a_variable" is used to represent a "policy parameter," it would refer to a value that determines the probability of the agent taking a certain action in a given state. This parameter is part of the agent's policy, which is a function that maps states to actions.
        """
        T = self.data.T     
        step_ = self.data.n 
        actions_for_state = self.actions[step_-1:][:T]

        diff = T - len(actions_for_state) -1
        if diff > 0:
            actions_for_state.extend([0] * diff) # If we will analyze over the period of previous 10 steps, then, at the beggining, previous actions should be 0 as we didnt do anything

        actions_for_state.append(a_variable)  # A variable is the next taken action which weather -1, 0, 1

        result = []
        for s, a in zip(state, actions_for_state):
            new_s = deepcopy(list(s))
            new_s.extend(hot_encoding(a))
            result.append(new_s)
        result = np.asarray(result)
        return result


    def reset(self):
        self.portfolio = [float(self.initial_value)]
        self.trade_size = self.initial_value
        self.data.reset()
        self.actions.append(0)
        return_, state_initial = self.data.next()
        self.data.n -= 1
        self.return_ = return_
        return self.merge_state_action(state_initial, 0)

    # Returns: actions, rewards, new_states, selected new_state, done
    def step(self, action):

        # Position handling
        last_action = self.actions[-1]
        position = self.previous_position
        if position == "None":
            if last_action == 1:
                self.previous_position = 'Long'
            if last_action == 0:
                self.previous_position = 'Short'
        if position == 'Short':
            if last_action == 1:
                self.previous_position = 'None'
            if last_action in [0, -1] :
                self.previous_position = 'Short'        
        if position == 'Long':
            if last_action == -1:
                self.previous_position = 'None'
            if last_action in [0, 1] :
                self.previous_position = 'Long'  

        # Data handling
        actions_ = [-1, 0, 1]
        v_old = self.portfolio[-1]
        try:
            return_, state_next = self.data.next()
            done = False
        except:
            state_next = None
            done = True
        new_states = []
        for a in actions_:
            new_states.append(self.merge_state_action(state_next, a))

        # Portfolio handling
        trade_cost = (self.trade_size * 2) * self.spread * self.commision 
        maintanance_cost = self.margin_maintanance * self.trade_size
        v_new = []
        for a in actions_:
            r = return_*self.trade_size * a
            # For hold
            if a == 0 and self.previous_position == "None": 
                v_new.append(v_old)
            if a == 0 and (self.previous_position == 'Long' or self.previous_position == 'Short'): 
                v_new.append(v_old - maintanance_cost + r) 
            # For short
            if a == -1 and self.previous_position == 'Short':
                v_new.append(v_old - maintanance_cost + r)
            if a == -1 and self.previous_position == 'None':
                v_new.append(v_old - maintanance_cost - trade_cost + r) 
            if a == -1 and self.previous_position == 'Long':
                v_new.append(v_old - maintanance_cost - trade_cost) # No reward scince we are closing the position
            # For long
            if a == 1 and self.previous_position == 'Short':
                v_new.append(v_old - maintanance_cost - trade_cost) 
            if a == 1 and self.previous_position == 'None':
                v_new.append(v_old - maintanance_cost + r) # No returns scince the previous position was None. Wait for the next trading day
            if a == 1 and self.previous_position == 'Long':
                v_new.append(v_old - maintanance_cost + r)
        v_new = np.asarray(v_new)


        rewards = []
        for i in range(len(v_new)):
            if v_new[i] * v_old > 0  and v_old != 0:          
                rewards.append(np.log(v_new[i] / v_old)) # Rewards is an array of loged percentage return
            else:
                rewards.append(-1)    # In case we lost all of our money
        rewards = np.asarray(rewards) 

        self.actions.append(int(action))   # New action
        self.portfolio.append(float(v_new[action+1])) # New value of the portfolio. Since -1 at position 0, 0 at position 1 and 1 at position 2 of the list

        return actions_, rewards, new_states, new_states[action+1], done
    
    def print_stats(self, episode):
        save_data_structure(f"results/action.json", episode, self.actions)
        save_data_structure(f"results/portfolio.json", episode, self.portfolio)





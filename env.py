# https://github.com/bassemfg/ddpg-rl-portfolio-management/tree/master

import yfinance as yf
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import gym


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough =1 #returns[returns.argmax():].min()
    return (trough - peak) / peak

def sharpe(returns, risk_free_rate=0):
    """
    Calculates the Sharpe ratio, a measure of risk-adjusted return.
    Parameters:
        returns (np.ndarray): A NumPy array of portfolio returns.
        risk_free_rate (float): The risk-free rate of return.
    Returns:
        float: The Sharpe ratio.
    """

    excess_returns = returns - risk_free_rate
    average_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)
    sharpe_ratio = average_excess_return / std_dev_excess_return
    return sharpe_ratio


df = pd.DataFrame(yf.Ticker("KO").history(start='2018-01-01', end='2023-09-30', interval='1d'))
# Add 10, 20, 50, 100 moving averages, both simple and exponential
for length in [10, 20, 50, 100]:
    df[f'ma_{length}'] = df['Close'].rolling(length).mean()
    df[f'ema_{length}'] = df['Close'].ewm(span = length, adjust = False ).mean()
df.insert(0, 'return', df['Close'].pct_change())
# You need to shift all columns one day otherwise the algo has the exact numbers of happenings
df_shifted = df.drop('return', axis=1)
df_shifted = df_shifted.shift(-1)
df_shifted.insert(0, 'return', df['return'])
df_shifted = df_shifted[:-1] # Remove the last line as it is empty
df = df_shifted.tail(1195)
# Rename all columns to lowercase
df.columns = df.columns.str.lower()
df = df.drop(['dividends', 'stock splits'], axis=1)



class DataGenerator(object):
    """Acts as data provider for each new episode."""
    def __init__(self, history=df, steps=1144, window_length=50, start_idx=0, normalize='minmaxStand'):
        """
        Args:
            history: open, high, low, close, volume and other features. All features must be lowercased. 'returns' must be the first column
            steps: cumulative number of decisions to make in an episode. Max len(steps) = len(history) - window_length
            window_length: observation window, must be less than 50
            normalize: normalize the data. Options: minmax, minmaxStand
        """
        import copy
        self.step = 0
        self.steps = steps
        assert self.steps < len(history) - 50
        self.window_length = window_length
        self.start_idx = start_idx
        self.normalize = normalize

        # make immutable class that will be reused after using reset
        self._data = history.copy()  # all data
        # Get rid of NaNs
        self._data.replace(np.nan, 0, inplace=True)
        self._data = self._data.ffill()
        # Normalize the data
        if self.normalize == 'minmax':
            rew =  self._data['return']
            self._data = self._data[[x for x in self._data.columns if x != 'return']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            self._data.insert(0, 'reward', rew)
        if self.normalize == 'minmaxStand':
            rew =  self._data['return']
            self._data = self._data[[x for x in self._data.columns if x != 'return']].apply(lambda x: (x - x.mean()) / x.std()).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            self._data.insert(0, 'reward', rew)
        # Get observation matrix
        self.data = self._data.values
        self.features = self._data.columns
        # Index to start in case if the number of actions in each episode is not equal to the number of observations
        self.idx = np.random.randint(low = self.window_length, high = self._data.shape[0] - self.steps)

    def _step(self):
        # get observation matrix from history
        self.step += 1
        obs = self.data[self.step:self.step + self.window_length, :].copy()
        # normalize obs with open price
        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        # Resets the dataset
        self.step = 0
        self.idx = np.random.randint(low = self.window_length, high = self._data.shape[0] - self.steps)
        self.data = self._data.values
        return self.data[self.step:self.step + self.window_length, :].copy(), self.data[self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        


class PortfolioSim(object):
    """
    Portfolio management simmulator.
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, initial_portfolio_value=10000, steps=730, trading_cost=0.01/100, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.inp = initial_portfolio_value
        self.p0 = initial_portfolio_value
        self.infos = []

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action -1 short 0 hold 1 long and everything in between
        y1 - return
        Numbered equations are from https://arxiv.org/abs/1706.10059 -- may be used to develop a multi portfolio simulator
        """
        # Calculate the amount of fees to change the portfolio
        fee = self.cost * abs(w1) *  self.p0  
        # Calculate the return after a single action 
        return_ = (self.p0 - fee) * y1 * w1
        # Add the cost of holding
        return_ = return_ * (1 - self.time_cost)
        # final portfolio value
        p1 =  self.p0 + return_
        # rate of returns
        rho1 = p1 / self.p0 - 1  
        # log rate of return
        r1 = np.log((p1) / (self.p0))  
        reward = r1 / self.steps * 1000  # (22) average logarithmic accumulated return
        # remember for next step
        self.p0 = p1 
        # if we run out of money, we're done 
        done = return_ == 0
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": self.p0 ,
            "return": y1,
            "rate_of_return": rho1,
            "weights": w1,
            "cost": fee,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = self.inp



class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products, but, in this case, into a single financial product.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """
    metadata = {'render.modes': ['human', 'ansi']} 
    def __init__(self,
                 history=df,
                 steps=1144,
                 window_length = 50,
                 normalize='minmaxStand',
                 trading_cost=0.01/100,
                 time_cost=0.00,
                 initial_portfolio_value=10000,
                 start_idx=0
                 ):

        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """

        self.window_length = window_length

        self.src = DataGenerator(history=history, 
                                 steps=steps, 
                                 window_length=window_length,
                                 normalize=normalize,
                                 start_idx=start_idx)

        self.sim = PortfolioSim(initial_portfolio_value=initial_portfolio_value,
                                trading_cost=trading_cost,
                                time_cost=time_cost,
                                steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from -1 to 1 as you can also short
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, ), dtype=np.float32)

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, 
                                                high=np.inf, 
                                                shape=(window_length, 
                                                       history.shape[-1]), 
                                                dtype=np.float32)
        self.infos = []

        
    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        - Where wn is a portfolio continuous weight from -1 to 1.
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        # normalise just in case
        action = np.clip(action, -1, 1)
        observation, done1, ground_truth_obs = self.src._step()
        # The last observation day 
        y1 = observation[-1, 0]
        reward, info, done2 = self.sim._step(action, y1)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        self.infos.append(info)
        # Do not put rewards into that crap
        return observation[:, 1:], reward, done1 or done2, info
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            return self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return, risk_free_rate=0)
        title = f'max_drawdown={mdd} sharpe_ratio={sharpe_ratio}'    
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)


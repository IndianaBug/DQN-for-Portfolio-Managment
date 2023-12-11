from collections import deque
import pandas as pd
import numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from RL.utilis import get_dataset


class Data():
    def __init__(self, T=96):
        self.state_space = None
        self.T = T
        self.n = 0
        self.state_size = 0
        self.load()


    def load(self):
        # Load, normalize and scale data the data
        df = get_dataset()
        non_feature_data = df.iloc[:, 0]
        features = df.iloc[:, 1:]
        standard_scaler = StandardScaler()
        standardized_features = standard_scaler.fit_transform(features)
        min_max_scaler = MinMaxScaler()
        normalized_features = min_max_scaler.fit_transform(standardized_features)
        self.returns = non_feature_data.values
        self.state_space = normalized_features
        self.state_size = features.shape[1]
        self.it = self.iterator()

    def next(self):
        self.n += 1
        return next(self.it)

    def iterator(self):
        d = deque()
        for v in zip(self.returns, self.state_space):
            closing = v[0]
            features_total = v[1]
            d.append(features_total)
            while len(d) > self.T:
                d.popleft()
            if len(d) == self.T:
                yield closing, list(d)

    def reset(self):
        self.it = self.iterator()
        self.n = 0
    
    def __len__(self):
        return len(self.state_space)
    



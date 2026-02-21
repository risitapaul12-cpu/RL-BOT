import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):

    def __init__(self, price_data):
        super().__init__()

        self.price_data = price_data
        self.num_assets = 15
        self.features_per_asset = 15
        self.portfolio_features = 4

        self.state_dim = (self.num_assets * self.features_per_asset) + self.portfolio_features

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.initial_cash = 1_000_000
        self.max_position_size = 0.1
        self.reset()

    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets)
        self.entry_prices = np.zeros(self.num_assets)
        self.days_held = np.zeros(self.num_assets)
        self.peak_value = self.initial_cash

        return self._get_state(), {}

    
    def step(self, action):

        action = np.clip(action, -1, 1)

        current_prices = self.price_data[self.current_step]

        target_positions = action * self.max_position_size

        portfolio_value_before = self._calculate_portfolio_value(current_prices)

        self.positions = target_positions

        portfolio_value_after = self._calculate_portfolio_value(current_prices)

        reward = portfolio_value_after - portfolio_value_before

        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1

        return self._get_state(), reward, done, False, {}

    
    def _calculate_portfolio_value(self, prices):
        position_value = np.sum(self.positions * prices)
        total_value = self.cash + position_value
        self.peak_value = max(self.peak_value, total_value)
        return total_value

    
    def _get_state(self):

        asset_features = np.zeros(self.num_assets * self.features_per_asset)

        cash_ratio = self.cash / self.initial_cash
        drawdown = (self.peak_value - self.initial_cash) / self.peak_value
        portfolio_vol = 0
        gross_exposure = np.sum(np.abs(self.positions))

        portfolio_features = np.array([
            cash_ratio,
            drawdown,
            portfolio_vol,
            gross_exposure
        ])

        state = np.concatenate([asset_features, portfolio_features])

        assert len(state) == self.state_dim

        return state.astype(np.float32)
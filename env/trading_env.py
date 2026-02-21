import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):

    def __init__(self, price_data):
        super().__init__()

        self.price_data = price_data
        self.num_assets = price_data.shape[1]
        self.features_per_asset = 15
        self.portfolio_features = 4

        self.state_dim = (self.num_assets * self.features_per_asset) + self.portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.initial_cash = 1_000_000
        self.max_position_pct = 0.1  # 10% max allocation per asset
        self.transaction_cost_pct = 0.001  # 0.1% per trade

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets)  # number of shares
        self.entry_prices = np.zeros(self.num_assets)
        self.days_held = np.zeros(self.num_assets)
        self.peak_value = self.initial_cash

        return self._get_state(), {}

    def step(self, action):

        action = np.clip(action, -1, 1)

        prices_t = self.price_data[self.current_step]
        portfolio_value_before = self._calculate_portfolio_value(prices_t)

        # Convert action into target portfolio allocation
        target_allocations = action * self.max_position_pct
        target_values = target_allocations * portfolio_value_before

        current_values = self.positions * prices_t
        trade_values = target_values - current_values

        # Execute trades
        for i in range(self.num_assets):

            trade_value = trade_values[i]
            price = prices_t[i]

            if trade_value > 0:
                # BUY
                cost = trade_value * (1 + self.transaction_cost_pct)
                if self.cash >= cost:
                    shares_to_buy = trade_value / price
                    self.positions[i] += shares_to_buy
                    self.cash -= cost
                    self.entry_prices[i] = price
                    self.days_held[i] = 0

            elif trade_value < 0:
                # SELL
                shares_to_sell = min(abs(trade_value) / price, self.positions[i])
                proceeds = shares_to_sell * price
                proceeds_after_cost = proceeds * (1 - self.transaction_cost_pct)

                self.positions[i] -= shares_to_sell
                self.cash += proceeds_after_cost

                if self.positions[i] == 0:
                    self.entry_prices[i] = 0
                    self.days_held[i] = 0

        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1

        prices_next = self.price_data[self.current_step]
        portfolio_value_after = self._calculate_portfolio_value(prices_next)

        reward = portfolio_value_after - portfolio_value_before

        self.peak_value = max(self.peak_value, portfolio_value_after)

        return self._get_state(), reward, done, False, {}

    def _calculate_portfolio_value(self, prices):
        position_value = np.sum(self.positions * prices)
        return self.cash + position_value

    def _get_state(self):

        # Placeholder features (replace later with real features)
        asset_features = np.zeros(self.num_assets * self.features_per_asset)

        current_prices = self.price_data[self.current_step]
        portfolio_value = self._calculate_portfolio_value(current_prices)

        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        portfolio_vol = 0  # can implement later
        gross_exposure = np.sum(np.abs(self.positions * current_prices)) / portfolio_value

        portfolio_features = np.array([
            cash_ratio,
            drawdown,
            portfolio_vol,
            gross_exposure
        ])

        state = np.concatenate([asset_features, portfolio_features])

        assert len(state) == self.state_dim

        return state.astype(np.float32)
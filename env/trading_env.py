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
        self.max_position_pct = 0.1

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

        prices_t = self.price_data[self.current_step]
        portfolio_value_before = self._calculate_portfolio_value(prices_t)

        target_allocations = action * self.max_position_pct
        target_values = target_allocations * portfolio_value_before

        current_values = self.positions * prices_t
        trade_values = target_values - current_values

        for i in range(self.num_assets):

            trade_value = trade_values[i]
            price = prices_t[i]

            if trade_value > 0:
                buy_value = trade_value
                total_cost = self._calculate_buy_cost(buy_value)

                if self.cash >= total_cost:
                    shares_to_buy = buy_value / price
                    self.positions[i] += shares_to_buy
                    self.cash -= total_cost
                    self.entry_prices[i] = price
                    self.days_held[i] = 0

            elif trade_value < 0:
                sell_value = min(abs(trade_value), self.positions[i] * price)

                shares_to_sell = sell_value / price
                proceeds = self._calculate_sell_proceeds(sell_value)

                self.positions[i] -= shares_to_sell
                self.cash += proceeds

                if self.positions[i] <= 1e-8:
                    self.positions[i] = 0
                    self.entry_prices[i] = 0
                    self.days_held[i] = 0

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

    def _calculate_buy_cost(self, buy_value):
        stt = 0.001 * buy_value
        txn_charges = 0.0000297 * buy_value
        gst = 0.18 * txn_charges
        stamp = 0.00015 * buy_value
        return buy_value + stt + txn_charges + gst + stamp

    def _calculate_sell_proceeds(self, sell_value):
        stt = 0.001 * sell_value
        txn_charges = 0.0000297 * sell_value
        gst = 0.18 * txn_charges
        return sell_value - stt - txn_charges - gst

    def _get_state(self):

        asset_features = np.zeros(self.num_assets * self.features_per_asset)

        current_prices = self.price_data[self.current_step]
        portfolio_value = self._calculate_portfolio_value(current_prices)

        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        portfolio_vol = 0
        gross_exposure = np.sum(np.abs(self.positions * current_prices)) / portfolio_value

        portfolio_features = np.array([
            cash_ratio,
            drawdown,
            portfolio_vol,
            gross_exposure
        ])

        state = np.concatenate([asset_features, portfolio_features])

        return state.astype(np.float32)
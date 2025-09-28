import sqlite3
import numpy as np
import random
from datetime import datetime, timedelta

DB_PATH = "your_database.db"  # Set your SQLite path

# Battery constraints
BATTERY_CAPACITY = 30_000  # Wh
BATTERY_MIN_SOC = 0.10
CHARGE_POWER = 4_000  # W  <-- Corrected charging power!
INTERVALS = 6
INTERVAL_LENGTH = 24 * 60 // INTERVALS  # minutes per interval
SOC_TARGETS = [10, 20, 40, 60, 80, 100]  # Percent targets

def load_prices(conn):
    cur = conn.cursor()
    cur.execute("SELECT from_time, market_price_tax FROM electricity_prices ORDER BY from_time")
    prices = {row[0]: row[1] for row in cur.fetchall()}
    return prices

def load_solar_forecast(conn):
    cur = conn.cursor()
    cur.execute("SELECT watt_hours_period, watt_hours_value FROM solar_forecast_hourly")
    forecast = {row[0]: row[1] for row in cur.fetchall()}
    return forecast

def load_consumption(conn):
    cur = conn.cursor()
    cur.execute("SELECT from_time, usageTotal FROM electricity_usage_household_hourly")
    consumption = {row[0]: row[1] for row in cur.fetchall()}
    return consumption

def get_next_day_hours():
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return [(base + timedelta(minutes=i*INTERVAL_LENGTH)).strftime("%H%M")
            for i in range(INTERVALS)]

def make_intervals():
    times = get_next_day_hours()
    return times

# Simple Q-learning agent
class BatterySchedulerRL:
    def __init__(self, prices, solar, consumption):
        self.prices = prices
        self.solar = solar
        self.consumption = consumption
        self.q_table = dict()
        self.epsilon = 0.3
        self.alpha = 0.1
        self.gamma = 0.9

    def get_state(self, t, soc):
        price = self.prices.get(t, 0)
        solar = self.solar.get(t, 0)
        cons = self.consumption.get(t, 0)
        return (t, int(soc), int(price), int(solar), int(cons))

    def choose_action(self, state):
        actions = []
        for target_soc in SOC_TARGETS:
            actions.append((1, target_soc))
        actions.append((0, int(state[1])))
        if random.random() < self.epsilon:
            return random.choice(actions)
        key = (state[0], state[1])
        if key in self.q_table:
            return max(self.q_table[key], key=lambda x: self.q_table[key][x])
        else:
            return random.choice(actions)

    def update_q(self, state, action, reward, next_state):
        key = (state[0], state[1])
        if key not in self.q_table:
            self.q_table[key] = dict()
        self.q_table[key][action] = self.q_table[key].get(action, 0) + self.alpha * (
            reward + self.gamma * self.max_q(next_state) - self.q_table[key].get(action, 0)
        )

    def max_q(self, state):
        key = (state[0], state[1])
        if key not in self.q_table or not self.q_table[key]:
            return 0
        return max(self.q_table[key].values())

    def train(self, episodes=100):
        for ep in range(episodes):
            soc = BATTERY_MIN_SOC * 100  # Start at min SOC
            times = make_intervals()
            for t in times:
                state = self.get_state(t, soc)
                action = self.choose_action(state)
                enabled, target_soc = action
                price = self.prices.get(t, 0)
                solar = self.solar.get(t, 0)
                cons = self.consumption.get(t, 0)
                grid_power = max(0, cons - solar)
                charge_amount = (target_soc - soc) * BATTERY_CAPACITY / 100 if enabled else 0
                cost = charge_amount / 1000 * price if enabled else 0
                reward = -cost
                if target_soc > 100:
                    reward -= 100
                soc = min(target_soc, 100)
                next_state = self.get_state(t, soc)
                self.update_q(state, action, reward, next_state)

    def schedule_next_day(self):
        soc = BATTERY_MIN_SOC * 100
        times = make_intervals()
        payload = {"time": {}, "power": {}, "voltage": {}, "soc": {}, "enabled": {}}
        for i, t in enumerate(times, 1):
            state = self.get_state(t, soc)
            action = self.choose_action(state)
            enabled, target_soc = action
            payload["time"][str(i)] = t
            payload["power"][str(i)] = CHARGE_POWER
            payload["voltage"][str(i)] = 49
            payload["soc"][str(i)] = target_soc
            payload["enabled"][str(i)] = enabled
            soc = target_soc
        return payload

def main():
    conn = sqlite3.connect(DB_PATH)
    prices = load_prices(conn)
    solar = load_solar_forecast(conn)
    consumption = load_consumption(conn)

    agent = BatterySchedulerRL(prices, solar, consumption)
    agent.train(episodes=200)
    schedule = agent.schedule_next_day()
    print(schedule)

if __name__ == "__main__":
    main()
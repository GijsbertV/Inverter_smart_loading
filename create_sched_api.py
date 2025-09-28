import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, request, jsonify

DB_PATH = "/app/energy_2.db"

# Battery constraints
BATTERY_CAPACITY = 30_000  # Wh
BATTERY_MIN_SOC = 10       # percent
BATTERY_MAX_SOC = 100      # percent

POWER = 6000               # W, inverter/discharge (payload output only)
CHARGE_POWER = 4000        # W, actual charging speed (used for charging logic)
INTERVALS = 6

app = Flask(__name__)

def load_prices(conn):
    cur = conn.cursor()
    cur.execute("SELECT from_time, market_price FROM electricity_prices ORDER BY from_time")
    prices = {}
    for row in cur.fetchall():
        utc_dt = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%fZ")
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
        cet_dt = utc_dt.astimezone(ZoneInfo("Europe/Amsterdam"))
        cet_str = cet_dt.strftime("%Y-%m-%d %H:%M:%S")
        prices[cet_str] = row[1]
    return prices

def load_consumption(conn):
    cur = conn.cursor()
    cur.execute("SELECT from_time, usageTotal FROM electricity_usage_household_hourly")
    consumption = {}
    for row in cur.fetchall():
        utc_dt = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%fZ")
        utc_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC"))
        cet_dt = utc_dt.astimezone(ZoneInfo("Europe/Amsterdam"))
        cet_str = cet_dt.strftime("%Y-%m-%d %H:%M:%S")
        consumption[cet_str] = row[1]
    return consumption

def load_solar_forecast(conn):
    cur = conn.cursor()
    cur.execute("SELECT watt_hours_period, watt_hours_value FROM solar_forecast_hourly")
    forecast = {}
    for row in cur.fetchall():
        cet_dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        aligned_dt = cet_dt + timedelta(hours=-1)
        aligned_str = aligned_dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast[aligned_str] = row[1]
    return forecast

def build_hourly_times(forecast_date: str) -> list:
    times = [f"{forecast_date} {hour:02d}:00:00" for hour in range(24)]
    dt_next_day = datetime.strptime(forecast_date, "%Y-%m-%d") + timedelta(days=1)
    times.append(f"{dt_next_day.strftime('%Y-%m-%d')} 00:00:00")
    return times

def get_prices_for_times(prices: dict, times: list) -> list:
    return [prices.get(t, 0) for t in times]

import numpy as np

def find_two_largest_cheap_blocks(prices: list, n_blocks: int = 6, min_charging_block_hours: int = 4) -> list:
    """
    Find two non-overlapping blocks with the lowest average prices,
    each at least min_charging_block_hours long, and make sure the remaining
    hours can be split into exactly (n_blocks - 2) intervals, each at least 1 hour.
    """
    block_candidates = []
    min_block_size = min_charging_block_hours

    # The largest allowed sum of charging block sizes:
    max_total_charging = 24 - (n_blocks - 2)

    for size1 in range(min_block_size, max_total_charging + 1):
        for start1 in range(0, 24 - size1 + 1):
            end1 = start1 + size1
            avg1 = np.mean(prices[start1:end1])

            for size2 in range(min_block_size, max_total_charging + 1):
                for start2 in range(0, 24 - size2 + 1):
                    end2 = start2 + size2

                    # Check for non-overlap
                    if end2 <= start1 or start2 >= end1:
                        # Check total charging block size
                        total_charging = size1 + size2
                        if total_charging > max_total_charging:
                            continue

                        avg2 = np.mean(prices[start2:end2])

                        # Find gaps between charging blocks
                        gaps = []
                        # Before first block
                        if start1 > 0:
                            gaps.append((0, start1))
                        # Between blocks
                        if end1 < start2:
                            gaps.append((end1, start2))
                        elif end2 < start1:
                            gaps.append((end2, start1))
                        # After second block
                        if end2 < 24:
                            gaps.append((end2, 24))

                        # Count total non-charging hours
                        total_gap_hours = sum(gap[1] - gap[0] for gap in gaps)
                        # Must be at least enough for n_blocks-2 intervals, each at least 1 hour
                        if total_gap_hours < (n_blocks - 2):
                            continue

                        # Can we split the gaps into exactly n_blocks-2 intervals, each at least 1 hour?
                        # Flatten gaps into list of hours
                        flat_hours = []
                        for rng in gaps:
                            flat_hours += list(range(rng[0], rng[1]))
                        if len(flat_hours) < (n_blocks - 2):
                            continue  # not enough hours

                        # Score: prefer larger blocks and lower prices
                        score = (total_charging, -(avg1 + avg2) / 2)
                        block_candidates.append((score, (start1, end1), (start2, end2)))

    if not block_candidates:
        # fallback: just use two blocks respecting the minimum size
        return [(0, min(24, min_charging_block_hours)), (min(24, min_charging_block_hours), 24)]

    # Sort by score: largest blocks, lowest average prices first
    block_candidates.sort(reverse=True)
    best = block_candidates[0]
    return sorted([best[1], best[2]], key=lambda x: x[0])

def build_intervals_with_min_charging_blocks(blocks, n_blocks=6, min_charging_block_hours=4):
    # blocks: list of (start, end) for two charging blocks
    # n_blocks: total intervals (e.g. 6)
    # min_charging_block_hours: minimum length of charging blocks (in hours)
    blocks = sorted(blocks, key=lambda x: x[0])
    charging_intervals = [blocks[0], blocks[1]]

    # Find non-charging ranges
    non_charging_ranges = []
    if charging_intervals[0][0] > 0:
        non_charging_ranges.append((0, charging_intervals[0][0]))
    if charging_intervals[0][1] < charging_intervals[1][0]:
        non_charging_ranges.append((charging_intervals[0][1], charging_intervals[1][0]))
    if charging_intervals[1][1] < 24:
        non_charging_ranges.append((charging_intervals[1][1], 24))

    # Number of non-charging intervals to allocate
    n_non_charging = n_blocks - 2
    # Get total length of non-charging hours
    total_non_charging = sum(r[1] - r[0] for r in non_charging_ranges)
    other_intervals = []

    # If there are non-charging hours and intervals to fill
    if n_non_charging > 0 and total_non_charging > 0:
        # Split as evenly as possible
        interval_size = total_non_charging / n_non_charging
        # Flatten non-charging ranges into one timeline
        ptr = 0
        intervals_added = 0
        flat_hours = []
        for rng in non_charging_ranges:
            flat_hours += list(range(rng[0], rng[1]))
        while intervals_added < n_non_charging:
            start_idx = int(round(intervals_added * interval_size))
            end_idx = int(round((intervals_added + 1) * interval_size))
            if end_idx > len(flat_hours):
                end_idx = len(flat_hours)
            if start_idx < end_idx:
                start = flat_hours[start_idx]
                end = flat_hours[end_idx - 1] + 1 if end_idx - 1 < len(flat_hours) else flat_hours[-1] + 1
                other_intervals.append((start, end))
            intervals_added += 1

    # Combine into total intervals
    all_intervals = []
    all_intervals.extend(other_intervals)
    all_intervals.append(charging_intervals[0])
    all_intervals.append(charging_intervals[1])
    all_intervals = sorted(all_intervals, key=lambda x: x[0])
    # Ensure intervals are contiguous and fill all 24 hours
    cleaned_intervals = []
    prev_end = 0
    for start, end in all_intervals:
        if start > prev_end:
            cleaned_intervals.append((prev_end, start))
        cleaned_intervals.append((start, end))
        prev_end = end
    # If last interval does not reach 24, fill to 24
    if cleaned_intervals[-1][1] < 24:
        cleaned_intervals.append((cleaned_intervals[-1][1], 24))
    # Remove intervals of length 0
    cleaned_intervals = [iv for iv in cleaned_intervals if iv[1] > iv[0]]
    # If too many, merge smallest; if too few, split largest
    while len(cleaned_intervals) > n_blocks:
        # Merge smallest
        min_len = min([x[1] - x[0] for x in cleaned_intervals])
        for i in range(len(cleaned_intervals) - 1):
            if cleaned_intervals[i][1] - cleaned_intervals[i][0] == min_len:
                cleaned_intervals[i] = (cleaned_intervals[i][0], cleaned_intervals[i + 1][1])
                cleaned_intervals.pop(i + 1)
                break
    while len(cleaned_intervals) < n_blocks:
        # Split largest
        max_len = max([x[1] - x[0] for x in cleaned_intervals])
        for i in range(len(cleaned_intervals)):
            if cleaned_intervals[i][1] - cleaned_intervals[i][0] == max_len:
                start = cleaned_intervals[i][0]
                end = cleaned_intervals[i][1]
                mid = start + (end - start) // 2
                cleaned_intervals[i] = (start, mid)
                cleaned_intervals.insert(i + 1, (mid, end))
                break
    return cleaned_intervals[:n_blocks]

def convert_intervals_to_times(intervals: list, times: list) -> list:
    result = []
    for start, end in intervals:
        start_time = times[start]
        end_time = times[end] if end < len(times) else times[-1]
        result.append((start_time, end_time))
    return result

def get_smart_intervals(prices, tomorrow, n_blocks=6, min_charging_block_hours=4):
    hourly_times = build_hourly_times(tomorrow)
    hourly_prices = get_prices_for_times(prices, hourly_times)
    blocks = find_two_largest_cheap_blocks(hourly_prices, n_blocks=n_blocks, min_charging_block_hours=min_charging_block_hours)
    intervals = build_intervals_with_min_charging_blocks(blocks, n_blocks=n_blocks)
    interval_times = convert_intervals_to_times(intervals, hourly_times)
    interval_starts = [tup[0] for tup in interval_times]
    return interval_starts, interval_times

def hhmm_from_datetime_string(dt_str):
    return dt_str[11:13] + dt_str[14:16]

def forecast_next_day_consumption(consumption, forecast_date, N_days=14):
    forecast = {}
    dt_forecast = datetime.strptime(forecast_date, "%Y-%m-%d")
    for hour in range(24):
        vals = []
        for days_back in range(1, N_days+1):
            dt_prev = dt_forecast - timedelta(days=days_back)
            t_prev = "%s %02d:00:00" % (dt_prev.strftime("%Y-%m-%d"), hour)
            val = consumption.get(t_prev, None)
            if val is not None:
                vals.append(val)
        avg_val = np.mean(vals) if vals else 0
        t_next = "%s %02d:00:00" % (forecast_date, hour)
        forecast[t_next] = avg_val
    return forecast

def lookahead_schedule(
    prices, solar, consumption, interval_starts, interval_times,
    initial_soc, margin_factor=1.10, diagnostics_enabled=True, price_percentile=75
):
    # ...[existing code above unchanged]...
    diagnostics = {}
    intervals = interval_times
    payload = {
        "time": {},
        "power": {},
        "voltage": {},
        "soc": {},
        "enabled": {},
    }
    price_per_interval = []
    net_demand_per_interval = []
    interval_durations = []
    prices_in_interval = []

    for start, end in intervals:
        interval_times_list = [t for t in prices.keys() if start <= t < end]
        hourly_prices = [prices[t] for t in interval_times_list]
        avg_price = np.mean(hourly_prices) if hourly_prices else 0
        price_per_interval.append(avg_price)
        total_consumption = np.sum([consumption.get(t,0) for t in interval_times_list])
        total_solar = np.sum([solar.get(t,0) for t in interval_times_list])
        net_demand = max(0, total_consumption - total_solar)
        net_demand_per_interval.append(net_demand)
        dt_start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        dt_end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        interval_durations.append((dt_end - dt_start).total_seconds() / 3600)
        prices_in_interval.append({t: float(prices[t]) for t in interval_times_list})

    # --- UPDATED LINE: use the parameter!
    price_threshold = np.percentile(price_per_interval, price_percentile) if price_per_interval else 0
    expensive_intervals = [
        i for i, p in enumerate(price_per_interval)
        if p >= price_threshold and net_demand_per_interval[i] > 0
    ]
    total_battery_wh_needed = sum([net_demand_per_interval[i] for i in expensive_intervals]) * margin_factor

    if diagnostics_enabled:
        diagnostics["interval_starts"] = interval_starts
        diagnostics["intervals"] = intervals
        diagnostics["price_per_interval"] = [float(x) for x in price_per_interval]
        diagnostics["prices_in_interval"] = prices_in_interval
        diagnostics["net_demand_per_interval"] = [float(x) for x in net_demand_per_interval]
        diagnostics["interval_durations"] = interval_durations
        diagnostics["price_threshold"] = float(price_threshold)
        diagnostics["expensive_intervals"] = expensive_intervals
        diagnostics["total_battery_wh_needed"] = float(total_battery_wh_needed)
        diagnostics["initial_soc"] = float(initial_soc)
        diagnostics["margin_factor"] = float(margin_factor)
        diagnostics["price_percentile"] = float(price_percentile)

    soc = initial_soc
    battery_wh = BATTERY_CAPACITY * soc / 100

    for i in range(len(intervals)):
        idx = str(i+1)
        payload["time"][idx] = hhmm_from_datetime_string(intervals[i][0])
        payload["power"][idx] = POWER
        payload["voltage"][idx] = 49

        interval_hours = interval_durations[i]
        charge_wh_possible = CHARGE_POWER * interval_hours

        if i not in expensive_intervals and soc < 100 and total_battery_wh_needed > 0:
            charge_wh = min(charge_wh_possible, total_battery_wh_needed, BATTERY_CAPACITY * (100-soc)/100)
            soc += charge_wh * 100 / BATTERY_CAPACITY
            soc = min(soc, 100)
            payload["enabled"][idx] = 1
            payload["soc"][idx] = int(soc)
            total_battery_wh_needed -= charge_wh
            action = f"Charge {charge_wh:.2f} Wh"
        elif i in expensive_intervals:
            discharge_wh = min(net_demand_per_interval[i], BATTERY_CAPACITY * soc / 100)
            soc -= discharge_wh * 100 / BATTERY_CAPACITY
            soc = max(soc, BATTERY_MIN_SOC)
            payload["enabled"][idx] = 0
            payload["soc"][idx] = int(soc)
            action = f"Discharge {discharge_wh:.2f} Wh"
        else:
            payload["enabled"][idx] = 0
            payload["soc"][idx] = int(soc)
            action = "Idle"

        if diagnostics_enabled:
            diagnostics[f"interval_{idx}"] = {
                "action": action,
                "soc_after": soc
            }

    if diagnostics_enabled:
        payload["diagnostics"] = diagnostics

    return payload

@app.route('/schedule', methods=['GET'])
def get_schedule():
    try:
        current_soc = float(request.args.get("current_soc", BATTERY_MIN_SOC))
        margin_factor = float(request.args.get("margin_factor", 1.10))
        n_days = int(request.args.get("n_days", 14))
        min_charging_block_hours = int(request.args.get("min_charging_block_hours", 4))
        price_percentile = float(request.args.get("price_percentile", 75))  # <-- add this line
        conn = sqlite3.connect(DB_PATH)
        prices = load_prices(conn)
        raw_consumption = load_consumption(conn)
        solar = load_solar_forecast(conn)
        conn.close()

        tomorrow = (datetime.now(ZoneInfo("Europe/Amsterdam")) + timedelta(days=1)).strftime("%Y-%m-%d")
        consumption_forecast = forecast_next_day_consumption(raw_consumption, tomorrow, N_days=n_days)
        interval_starts, interval_times = get_smart_intervals(
            prices, tomorrow, n_blocks=INTERVALS, min_charging_block_hours=min_charging_block_hours)
        schedule = lookahead_schedule(
            prices, solar, consumption_forecast, interval_starts, interval_times,
            current_soc, margin_factor, diagnostics_enabled=True, price_percentile=price_percentile
        )
        return jsonify(schedule)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/consumption_forecast', methods=['GET'])
def get_consumption_forecast():
    try:
        n_days = int(request.args.get("n_days", 14))
        min_charging_block_hours = int(request.args.get("min_charging_block_hours", 4))
        conn = sqlite3.connect(DB_PATH)
        raw_consumption = load_consumption(conn)
        solar_forecast = load_solar_forecast(conn)
        conn.close()
        tomorrow = (datetime.now(ZoneInfo("Europe/Amsterdam")) + timedelta(days=1)).strftime("%Y-%m-%d")
        consumption_forecast = forecast_next_day_consumption(raw_consumption, tomorrow, N_days=n_days)

        forecast_output = {}
        for hour in range(24):
            t_next = "%s %02d:00:00" % (tomorrow, hour)
            consumption = consumption_forecast.get(t_next, 0)
            solar = solar_forecast.get(t_next, 0)
            net_need = max(0, consumption - solar)
            forecast_output[t_next] = {
                "forecasted_consumption": consumption,
                "solar_forecast": solar,
                "net_need": net_need
            }
        return jsonify({
            "forecast_date": tomorrow,
            "n_days": n_days,
            "min_charging_block_hours": min_charging_block_hours,
            "forecast": forecast_output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
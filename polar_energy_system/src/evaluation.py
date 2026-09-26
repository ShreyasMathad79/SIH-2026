"""
Evaluation & Benchmark Engine
Calculates 24/48/72-Hour Energy Survival Forecast, Risk Alerts, and
Baseline vs AI-Optimized performance comparison metrics.
"""

import numpy as np
import pandas as pd

def calculate_survival_forecast(
    optimization_df,
    current_fuel_liters=5000.0,
    battery_capacity_kwh=200.0,
    min_soc=20.0
):
    """
    Evaluates 24-72h energy survival metrics and generates dynamic operator alerts.
    """
    T = len(optimization_df)
    alerts = []
    
    # 1. Deficit & Load Shedding Detection
    total_shed = optimization_df['total_shed_kw'].sum()
    crit_shed = optimization_df['shed_critical_kw'].sum()
    
    if crit_shed > 0.1:
        first_crit_h = optimization_df[optimization_df['shed_critical_kw'] > 0.1].index[0] + 1
        alerts.append({
            'type': 'CRITICAL',
            'icon': '🚨',
            'title': 'Critical Energy Deficit Imminent',
            'message': f"Critical load shortage predicted in {first_crit_h} hours. Emergency reserves insufficient!"
        })
    elif total_shed > 0.1:
        first_shed_h = optimization_df[optimization_df['total_shed_kw'] > 0.1].index[0] + 1
        alerts.append({
            'type': 'WARNING',
            'icon': '⚠',
            'title': 'Non-Critical Load Deferral Required',
            'message': f"Energy deficit predicted in {first_shed_h} hours. Non-critical loads will be deferred."
        })
        
    # 2. Battery Depletion Trend
    soc_series = optimization_df['battery_soc'].values
    min_soc_reached = np.min(soc_series)
    if min_soc_reached <= min_soc + 2.0:
        low_soc_h = np.where(soc_series <= min_soc + 2.0)[0][0] + 1
        alerts.append({
            'type': 'WARNING',
            'icon': '🔋',
            'title': 'Rapid Battery Depletion',
            'message': f"Battery SOC projected to hit minimum reserve ({round(min_soc_reached,1)}%) within {low_soc_h} hours."
        })
        
    # 3. Renewable Availability Trend
    sol_sum = optimization_df['solar_available_kw'].sum()
    wind_sum = optimization_df['wind_available_kw'].sum()
    ren_total = sol_sum + wind_sum
    load_total = optimization_df['demand_kw'].sum()
    
    ren_share = (ren_total / max(1e-5, load_total)) * 100
    if ren_share < 25.0:
        alerts.append({
            'type': 'INFO',
            'icon': '❄',
            'title': 'Low Renewable Generation Period',
            'message': f"Renewable contribution expected to drop to {round(ren_share,1)}% over the next {T} hours."
        })
        
    # 4. Diesel Fuel Endurance
    diesel_consumed = optimization_df['diesel_fuel_liters'].sum()
    rem_fuel = current_fuel_liters - diesel_consumed
    
    avg_diesel_rate = optimization_df['diesel_fuel_liters'].mean()
    if avg_diesel_rate > 0.1:
        hours_fuel_left = rem_fuel / avg_diesel_rate
    else:
        hours_fuel_left = 999.0
        
    if rem_fuel < 500.0:
        alerts.append({
            'type': 'CRITICAL',
            'icon': '⛽',
            'title': 'Low Diesel Fuel Warning',
            'message': f"Remaining fuel stock ({round(rem_fuel,1)} L) critically low! Estimated endurance: {round(hours_fuel_left,1)} hours."
        })
        
    if len(alerts) == 0:
        alerts.append({
            'type': 'SUCCESS',
            'icon': '✓',
            'title': 'Energy Reserves Healthy',
            'message': f"Station energy supply stable for the next {T} hours. Renewable share: {round(ren_share,1)}%."
        })
        
    survival_metrics = {
        'horizon_hours': T,
        'renewable_share_pct': round(ren_share, 1),
        'total_diesel_consumed_l': round(diesel_consumed, 1),
        'min_battery_soc_pct': round(min_soc_reached, 1),
        'total_unmet_demand_kwh': round(total_shed, 1),
        'critical_loads_protected_pct': round(100.0 - (crit_shed / max(1e-5, load_total * 0.5)) * 100, 1),
        'estimated_fuel_endurance_hours': round(hours_fuel_left, 1),
        'alerts': alerts
    }
    
    return survival_metrics

def compare_baseline_vs_optimized(
    load_forecast_kw,
    solar_forecast_kw,
    wind_forecast_kw,
    optimized_df,
    initial_soc=75.0,
    fuel_cost_per_liter=2.50
):
    """
    Simulates baseline (naive diesel-first / threshold rule strategy) vs AI-Optimized strategy.
    Returns comparison dictionary with metrics and savings %.
    """
    T = len(load_forecast_kw)
    
    # 1. Baseline Simulation (Diesel-first rule-based without lookahead optimization)
    b_soc = initial_soc
    b_diesel_fuel = 0.0
    b_diesel_kw = 0.0
    b_ren_used = 0.0
    b_ren_curtailed = 0.0
    b_deficits = 0.0
    
    for t in range(T):
        l = load_forecast_kw[t]
        s = solar_forecast_kw[t]
        w = wind_forecast_kw[t]
        ren = s + w
        
        if ren >= l: # Direct renewable supply
            b_ren_used += l
            b_ren_curtailed += (ren - l)
        else: # Deficit -> turn on diesel generator immediately to supply gap
            gap = l - ren
            b_ren_used += ren
            # Diesel runs at gap or min load 20 kW
            d_out = max(20.0, min(100.0, gap))
            fuel = 3.5 + 0.28 * d_out
            b_diesel_fuel += fuel
            b_diesel_kw += d_out
            
            if d_out < gap:
                b_deficits += (gap - d_out)
                
    # 2. Optimized Metrics from PuLP dispatch
    opt_diesel_fuel = optimized_df['diesel_fuel_liters'].sum()
    opt_solar_used = optimized_df['solar_used_kw'].sum()
    opt_wind_used = optimized_df['wind_used_kw'].sum()
    opt_ren_used = opt_solar_used + opt_wind_used
    
    total_solar_avail = optimized_df['solar_available_kw'].sum()
    total_wind_avail = optimized_df['wind_available_kw'].sum()
    opt_ren_curtailed = (total_solar_avail + total_wind_avail) - opt_ren_used
    opt_deficits = optimized_df['total_shed_kw'].sum()
    
    # Calculations
    fuel_saved_l = max(0.0, b_diesel_fuel - opt_diesel_fuel)
    fuel_saved_pct = (fuel_saved_l / max(1e-5, b_diesel_fuel)) * 100.0
    cost_saved_usd = fuel_saved_l * fuel_cost_per_liter
    
    # CO2 emission reduction: ~ 2.68 kg CO2 per liter of diesel fuel burned
    co2_saved_kg = fuel_saved_l * 2.68
    
    total_load = sum(load_forecast_kw)
    baseline_ren_util = (b_ren_used / max(1e-5, total_solar_avail + total_wind_avail)) * 100.0
    optimized_ren_util = (opt_ren_used / max(1e-5, total_solar_avail + total_wind_avail)) * 100.0
    
    return {
        'baseline': {
            'diesel_fuel_liters': round(b_diesel_fuel, 1),
            'fuel_cost_usd': round(b_diesel_fuel * fuel_cost_per_liter, 2),
            'renewable_utilization_pct': round(min(100.0, baseline_ren_util), 1),
            'renewable_curtailed_kwh': round(b_ren_curtailed, 1),
            'energy_deficit_kwh': round(b_deficits, 1),
            'co2_emissions_kg': round(b_diesel_fuel * 2.68, 1)
        },
        'optimized': {
            'diesel_fuel_liters': round(opt_diesel_fuel, 1),
            'fuel_cost_usd': round(opt_diesel_fuel * fuel_cost_per_liter, 2),
            'renewable_utilization_pct': round(min(100.0, optimized_ren_util), 1),
            'renewable_curtailed_kwh': round(max(0.0, opt_ren_curtailed), 1),
            'energy_deficit_kwh': round(opt_deficits, 1),
            'co2_emissions_kg': round(opt_diesel_fuel * 2.68, 1)
        },
        'savings': {
            'fuel_saved_liters': round(fuel_saved_l, 1),
            'fuel_saved_pct': round(fuel_saved_pct, 1),
            'cost_saved_usd': round(cost_saved_usd, 2),
            'co2_saved_kg': round(co2_saved_kg, 1)
        }
    }

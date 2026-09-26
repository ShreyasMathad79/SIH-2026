"""
Energy Optimization Engine
Uses Mixed-Integer Linear Programming (MILP via PuLP) to optimize energy dispatch across Solar, Wind, Battery, Diesel, and Tiered Load Shedding.
"""

import pulp
import numpy as np
import pandas as pd

class EnergyOptimizer:
    def __init__(
        self,
        battery_capacity_kwh=200.0,
        battery_min_soc=20.0,
        battery_max_soc=95.0,
        battery_max_chg_kw=40.0,
        battery_max_dis_kw=50.0,
        battery_efficiency=0.92,
        diesel_max_kw=100.0,
        diesel_min_kw=15.0,
        fuel_cost_per_liter=2.50,
        battery_deg_cost=0.08,
        curtailment_penalty=0.01,
        critical_penalty=10000.0,
        important_penalty=500.0,
        noncrit_penalty=50.0
    ):
        self.b_cap = float(battery_capacity_kwh)
        self.min_soc = float(battery_min_soc)
        self.max_soc = float(battery_max_soc)
        self.max_chg = float(battery_max_chg_kw)
        self.max_dis = float(battery_max_dis_kw)
        self.eta = float(battery_efficiency)
        
        self.diesel_max = float(diesel_max_kw)
        self.diesel_min = float(diesel_min_kw)
        self.fuel_cost = float(fuel_cost_per_liter)
        
        self.c_deg = float(battery_deg_cost)
        self.c_curt = float(curtailment_penalty)
        self.p_crit = float(critical_penalty)
        self.p_imp = float(important_penalty)
        self.p_noncrit = float(noncrit_penalty)

    def optimize_dispatch(
        self,
        load_forecast_kw,
        solar_forecast_kw,
        wind_forecast_kw,
        initial_soc=75.0,
        timestamps=None
    ):
        """
        Solves the MILP energy dispatch problem for T hours horizon.
        """
        T = len(load_forecast_kw)
        if timestamps is None:
            timestamps = [f"t+{i}h" for i in range(T)]
            
        prob = pulp.LpProblem("PolarStationEnergyOptimization", pulp.LpMinimize)
        
        # Decision Variables
        S_used = [pulp.LpVariable(f"S_used_{t}", lowBound=0, upBound=solar_forecast_kw[t]) for t in range(T)]
        W_used = [pulp.LpVariable(f"W_used_{t}", lowBound=0, upBound=wind_forecast_kw[t]) for t in range(T)]
        
        P_chg = [pulp.LpVariable(f"P_chg_{t}", lowBound=0, upBound=self.max_chg) for t in range(T)]
        P_dis = [pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=self.max_dis) for t in range(T)]
        
        D_gen = [pulp.LpVariable(f"D_gen_{t}", lowBound=0, upBound=self.diesel_max) for t in range(T)]
        u_diesel = [pulp.LpVariable(f"u_diesel_{t}", cat=pulp.LpBinary) for t in range(T)]
        
        SoC = [pulp.LpVariable(f"SoC_{t}", lowBound=self.min_soc, upBound=self.max_soc) for t in range(T)]
        
        # Load shedding by priority tier
        # Total load = 50% Critical, 30% Important, 20% Non-Critical
        L_crit = [0.50 * load_forecast_kw[t] for t in range(T)]
        L_imp = [0.30 * load_forecast_kw[t] for t in range(T)]
        L_noncrit = [0.20 * load_forecast_kw[t] for t in range(T)]
        
        Shed_crit = [pulp.LpVariable(f"Shed_crit_{t}", lowBound=0, upBound=L_crit[t]) for t in range(T)]
        Shed_imp = [pulp.LpVariable(f"Shed_imp_{t}", lowBound=0, upBound=L_imp[t]) for t in range(T)]
        Shed_noncrit = [pulp.LpVariable(f"Shed_noncrit_{t}", lowBound=0, upBound=L_noncrit[t]) for t in range(T)]
        
        # Objective Function
        objective_terms = []
        for t in range(T):
            # Fuel cost: ~ (3.0 u_diesel + 0.25 D_gen) * fuel_cost
            fuel_cost_t = (3.0 * u_diesel[t] + 0.25 * D_gen[t]) * self.fuel_cost
            deg_cost_t = (P_chg[t] + P_dis[t]) * self.c_deg
            curt_cost_t = ((solar_forecast_kw[t] - S_used[t]) + (wind_forecast_kw[t] - W_used[t])) * self.c_curt
            shed_cost_t = (Shed_crit[t] * self.p_crit) + (Shed_imp[t] * self.p_imp) + (Shed_noncrit[t] * self.p_noncrit)
            
            objective_terms.extend([fuel_cost_t, deg_cost_t, curt_cost_t, shed_cost_t])
            
        prob += pulp.lpSum(objective_terms)
        
        # Constraints
        for t in range(T):
            # 1. Power Balance: Solar + Wind + Discharge + Diesel - Charge + Shedding == Total Load
            total_supply = S_used[t] + W_used[t] + P_dis[t] + D_gen[t] - P_chg[t] + Shed_crit[t] + Shed_imp[t] + Shed_noncrit[t]
            prob += (total_supply == load_forecast_kw[t]), f"PowerBalance_{t}"
            
            # 2. Diesel limits when ON
            prob += (D_gen[t] >= self.diesel_min * u_diesel[t]), f"DieselMin_{t}"
            prob += (D_gen[t] <= self.diesel_max * u_diesel[t]), f"DieselMax_{t}"
            
            # 3. Dynamic SoC update
            soc_change = ((P_chg[t] * self.eta - P_dis[t] / self.eta) / self.b_cap) * 100.0
            if t == 0:
                prob += (SoC[t] == initial_soc + soc_change), f"SoC_Update_{t}"
            else:
                prob += (SoC[t] == SoC[t-1] + soc_change), f"SoC_Update_{t}"

        # Solve MILP
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
        status = prob.solve(solver)
        
        if pulp.LpStatus[status] in ['Optimal', 'Not Solved']:
            # Extract results
            results = []
            for t in range(T):
                s_u = max(0.0, float(pulp.value(S_used[t]) or 0))
                w_u = max(0.0, float(pulp.value(W_used[t]) or 0))
                p_c = max(0.0, float(pulp.value(P_chg[t]) or 0))
                p_d = max(0.0, float(pulp.value(P_dis[t]) or 0))
                d_g = max(0.0, float(pulp.value(D_gen[t]) or 0))
                soc_val = max(self.min_soc, min(self.max_soc, float(pulp.value(SoC[t]) or initial_soc)))
                
                sh_c = max(0.0, float(pulp.value(Shed_crit[t]) or 0))
                sh_i = max(0.0, float(pulp.value(Shed_imp[t]) or 0))
                sh_nc = max(0.0, float(pulp.value(Shed_noncrit[t]) or 0))
                
                fuel_used = (3.0 * (1 if d_g > 0.1 else 0) + 0.25 * d_g) if d_g > 0.1 else 0.0
                
                # Action text description
                if d_g > 0.5:
                    action = f"Diesel Support ({round(d_g,1)} kW)"
                elif p_d > 0.5:
                    action = f"Battery Discharge ({round(p_d,1)} kW)"
                elif p_c > 0.5:
                    action = f"Battery Charge ({round(p_c,1)} kW)"
                elif (solar_forecast_kw[t] + wind_forecast_kw[t]) >= load_forecast_kw[t]:
                    action = "100% Renewable Direct Supply"
                else:
                    action = "Load Management Active"
                    
                results.append({
                    'timestamp': timestamps[t],
                    'demand_kw': round(load_forecast_kw[t], 2),
                    'solar_available_kw': round(solar_forecast_kw[t], 2),
                    'wind_available_kw': round(wind_forecast_kw[t], 2),
                    'solar_used_kw': round(s_u, 2),
                    'wind_used_kw': round(w_u, 2),
                    'battery_charge_kw': round(p_c, 2),
                    'battery_discharge_kw': round(p_d, 2),
                    'battery_soc': round(soc_val, 2),
                    'diesel_output_kw': round(d_g, 2),
                    'diesel_fuel_liters': round(fuel_used, 2),
                    'shed_critical_kw': round(sh_c, 2),
                    'shed_important_kw': round(sh_i, 2),
                    'shed_noncritical_kw': round(sh_nc, 2),
                    'total_shed_kw': round(sh_c + sh_i + sh_nc, 2),
                    'action': action
                })
            return pd.DataFrame(results)
        else:
            # Fallback heuristic rule-based optimizer if solver returns infeasible
            return self._heuristic_fallback(load_forecast_kw, solar_forecast_kw, wind_forecast_kw, initial_soc, timestamps)

    def _heuristic_fallback(self, load_kw, solar_kw, wind_kw, initial_soc, timestamps):
        """Rule-based heuristic optimizer fallback."""
        T = len(load_kw)
        soc = initial_soc
        results = []
        
        for t in range(T):
            l = load_kw[t]
            s = solar_kw[t]
            w = wind_kw[t]
            ren = s + w
            net = l - ren
            
            p_c = 0.0
            p_d = 0.0
            d_g = 0.0
            sh_c = 0.0
            sh_i = 0.0
            sh_nc = 0.0
            
            if net <= 0: # Renewable surplus
                surplus = -net
                headroom_kwh = (self.max_soc - soc) / 100.0 * self.b_cap
                p_c = min(surplus, self.max_chg, headroom_kwh / self.eta)
                soc += (p_c * self.eta / self.b_cap) * 100.0
                action = "Renewable Surplus -> Battery Charge"
            else: # Renewable deficit
                avail_kwh = (soc - self.min_soc) / 100.0 * self.b_cap
                p_d = min(net, self.max_dis, avail_kwh * self.eta)
                soc -= (p_d / self.eta / self.b_cap) * 100.0
                rem_def = net - p_d
                
                if rem_def > 0:
                    d_g = max(self.diesel_min, min(self.diesel_max, rem_def))
                    rem_def2 = rem_def - d_g
                    if rem_def2 > 0:
                        # Shed non-critical first, then important, then critical
                        noncrit_max = 0.20 * l
                        sh_nc = min(rem_def2, noncrit_max)
                        rem_def3 = rem_def2 - sh_nc
                        if rem_def3 > 0:
                            imp_max = 0.30 * l
                            sh_i = min(rem_def3, imp_max)
                            sh_c = rem_def3 - sh_i
                    action = "Diesel Generator Support"
                else:
                    action = "Battery Discharge Support"
                    
            soc = np.clip(soc, self.min_soc, self.max_soc)
            fuel_used = (3.0 + 0.25 * d_g) if d_g > 0.1 else 0.0
            
            results.append({
                'timestamp': timestamps[t],
                'demand_kw': round(l, 2),
                'solar_available_kw': round(s, 2),
                'wind_available_kw': round(w, 2),
                'solar_used_kw': round(s, 2),
                'wind_used_kw': round(w, 2),
                'battery_charge_kw': round(p_c, 2),
                'battery_discharge_kw': round(p_d, 2),
                'battery_soc': round(soc, 2),
                'diesel_output_kw': round(d_g, 2),
                'diesel_fuel_liters': round(fuel_used, 2),
                'shed_critical_kw': round(sh_c, 2),
                'shed_important_kw': round(sh_i, 2),
                'shed_noncritical_kw': round(sh_nc, 2),
                'total_shed_kw': round(sh_c + sh_i + sh_nc, 2),
                'action': action
            })
            
        return pd.DataFrame(results)

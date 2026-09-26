"""
Battery Energy Storage System (BESS) Model
Manages battery capacity, state of charge (SoC), charge/discharge rates, efficiency, and reserve constraints.
"""

class BatteryStorage:
    def __init__(
        self,
        capacity_kwh=200.0,
        current_soc=75.0,
        max_charge_power_kw=40.0,
        max_discharge_power_kw=50.0,
        charge_efficiency=0.92,
        discharge_efficiency=0.92,
        min_soc=20.0,
        max_soc=95.0,
        degradation_cost_per_kwh=0.08
    ):
        self.capacity_kwh = float(capacity_kwh)
        self.current_soc = float(current_soc)
        self.max_charge_power_kw = float(max_charge_power_kw)
        self.max_discharge_power_kw = float(max_discharge_power_kw)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.min_soc = float(min_soc)
        self.max_soc = float(max_soc)
        self.degradation_cost_per_kwh = float(degradation_cost_per_kwh)

    def available_energy_kwh(self):
        """Available dischargeable energy above min_soc reserve (kWh)."""
        usable_soc = max(0.0, self.current_soc - self.min_soc)
        return (usable_soc / 100.0) * self.capacity_kwh * self.discharge_efficiency

    def available_charge_headroom_kwh(self):
        """Headroom available for charging up to max_soc (kWh)."""
        headroom_soc = max(0.0, self.max_soc - self.current_soc)
        return (headroom_soc / 100.0) * self.capacity_kwh / self.charge_efficiency

    def update_soc(self, charge_power_kw=0.0, discharge_power_kw=0.0, duration_hours=1.0):
        """
        Updates battery state of charge based on power flow over duration_hours.
        charge_power_kw: power sent to battery (kW)
        discharge_power_kw: power pulled from battery (kW)
        """
        net_energy_stored = (charge_power_kw * self.charge_efficiency * duration_hours) - \
                            (discharge_power_kw / self.discharge_efficiency * duration_hours)
                            
        soc_change = (net_energy_stored / self.capacity_kwh) * 100.0
        self.current_soc = float(max(10.0, min(100.0, self.current_soc + soc_change)))
        return self.current_soc

    def get_state(self):
        return {
            'capacity_kwh': self.capacity_kwh,
            'current_soc': round(self.current_soc, 2),
            'min_soc': self.min_soc,
            'max_soc': self.max_soc,
            'max_charge_power_kw': self.max_charge_power_kw,
            'max_discharge_power_kw': self.max_discharge_power_kw,
            'available_energy_kwh': round(self.available_energy_kwh(), 2)
        }

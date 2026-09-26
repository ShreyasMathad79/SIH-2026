"""
Diesel Generator Model
Tracks generator capacity, fuel consumption rate, fuel tank reserves, and operating hours remaining.
"""

class DieselGenerator:
    def __init__(
        self,
        max_capacity_kw=100.0,
        min_generation_kw=15.0,
        specific_fuel_cons_l_kwh=0.28,
        current_fuel_liters=5000.0,
        fuel_tank_capacity_liters=10000.0,
        fuel_cost_per_liter=2.50,
        startup_cost=15.0
    ):
        self.max_capacity_kw = float(max_capacity_kw)
        self.min_generation_kw = float(min_generation_kw)
        self.specific_fuel_cons_l_kwh = float(specific_fuel_cons_l_kwh)
        self.current_fuel_liters = float(current_fuel_liters)
        self.fuel_tank_capacity_liters = float(fuel_tank_capacity_liters)
        self.fuel_cost_per_liter = float(fuel_cost_per_liter)
        self.startup_cost = float(startup_cost)

    def calculate_fuel_consumption(self, power_kw, duration_hours=1.0):
        """
        Calculates fuel consumed in Liters for given power generated over duration_hours.
        Includes base idling fuel consumption + load-proportional fuel.
        """
        if power_kw <= 0.1:
            return 0.0
        actual_kw = max(self.min_generation_kw, min(self.max_capacity_kw, power_kw))
        # Fuel curve: base idling ~3 L/h + 0.25 L/kWh
        fuel_liters = (3.0 + 0.25 * actual_kw) * duration_hours
        return float(fuel_liters)

    def estimate_remaining_hours(self, avg_power_kw=40.0):
        """Estimates remaining generator run hours based on current fuel level and average demand."""
        fuel_per_hour = self.calculate_fuel_consumption(avg_power_kw, 1.0)
        if fuel_per_hour <= 0:
            return float('inf')
        return float(self.current_fuel_liters / fuel_per_hour)

    def consume_fuel(self, power_kw, duration_hours=1.0):
        """Deducts fuel consumed from tank level and returns liters consumed."""
        fuel_used = self.calculate_fuel_consumption(power_kw, duration_hours)
        self.current_fuel_liters = max(0.0, self.current_fuel_liters - fuel_used)
        return float(fuel_used)

    def get_state(self):
        return {
            'max_capacity_kw': self.max_capacity_kw,
            'min_generation_kw': self.min_generation_kw,
            'current_fuel_liters': round(self.current_fuel_liters, 1),
            'fuel_tank_capacity_liters': self.fuel_tank_capacity_liters,
            'fuel_percentage': round((self.current_fuel_liters / self.fuel_tank_capacity_liters) * 100, 1),
            'estimated_operating_hours': round(self.estimate_remaining_hours(), 1)
        }

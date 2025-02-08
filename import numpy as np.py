import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

class Appliance:
    def __init__(self, name, power, usage_hours):
        self.name = name
        self.power = power  # kW
        self.usage_hours = usage_hours  # List of hours it runs

class SmartEnergyOptimizer:
    def __init__(self):
        self.appliances = []
        self.energy_prices = np.random.uniform(0.1, 0.5, 24)  # Simulated energy price for each hour
    
    def add_appliance(self, appliance):
        self.appliances.append(appliance)
    
    def train_model(self):
        """Train a simple model to predict optimal energy usage times"""
        # Simulated past data: hours and corresponding costs
        past_hours = np.array([i for i in range(24)]).reshape(-1, 1)
        past_costs = self.energy_prices  # Assume past prices are same as today's
        
        model = LinearRegression()
        model.fit(past_hours, past_costs)
        self.model = model

    def predict_optimal_hours(self, usage_hours):
        """Predict best times for energy consumption"""
        predicted_costs = self.model.predict(np.array(usage_hours).reshape(-1, 1))
        sorted_hours = [hour for _, hour in sorted(zip(predicted_costs, usage_hours))]
        return sorted_hours[: len(usage_hours)]  # Best times to run appliances

    def optimize_energy(self):
        """Optimize the energy usage based on predicted prices"""
        self.train_model()
        schedule = {}
        total_consumption = 0

        for appliance in self.appliances:
            best_hours = self.predict_optimal_hours(appliance.usage_hours)
            for hour in best_hours:
                if hour not in schedule:
                    schedule[hour] = []
                schedule[hour].append(appliance.name)
                total_consumption += appliance.power
        
        return schedule, total_consumption

# Example usage
if __name__ == "__main__":
    optimizer = SmartEnergyOptimizer()
    
    # Define appliances and their default usage patterns
    washing_machine = Appliance("Washing Machine", 0.5, [7, 19])
    fridge = Appliance("Fridge", 0.1, list(range(24)))  # Runs 24/7
    ac = Appliance("Air Conditioner", 1.5, [14, 15, 16, 17])

    # Add appliances to the optimizer
    optimizer.add_appliance(washing_machine)
    optimizer.add_appliance(fridge)
    optimizer.add_appliance(ac)

    # Optimize energy usage
    schedule, total_consumption = optimizer.optimize_energy()

    # Print results
    print("\nOptimized Energy Usage Schedule:")
    for hour in sorted(schedule.keys()):
        print(f"{hour}:00 - {', '.join(schedule[hour])}")

    print(f"\nTotal Energy Consumption: {total_consumption:.2f} kW")

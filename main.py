import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

k = 1
epsilon = 1
E_min = -13*epsilon

table_1 = {
    "N": [20, 40, 80, 160],
    "ree_a_ratio_squared": [66.7, 193, 549, 1555]
}
table_2 = {
    "E/epsilon": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13],
    "gE": [18_671_059_783.5, 15_687_265_041, 5_351_538_782, 1_222_946_058,
           234_326_487, 40_339_545, 5_824_861, 710_407,
           77_535, 9_046, 645, 86, 0, 1]
}


class Exercise1:
    def __init__(self, data):
        self.total_steps = np.array(data["N"])
        self.ree_a_ratio_squared = np.array(data["ree_a_ratio_squared"])

    def plot(self):
        log_N = np.log(self.total_steps)
        log_ree_a_ratio_squared = np.log(self.ree_a_ratio_squared)
        v, _, _, _, _ = linregress(log_N, log_ree_a_ratio_squared)

        plt.figure(figsize=(10, 6))
        plt.plot(log_N, log_ree_a_ratio_squared, label='Self-Avoiding Walk')
        plt.plot(log_N, log_N, label='Ordinary Random Walk')
        plt.xlabel('$log(N)$', fontsize=14)
        plt.ylabel('$log(r_{ee}^2/a^2)$', fontsize=14)
        plt.title('Self-Avoiding Walk vs Ordinary Random Walk')
        plt.legend()
        plt.text(3, 4, rf'$ \nu \approx {v / 2:.2f}$', fontsize=12)
        plt.savefig('exercise_1.png')
        plt.show()


class Exercise2b:
    def __init__(self, data, k, epsilon):
        self.E_min = -13 * epsilon
        self.E_epsilon_ratio = np.array(data["E/epsilon"])
        self.density_of_states = np.array(data["gE"])
        self.temperature_values = np.linspace(0.1, 1, 100)
        self.k = k

    def average_energy(self, temperature):
        numerator = np.sum(self.E_epsilon_ratio * self.density_of_states * np.exp(
            -(self.E_epsilon_ratio - self.E_min) / (self.k * temperature)))
        denominator = np.sum(
            self.density_of_states * np.exp(-(self.E_epsilon_ratio - self.E_min) / (self.k * temperature)))
        return numerator / denominator

    def average_energy_squared(self, temperature):
        numerator = np.sum(self.E_epsilon_ratio ** 2 * self.density_of_states * np.exp(
            -(self.E_epsilon_ratio - self.E_min) / (self.k * temperature)))
        denominator = np.sum(
            self.density_of_states * np.exp(-(self.E_epsilon_ratio - self.E_min) / (self.k * temperature)))
        return numerator / denominator

    def heat_capacity(self, temperature):
        return (1 / (self.k * temperature ** 2)) * (
                    self.average_energy_squared(temperature) - self.average_energy(temperature) ** 2)

    def plot(self):
        cv_values = [self.heat_capacity(t) for t in self.temperature_values]
        t_max = self.temperature_values[np.argmax(cv_values)]

        plt.figure(figsize=(10, 6))
        plt.plot(self.temperature_values, cv_values)
        plt.xlabel('$T\ [\\epsilon/k]$', fontsize=14)
        plt.ylabel('$C_V\ [k]$', fontsize=14)
        plt.axvline(t_max, color='r', linestyle='--')
        plt.text(0.43, 6, f'$T = {t_max:.3f} \ \\epsilon/k$', fontsize=12)
        plt.title('Heat Capacity vs Temperature')
        plt.savefig('exercise_2b.png')
        plt.show()


class Exercise2c:
    def __init__(self, data, k, epsilon):
        self.E_min = -13 * epsilon
        self.E_epsilon_ratio = np.array(data["E/epsilon"])
        self.density_of_states = np.array(data["gE"])
        self.temperature_values = np.linspace(0.1, 1, 100)
        self.k = k

    def probability(self, t):
        numerator = np.exp(-self.E_min / (self.k * t))
        denominator = np.sum(self.density_of_states * np.exp(-self.E_epsilon_ratio / (self.k * t)))
        return numerator / denominator

    def plot(self):
        P_values = [self.probability(t) for t in self.temperature_values]

        plt.figure(figsize=(10, 6))
        plt.plot(self.temperature_values, P_values)
        plt.xlabel('$T\ [\\epsilon/k]$', fontsize=14)
        plt.ylabel('$P_{nat}$', fontsize=14)
        plt.title('Probability of finding the chain in its minimum-energy state vs Temperature')
        plt.savefig('exercise_2c.png')
        plt.show()


exercise1 = Exercise1(table_1)
exercise1.plot()

exercise2b = Exercise2b(table_2, k, epsilon)
exercise2b.plot()

exercise2c = Exercise2c(table_2, k, epsilon)
exercise2c.plot()

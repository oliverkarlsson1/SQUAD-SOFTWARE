import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    pass

# Exercise 1
table_1 = {
    "N": [20, 40, 80, 160],
    "ree_a_ratio_squared": [66.7, 193, 549, 1555]
}

total_steps = np.array(table_1["N"])
ree_a_ratio_squared = np.array(table_1["ree_a_ratio_squared"])

step_length = 1


def ree_squared(n, a):
    return n*a**2


log_N = np.log(np.array(table_1["N"]))
log_ree_a_ratio_squared = np.log(table_1["ree_a_ratio_squared"])
v, _, _, _, _, = linregress(log_N, log_ree_a_ratio_squared)

plt.figure(figsize=(10, 6))
plt.plot(np.log(total_steps),
         np.log(ree_a_ratio_squared),
         label='Self-Avoiding Walk')
plt.plot(np.log(total_steps),
         np.log(ree_squared(total_steps, step_length)/(step_length**2)),
         label='Ordinary Random Walk')
plt.xlabel('$log(N)$',
           fontsize=14)
plt.ylabel('$log(r_{ee}^2/a^2)$',
           fontsize=14)
plt.title('Self-Avoiding Walk vs Ordinary Random Walk ')
plt.legend()

plt.text(3, 4, rf'$ \nu \approx {v/2:.2f}$',
         fontsize=12)

plt.show()

# Exercise 2b
table_2 = {
    "E/epsilon": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13],
    "gE": [18_671_059_783.5, 15_687_265_041, 5_351_538_782, 1_222_946_058,
           234_326_487, 40_339_545, 5_824_861, 710_407,
           77_535, 9_046, 645, 86, 0, 1]
}


E_epsilon_ratio = np.array(table_2["E/epsilon"])
print(len(E_epsilon_ratio))
density_of_states = np.array(table_2["gE"])

temperature_values = np.linspace(0.1, 1, 100)

k = 1
epsilon = 1
E_min = -13*epsilon


def average_energy(k, temperature):
    numerator = np.sum(E_epsilon_ratio * density_of_states * np.exp(-(E_epsilon_ratio - E_min) / (k * temperature)))
    denominator = np.sum(density_of_states * np.exp(-(E_epsilon_ratio - E_min) / (k * temperature)))
    return numerator / denominator


def average_energy_squared(k, temperature):
    numerator = np.sum(E_epsilon_ratio ** 2 * density_of_states * np.exp(-(E_epsilon_ratio - E_min) / (k * temperature)))
    denominator = np.sum(density_of_states * np.exp(-(E_epsilon_ratio - E_min) / (k * temperature)))
    return numerator / denominator


def heat_capacity(k, temperature):
    return (1/(k*temperature**2)) * (average_energy_squared(k, temperature) - average_energy(k, temperature)**2)


CV_values = [heat_capacity(k, t) for t in temperature_values]

plt.figure(figsize=(10, 6))
plt.plot(temperature_values, CV_values)
plt.xlabel('$T\ [\\epsilon/k]$',
           fontsize=14)
plt.ylabel('$C_V\ [k]$',
           fontsize=14)

plt.title('Heat Capacity vs Temperature')

plt.show()

T_max = temperature_values[np.argmax(CV_values)]
print("Temperature at which CV is maximal:", T_max)

# Exercise 2c
def probability(k, t):
    numerator = np.exp(-E_min/(k*t))
    denominator = np.sum(density_of_states * np.exp(-E_epsilon_ratio/(k*t)))
    return numerator / denominator


P_values = [probability(k, t) for t in temperature_values]

plt.figure(figsize=(10, 6))
plt.plot(temperature_values, P_values)
plt.xlabel('$T\ [\\epsilon/k]$',
           fontsize=14)
plt.ylabel('$P_{nat}$',
           fontsize=14)
plt.title('Probability of finding the chain in its minimum-energy state vs Temperature')

plt.show()

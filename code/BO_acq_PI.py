#!/usr/bin/env python
# coding: utf-8

# In[3]:


from skopt import Optimizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt.space import Space
from skopt.sampler import Lhs
from skopt.sampler import Grid
from scipy.spatial.distance import pdist
from skopt import Optimizer
from skopt.plots import plot_objective
from skopt.plots import plot_evaluations
from skopt.space import Integer, Categorical
import warnings
import openpyxl

warnings.filterwarnings('ignore')

def get_result_from_excel(var1, var2, var3, var4, var5):

    workbook = openpyxl.load_workbook('.../Cancer.xlsx')
    sheet = workbook.active
    for row in sheet.iter_rows(min_row=2):
        if row[0].value == var1 and row[1].value == var2 and row[2].value == var3 and row[3].value == var4 and row[4].value == var5:
            result = -row[5].value
            return result
    return -664

num_cycles = 10
iterations_per_cycle = 50

max_values_over_cycles = []
f_values_over_cycles = []
iterations_over_cycles = []
input_variables_over_cycles = []
all_max_values = []
mean_values = []
std_values = []

for cycle in range(num_cycles):

    plt.figure(figsize=(10, 6))

    # Bayesian optimization
    opt = Optimizer([Integer(1, 3, name = 'texture'),
                     Integer(1, 3, name = 'smoothness'),  Integer(1, 3, name = 'concavity'),
                     Integer(1, 3, name = 'fractal'), Integer(1, 2, name = 'symmetry')], "GP", acq_func='PI', 
                    acq_optimizer="sampling", initial_point_generator="random",
                    n_initial_points=1, random_state=None)

    f_values = []
    max_values = []
    max_value_found = -np.inf  
    iteration_with_max_value = -1  
    input_variable_with_max_value = None  
    
    for i in range(iterations_per_cycle):

        next_point = opt.ask()      
        print(f"Next point at iteration {i + 1}: {next_point}")

        var1, var2, var3, var4, var5= next_point
        f_val = get_result_from_excel(var1, var2, var3, var4, var5)     
        f_values.append(-f_val)
        res = opt.tell(next_point, f_val)
        iterations_over_cycles.append(i + 1)

        if -f_val > max_value_found:
            max_value_found = -f_val
            iteration_with_max_value = i + 1
            input_variable_with_max_value = next_point

        max_values.append(max_value_found)
        
    max_values_over_cycles.append(max_values)
    f_values_over_cycles.append(f_values)
    iterations_over_cycles.append(iteration_with_max_value)
    input_variables_over_cycles.append(input_variable_with_max_value)

    all_max_values.extend(max_values)

    cycle_max_mean = np.mean(max_values)
    cycle_mean = np.mean(f_values)
    cycle_std = np.std(f_values)
    cycle_max_std = np.std(max_values)
    mean_values.append(cycle_mean)
    std_values.append(cycle_std)

    print(f"Cycle {cycle + 1}: Mean Max Val = {cycle_max_mean:.6f}, Mean Max std = {cycle_max_std:.6f}, Mean std = {cycle_std:.6f}  Mean Val = {cycle_mean:.6f}")

    plt.close()

mean_max_values_per_iteration = np.mean(max_values_over_cycles, axis=0)
std_max_values_per_iteration = np.std(max_values_over_cycles, axis=0)

mean_f_values_per_iteration = np.mean(f_values_over_cycles, axis=0)
std_f_values_per_iteration = np.std(f_values_over_cycles, axis=0)

plt.figure(figsize=(10, 5.5))
for cycle, max_values in enumerate(max_values_over_cycles):
    plt.plot(range(1, len(max_values) + 1), max_values, marker='o', linestyle='-', label=f'Cycle {cycle + 1}')
for cycle, max_value, iteration, input_variable in zip(range(num_cycles), max_values_over_cycles, iterations_over_cycles, input_variables_over_cycles):
    print(f"Cycle {cycle + 1}: Maximum Value = {max_value[-1]}, Iteration = {iteration}, Input Variable = {input_variable}")
plt.xlabel("Total Iterations", fontsize = 16)
plt.ylabel("Maximum Value", fontsize = 16)
plt.title("Bayesian optimization for EI", fontsize = 16)
plt.legend(loc='lower right', prop={'size': 16})
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 5.5))
plt.plot(range(1, len(mean_max_values_per_iteration) + 1), mean_max_values_per_iteration, marker='o', linestyle='-', label='Mean')
plt.fill_between(range(1, len(mean_max_values_per_iteration) + 1), mean_max_values_per_iteration - std_max_values_per_iteration, mean_max_values_per_iteration + std_max_values_per_iteration, color='gray', alpha=0.5, label='Std Dev')
plt.xlabel("Total Iterations", fontsize = 16)
plt.ylabel("Mean and standard deviation", fontsize = 16)
plt.title("Bayesian optimization for EI", fontsize = 16)
plt.legend(loc='lower right', prop={'size': 16})
plt.grid(False)
plt.show()


# In[ ]:





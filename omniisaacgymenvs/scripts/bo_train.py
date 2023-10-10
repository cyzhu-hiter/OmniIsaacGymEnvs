import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from itertools import product
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import random
import subprocess
import pandas as pd

np.random.seed(2609)
random.seed(2609)


def read_csv_to_dataframe(filename="runs/setting_record_init.csv"):
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(filename, index_col=None)

def get_tensorboard_target(set_num):
    """
    Load TensorBoard logs from the given directory into a pandas DataFrame.
    """
    # Initialize an event accumulator
    event_acc = EventAccumulator(f'runs/{set_num}/summaries')
    event_acc.Reload()  # Load all the events

    # Extract scalars
    data = {}

    # Here, we're focusing on scalar data, but there are other types of data you can extract
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        values = [e.value for e in events]
        steps = [e.step for e in events]
        data[tag] = pd.Series(values, index=steps)

    return data

# Method help convert DataFrame row values into dict, and save interval numbers.
def list_to_dict(lst, target=None):
    # Calculate the intervals
    iteration_num = len(lst) - 1 if target is None else len(lst) - 2
    intervals = [int(lst[0])] + [int(lst[i+1]) - int(lst[i]) for i in range(iteration_num)]

    
    # Create the dictionary
    result = {f'N{i+1}': intervals[i] for i in range(len(intervals))}
    result['target'] = lst[-1] if target is None else target
    return result

def calculate_target(i):
    return 100 - int((get_tensorboard_target(f'set{i}')["episode_lengths/step"] > 490.0).idxmax()/8192)


#define the black box function
def black_box_function(q_s1, q_s2, q_s3, q_s4, q_s5, q_s6, q_s7, q_s8, q_s9):

    # Get the performance from the parallel RL training
    performance = get_performance(q_s1, q_s2, q_s3, q_s4, q_s5, q_s6, q_s7, q_s8, q_s9)
    
    return performance

# Exploration factor kappa
def dynamic_delta(num_priors, initial_delta, scaling_factor):
    delta = initial_delta / (1 + scaling_factor * num_priors)
    return delta

def sqrt_beta(t=6, delta=0.5, d=4):
    # Confidence Bound for Fixed Budget (CBFB) kauffman et al 2017:
    value = np.sqrt((2 * np.log(t**(d + 2) * np.pi**2 / (3 * delta))) / t)
    return value
        

# Generate discrete search space
import itertools

print("Start Initializing the grid points.")
# Define the number of variables and the target sum
num_variables = 9
target_sum = 32

# Generate all combinations of 9 variables that sum to 32
grid_points = []
for combo in itertools.combinations_with_replacement(range(target_sum + 1), num_variables):
    if sum(combo) == target_sum:
        grid_points.append(combo)
    
grid_points = np.array(grid_points)   


  
#Initial Priors
INITIALIZE_DATA = False

# Scratch the Initial csv file
if INITIALIZE_DATA:
    # Define the lists
    set1 = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    set2 = [4, 6, 10, 14, 18, 22, 26, 28, 32]
    set3 = [1, 3, 8, 10, 14, 19, 27, 28, 32]
    set4 = [2, 3, 5, 10, 14, 19, 21, 26, 32]
    set5 = [2, 3, 5, 8, 9, 17, 18, 28, 32]
    set6 = [2, 5, 7, 11, 16, 24, 28, 30, 32]

    # Convert the lists to a DataFrame
    data = [set1, set2, set3, set4, set5, set6]
    df = pd.DataFrame(data, columns=[f'N{i}' for i in range(1, 10)])

    # Add an extra 'target' column and set it with default value (e.g. 0)
    for i in range(1,7):
        df['target'] = calculate_target(i)

    # Save the DataFrame to a CSV file
    df.to_csv('runs/setting_record_init.csv', index=False)

priors = read_csv_to_dataframe()

num_set = len(priors)
priors = [list_to_dict(priors.iloc[i]) for i in range(num_set)]


for i in range(num_set+1, 200):
    print("Start Beysian Optimization Iteration {i}:")
    #Dinamic Exploration parameter
    ddelta = dynamic_delta(len(priors)+1, 0.6, 1)
    kappa = sqrt_beta(t=len(priors)+1, delta=ddelta)  # UCB kappa parameter/ t should be number of priors + 1 

    # Initialize the Gaussian process regressor
    kernel = RBF(length_scale=1.0)

    regressor = GaussianProcessRegressor(kernel=kernel,alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=13)

    # Prepare the data for Gaussian process regression
    P = np.array([[p['N1'], p['N2'], p['N3'], p['N4'],p['N5'], p['N6'], p['N7'], p['N8'],p['N9']]  for p in priors])
    Z = np.array([p['target'] for p in priors])

    # Fit the Gaussian process regressor
    regressor.fit(P, Z)


    mu, sigma = regressor.predict(grid_points, return_std=True)
    UCB = mu + kappa * sigma
    best_index = np.argmax(UCB)
    # Retrieve the best solution and its corresponding objective values
    best_solution = grid_points[best_index]
    best_objectives = UCB[best_index]

    print("Round{}, Point suggestion : {}, value: {}".format(i, best_solution, best_objectives))

    best_solution_str = '[' + ', '.join(map(str, best_solution.cumsum())) + ']'

    cmd = [
        '/isaac-sim/python.sh',
        'scripts/rlgames_train.py',
        'task=Cartpole',
        'headless=True',
        'group_wise=True',
        f'group_marks={best_solution_str}',
        f'experiment=set{i}'
    ]
    
    print(cmd)
    # Execute the command
    completed_process = subprocess.run(cmd)

    # # Check if the command ran successfully
    if completed_process.returncode != 0:
        ValueError("Command failed with return code:{}".format(completed_process.returncode))
    else:
        print("Command executed successfully!")

    new_prior = list_to_dict(best_solution.tolist(), target=calculate_target(i))
    priors.append(new_prior)

    with open('runs/setting_record.csv','a') as f:
        priors.to_csv(f, header=False, index=False)
    


    
       


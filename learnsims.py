import matplotlib.pyplot as plt
import numpy as np

import ActiveLearner as AL

import generateData as generate

import steps_simulator as sim

import pickle

### Parameters for simulations
NT = 1000
cell = ['10_20', '10_30', '10_40', '20_30', '20_40', '30_40', '7_10']
mesh = [5,2]

NP0 = [1e3, 1e4, 1e5]
D =  [1e-12, 1e-13, 1e-14, 1e-15, 0]
k_a = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
t_final =  [1*3600, 12*3600, 24*3600, 48*3600]
unif = [0,1]

p = [cell, mesh, D, NP0, k_a, t_final, unif]

### Load up initial simulator with features and feature labels
feature_labels = ['Lcell', 'Ldomain','mesh', 'NP0', 'D', 'k_a', 't_final', 'unif']
filenames = ['parameter_space', 'test_out', 'df_train', 'train_out']
s = sim.STEPS(feature_labels, filenames)

### Get intial data (which is used to seed the learner below)
all_data = s.loadParameterSpace()
traindata = s.generateTrainData()

with open('df_out_single_cell', "rb") as output_file:
	df_raw = pickle.load(output_file)

all_d = set(list(all_data.index))
sampled_d = set(list(df_raw['index'].unique()))

### Parameters for the learner
init_n = 5
n_samples = 5
iterations = 50
initial_sample = np.random.choice(list(all_d-sampled_d), init_n)
q = 'entropy'
p = [iterations, n_samples]

### Reset simulator (so settings aren't messed up)
s = sim.STEPS(feature_labels, filenames)

### Run the learner
learner = AL.ActiveLearner(s, p, initial_sample, costs=True, epsg = 0.05, queryMethod=q, unknownCosts=True) 
score, samples = learner.Learn()
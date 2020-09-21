from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  

import matplotlib.pyplot as plt
import numpy as np

import ActiveLearner as AL

def loadData(df):
    
    x_raw = df['data']
    y_raw = df['target']

    return [x_raw, y_raw]

### ------------- Load Data
df = load_breast_cancer()
data = loadData(df)

### ------------- Choose Learning parameters
pct_of_data = 0.05 
init_n = int(len(data[0])*pct_of_data) 
n_samples = 2
certainty_threshold =0.8
iterations = 50
initial_sample = np.random.choice(np.arange(len(data[0])), init_n)
p = [iterations, certainty_threshold, n_samples]

### ------------- Prepare data for fitting
# Dim reduce for cleaner fit
scaler = StandardScaler() 
scaler.fit(data[0]) 
data[0] = scaler.transform(data[0])

# Define our PCA transformer and fit it onto our raw dataset.
pca = PCA(n_components=2, random_state=100)
transformed_brest_cancer = pca.fit_transform(X=data[0])

# Isolate data for plotting.
x_component, y_component = transformed_brest_cancer[:, 0], transformed_brest_cancer[:, 1]

### ------------- Active Learner:
learner = AL.ActiveLearner(data, p, initial_sample)#, queryMethod='entropy')  
score0 = learner.Learn()

benchmarkLearner = AL.ActiveLearner(data, [iterations, 2, n_samples], initial_sample, queryMethod=False) 
score1 = benchmarkLearner.Learn()

#check learning ability
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
plt.plot(score0[0], label='Active Learner')
plt.plot(score1[0], label='Random sampling')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Accuracy')
plt.legend()
plt.show()
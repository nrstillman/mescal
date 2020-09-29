from sklearn.metrics import hamming_loss
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import sklearn.gaussian_process as gp

import matplotlib.pyplot as plt
import numpy as np

class ActiveLearner:

    def __init__(self, parameters, seed, costs = False, queryMethod='leastCertain', 
            unknownCosts = False, epsf = 0, epsg = 0, test_idx = False, simulator = False):
        
        self.simulator = simulator
        self.model = RandomForestClassifier(n_estimators=200, max_features=5)
        self.iterations = parameters[0]
        self.n_samples = parameters[1]
        # Option initialisation for learner
        self.queryMethod = queryMethod
        self.samples = seed
        self.costs = costs    
        self.unknownCosts = unknownCosts 
        self.epsf = epsf
        self.epsg = epsg
        # Create dataframe of available test data to keep track of what data has been sampled
        self.all_data = simulator.loadParameterSpace()

    def intialiseLearner(self):
        # Sample data
        train, test = self.evaluateSamples()
        # Score & rank data
        score, pred, rank = self.fitModel(train, test)

        print('\nInitial Score:  {}'.format(score))

        return score, rank

    def rankData(self, new_data, costs=False):

        # Calculate certainty - here we use sklearn's inbuilt prediction function
        certainty = []
        pred = self.model.predict(new_data)
        pred_prob = self.model.predict_proba(new_data)

        for i, idx in enumerate(pred):
            certainty.append(pred_prob[i][idx])
 
        # epsilon-frugal  <- could write this better...
        if costs is not 1:
            for i in range(len(costs)):
                if np.random.random() < self.epsf: 
                    costs[i] = 1 

        # Return ranking based on query method
        if self.queryMethod == 'leastCertain':
            return np.argsort(certainty/costs)
        elif self.queryMethod == 'margin':
            return np.argsort(abs(pred_prob[:,0] - pred_prob[:,1])*costs)
        elif self.queryMethod == 'entropy':
            return np.argsort(-(pred_prob*np.ma.log(pred_prob).filled(0)).sum(1)/costs)
        else:    
            a = np.arange(len(certainty))
            np.random.shuffle(a)
            return a

    def chooseTestParameters(self, rank=False):
        # build list of potential samples 
        potential_samples = np.delete(self.all_data.index, self.samples, 0)[rank[:self.n_samples]]

        # epsilon-greedy swap between potential samples and available samples
        for i in range(len(potential_samples)):
            if np.random.random() < self.epsg: 
                random_idx = np.random.randint(0, len(self.self.all_data))
                potential_samples[i], self.self.all_data.index[random_idx] = self.all_data.index[random_idx], potential_samples[i]

        self.samples = np.concatenate([self.samples, potential_samples])

    def evaluateSamples(self):
        #simulator that gives target based on features (and costs)
        train = self.simulator.generateTrainData()
        test = self.simulator.generateTestData(self.samples)
        
        # # Take only parameters with simulation data
        # train = [data[0][self.samples], data[1][self.samples]]
        # # Remove this data from test data (unassessed parameters...)
        # test = [np.delete(data[0], self.samples, 0), np.delete(data[1], self.samples, 0)]
        self.all_data = all_data.drop(self.samples)

        return train, test

    def learnCosts(self, train, costs, test_costs = False):
        # Fit costs regression model (using Gaussian Process)
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
        cmodel = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
        cmodel.fit(train, costs)
        costs, std = cmodel.predict(self.all_data, return_std=True)
        # if test_costs is not False: print('MSE = {}'.format(((costs-test_costs)**2).mean()))
        return costs

    def fitModel(self, train, test):
        # Fit model
        self.model.fit(train[0], train[1])
        # If costs are known, fit cost model:
        if self.unknownCosts:
            costs = self.learnCosts(train[0], self.all_data, train[2], test_costs = False)#self.costs[np.delete(self.test_idx, self.samples, 0)])
        else:
            costs = test[2] if self.costs is not False else 1
        # get evaluation metrics
        pred = self.model.predict(test[0])
        score = 1-hamming_loss(test[1], pred) if len(test) > 1 else -1
        rank = self.rankData(self.all_data, costs)

        return score, pred, rank

    def Learn(self):
        #Initialise same sample data to prime the model
        scores = []
        score, rank = self.intialiseLearner()
        scores.append(score)
        for n in range(self.iterations):
            self.chooseTestParameters(rank)
            train, test = self.evaluateSamples()
            # fit & predict using model
            score, pred, rank = self.fitModel(train, test)
            scores.append(score)

            if score != -1:
                print('\rRun {}, Score:{:.4g}'.format(n,score), end =" ") 

            self.simulator.combineData(train,test)
        return scores, self.samples
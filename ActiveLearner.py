from sklearn.metrics import hamming_loss
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np

class ActiveLearner:

    def __init__(self,simulator, parameters, seed, costs = False, queryMethod='leastCertain'):
        
        self.data = simulator
        self.model = RandomForestClassifier(n_estimators=200, max_features=5)
        self.iterations = parameters[0]
        self.n_samples = parameters[1]

        self.queryMethod = queryMethod
        self.samples = seed

        self.costs = costs    

        # Create list of indexes of available test data to keep track of what data has been sampled
        self.test_idx = np.arange(len(self.data[0]))

    def intialiseLearner(self):
        # Sample Data
        train, test = self.generateData()

        score, pred, rank = self.fitModel(train, test)

        print('\nInitial Score:  {}'.format(score))
        # self.rankData(test)

        return score, rank

    def rankData(self, test, costs=False):

        # Calculate certainty - here we use sklearn's inbuilt prediction function
        certainty = []
        pred = self.model.predict(test[0])
        pred_prob = self.model.predict_proba(test[0])

        costs = self.costs[np.delete(self.test_idx, self.samples, 0)] if self.costs is not False else 1
        for i, idx in enumerate(pred):
            certainty.append(pred_prob[i][idx])

        # Return ranking based on query method
        if self.queryMethod == 'leastCertain':
            return np.argsort(certainty)
        elif self.queryMethod == 'margin':
            return np.argsort(abs(pred_prob[:,0] - pred_prob[:,1])*costs)
        elif self.queryMethod == 'entropy':
            return np.argsort((pred_prob*np.ma.log(pred_prob).filled(0)).sum(1)/costs)
        elif self.queryMethod == False:    
            a = np.arange(len(certainty))
            np.random.shuffle(a)
            return a

    def chooseTestParameters(self, rank=False):
        potential_samples = np.delete(self.test_idx, self.samples, 0)[rank[:self.n_samples]]
        self.samples = np.concatenate([self.samples, potential_samples])

        return 0

    def generateData(self):
       # Replace this with simulator function eg self.simulator(self.samples) -> train,test
        train = [self.data[0][self.samples], self.data[1][self.samples]]
        # Remove this data from test data
        test = [np.delete(self.data[0], self.samples, 0), np.delete(self.data[1], self.samples, 0)]

        return train, test

    def fitModel(self, train, test):

        # Fit model
        self.model.fit(train[0], train[1])
        # get evaluation metrics
        pred = self.model.predict(test[0])
        score = 1-hamming_loss(test[1], pred)
        rank = self.rankData(test, self.costs)

        return score, pred, rank

    def Learn(self):
        #Initialise same sample data to prime the model
        scores = []
        score, rank = self.intialiseLearner()
        scores.append(score)
        for n in range(self.iterations):
            self.chooseTestParameters(rank)
            train,test = self.generateData()

            # fit & predict using model
            score, pred, rank = self.fitModel(train, test)

            print('\rRun {}, Score:{:.4g}'.format(n,score), end =" ")
            scores.append(score)
        return scores, self.samples
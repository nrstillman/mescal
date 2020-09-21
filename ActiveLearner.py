from sklearn.metrics import hamming_loss
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np

class ActiveLearner:

    def __init__(self,data, parameters, sample, costs = False, a = 1, queryMethod='leastCertain'):
        
        self.data = data
        self.model = RandomForestClassifier(n_estimators=200, max_features=5)
        self.iterations = parameters[0]
        self.threshold = parameters[1]
        self.n_samples = parameters[2]

        self.queryMethod = queryMethod
        self.samples = sample

        self.costs = costs    
        self.a =a

    def intialiseLearner(self):
        # Sample Data
        train,test = self.generateData()

        score, pred, rank = self.fitModel(train, test)

        print('\nInitial Score:  {}'.format(score))
        self.rankData(test)

        return score, rank

    def rankData(self, test, costs=False):

        # Calculate certainty - here we use sklearn's inbuilt prediction function
        certainty = []
        pred = self.model.predict(test[0])
        pred_prob = self.model.predict_proba(test[0])

        for i, idx in enumerate(pred):
            certainty.append(pred_prob[i][idx])

        # Return ranking based on query method
        if self.queryMethod == 'leastCertain':
            return np.argsort(certainty)
        elif self.queryMethod == 'margin':
            return np.argsort(abs(pred_prob[:,0] - pred_prob[:,1]))
        elif self.queryMethod == 'entropy':
            return np.argsort((pred_prob*np.ma.log(pred_prob).filled(0)).sum(1))
        elif self.queryMethod == False:    
            a = np.arange(len(certainty))
            np.random.shuffle(a)
            return a

    def chooseTestData(self, rank=False):
        # Create list of indexes of available test data to keep track of what data has been sampled
        test_idx = np.arange(len(self.data[0]))
        potential_samples = np.delete(test_idx, self.samples, 0)[rank[:self.n_samples]]

        if self.costs is not False:
            # calculate probability of choosing based on costs
            if type(self.a) is not float:
                prob_choice = 1- np.exp(-self.a[0]*self.costs[potential_samples]) 
                potential_samples = potential_samples[prob_choice > self.a[1]]
                prob_choice = prob_choice[prob_choice > self.a[1]]
            else:
                prob_choice = 1- np.exp(-self.a*self.costs[potential_samples]) 

            if len(potential_samples) < self.n_samples:
                return 
            prob_choice /= prob_choice.sum() # normalize to sum to 1
            new_sample = np.random.choice(potential_samples, self.n_samples, replace=False, p=prob_choice)
            if len(new_sample) > 0:
                print(len(new_sample))
                samples = np.concatenate([self.samples, new_sample])
            else:
                print('Not enough samples that meet cost criteria')            
                samples = np.concatenate([self.samples, 
                    np.random.choice(potential_samples, self.n_samples, replace=False)])

        else:
            if len(potential_samples) < self.n_samples:
                return 
            # samples = np.concatenate([self.samples, 
            #     np.random.choice(potential_samples, self.n_samples, replace=False)])
            samples = np.concatenate([self.samples, potential_samples])

        return samples

    def generateData(self):
       # Select this data
        train = [self.data[0][self.samples], self.data[1][self.samples]]
        # Remove this data from test data
        test = [np.delete(self.data[0], self.samples, 0), np.delete(self.data[1], self.samples, 0)]

        return train, test

    def fitModel(self, train, test):

        # Fit model
        self.model.fit(train[0], train[1])

        pred = self.model.predict(test[0])
        score = 1-hamming_loss(test[1], pred)

        rank = self.rankData(test)

        return score, pred, rank

    def Learn(self):
        #Initialise same sample data to prime the model
        scores = []
        score, rank = self.intialiseLearner()
        scores.append(score)
        for n in range(self.iterations):
            self.samples = self.chooseTestData(rank)
            train,test = self.generateData()

            # fit & predict using model
            score, pred, rank = self.fitModel(train, test)

            print('\rRun {}, Score:{:.4g}'.format(n,score), end =" ")
            scores.append(score)
        return scores, self.samples
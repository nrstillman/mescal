import generateData as gen
import os
import pickle
import itertools
import pandas as pd

class STEPS:

    def __init__(self, features, feature_labels, filenames):

        self.feature = features
        self.feature_labels = feature_labels

        self.testfilepath = filenames[0]
        self.test_out = filenames[1]

        self.trainfilepath = filenames[2]
        self.train_out = filenames[3]

    def loadParameterSpace(self):

        ### dataframe
        if os.path.exists(self.testfilepath) is False:
            raise Exception("Load option is selected but no file exists.")

        else:
            with open(self.testfilepath, "rb") as output_file:
                df = pickle.load(output_file)
        print('Total parameter space loaded')
        return df

    def generateTrainData(self):

        def initialTrainData():
            #TODO: Need to add this by running usual gen scripts with sampling here
            df_train = pd.DataFrame(columns=self.feature_labels)
            df_train.insert(7, 'Run', 0)
            df_train.insert(8, 'NPR', 0)
            df_train.insert(9, 'NPi', 0)
            df_train.insert(10, 'runtime', 0)

            return df_train

        def loadTrainData():
            #Prepare to populate output dataframe
            #check if it exists and append on to dataframe
            if os.path.exists(self.trainfilepath):
                # Create backup
                backupname = self.trainfilepath + '.bak'
                if os.path.exists(backupname):
                    os.remove(backupname)
                os.system('cp {} {}'.format(self.trainfilepath, backupname))  

                with open(self.trainfilepath, "rb") as output_file:
                    df_raw = pickle.load(output_file)

            else:
                raise Exception("Training data is either under a different name or not intialised.")

            df_raw["Lcell"] = df_raw["Lcell"].astype(str).astype(int)
            df_raw["Ldomain"] = df_raw["Ldomain"].astype(str).astype(int)

            df_train = pd.DataFrame()
            for i in range(len(df_raw['index'].unique())):
                df_train[i] = df_raw[df_raw['index'] == df_raw['index'].unique()[i]].mean()

            return df_train.transpose()
        
        df_train = loadTrainData()
        return [df_train[self.feature_labels], df_train['target']]


    def generateTestData(self, samples):
        #Samples simulation data based on samples and overall bounded parameter space
        def saveTestData():

                df = pd.DataFrame(columns = self.feature_labels)

                for idx, x in enumerate(itertools.product(*self.features)):
                    df.loc[idx] = x 

                # Create backup
                if os.path.exists(self.testfilepath):
                    backupname = self.testfilepath + '.bak'
                    if os.path.exists(backupname):
                        os.remove(backupname)
                    os.rename(self.testfilepath, backupname)

                with open(self.testfilepath, "wb") as output_file:
                    pickle.dump(df, output_file)
        
        #Load parameter space
        df = self.loadParameterSpace()

        #Prepare dataframe for saving results
        df_out = pd.DataFrame(columns=df.columns)
        #Run both scenarios
        df_wmxd = gen.wellMixed(df_out, df.iloc[samples,:])        
        df_singlecell = gen.singleCell(df_out, df.iloc[samples,:], Nruns=2)
        
        df_test = pd.DataFrame()
        tmp = pd.DataFrame()

        for i in range(len(df_singlecell['index'].unique())):
            df_test[i] = df_singlecell[df_singlecell['index'] == df_singlecell['index'].unique()[i]].mean()
            tmp[i] = df_wmxd[df_wmxd['index'] == df_singlecell['index'].unique()[i]].mean()
        
        df_test = df_test.transpose()
        tmp = tmp.transpose()
        
        npi_diff = df_test.NPi/tmp.NPi
        df_test['target'] = 1*(npi_diff > 0.2)
        df_test.drop(columns='Run', inplace=True)

        with open(self.test_out, "wb") as output_file:
            pickle.dump(df_singlecell, output_file)

        return [df_test[self.feature_labels], df_test['target']]


    # def combineData(self, train, test):



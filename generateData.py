import numpy as np
import random
import pandas as pd

import scipy.integrate as sp 

import itertools
from numpy.random import randint

import matplotlib.pyplot as plt
import os
import time

import pickle

import run_sim

def wellMixed(samples = False, NT =1000, Nruns = 10, 
                single_cell_filename = "df_out_single_cell", output_filename  = 'df_out_wmxd'):
    
    def run_iter(stepsParams, N, sim, rctn):
        ssa = []
        for j in range(N):        
            score = run_sim.main(sim, stepsParams)
            ssa.append(score[1][:,0,rctn])
        return ssa

    if samples is not True:
        with open(single_cell_filename, "rb") as output_file:
            samples = pickle.load(output_file)

    # wmxd_params = df[['D', 'k_a', 'Lcell', 't_final', 'index']]
    df_out = pd.DataFrame(columns=samples.columns)

    ### prepare STEPS params and call run_iter above
    for i in range(len(samples)):
        p = samples.iloc[i]

        opt = {"general":{"NT":int(NT), "t_final": int(p['t_final'])}}
        VC = {'S': 1e-6*int(p['Ldomain']), 'P0':{"D" :p['D'], "k_a" :p['k_a'], "NP0" : p['NP0']}}
        CC = {'S': 1e-6*int(p['Ldomain']), 'P0':{"k_a" :p['k_a'],"D" :p['D']}}

        stepsParams = {"category":opt, "cell":{"VC":VC, "CC":CC}}

        resi = run_iter(stepsParams, Nruns, 'wmxd_chamber', [2,3])

        df_run = pd.DataFrame(columns=samples.columns)

        for n in range(Nruns):
            df_run = df_run.append(samples.iloc[i], ignore_index=True)

        # for n in range(Nruns):
        df_run['Run'] = range(Nruns)

        df_run['NPR'] = [resi[n][-1][0] for n in range(Nruns)]
        df_run['NPi'] = [resi[n][-1][1] for n in range(Nruns)]

        df_out = df_out.append(df_run, ignore_index=True)

        with open(output_filename, "wb") as output_file:
            pickle.dump(df_out, output_file)
    
        print('\r--------Number {} out of {} completed ({:.2g}% Complete)--------'.format(i+1, len(samples), 100*(i+1)/(len(samples))), end =" ")

    print('\nData generation of comparison wmxd scenario complete')

    return df_out

def singleCell(parameters, NT=1000, Nruns=10, sample_size=100, opt='load', samples = False,
        picklename="rd_data_for_learning.pickle", output_filename = 'df_out_single_cell'):

    def run_iter(stepsParams, N, sim, rctn):
        ssa = []
        timings = []
        for j in range(N):        
            start = time.time()
            score = run_sim.main(sim, stepsParams)
            ssa.append(score[1][:,0,rctn])
            end = time.time()   
            timings.append(end-start)     
        return ssa,timings

    def loadData(opt, picklename, samples):
        ### dataframe
        if opt == 'save':

            df = pd.DataFrame(columns = ['cell','mesh', 'NP0', 'D', 'k_a', 't_final', 'unif'])

            for idx, x in enumerate(itertools.product(cell, mesh,NP0, D, k_a, t_final, unif)):
                df.loc[idx] = x 

            # Create backup
            if os.path.exists(picklename):
                backupname = picklename + '.bak'
                if os.path.exists(backupname):
                    os.remove(backupname)
                os.rename(picklename, backupname)


            with open(picklename, "wb") as output_file:
                pickle.dump(df, output_file)

        elif opt == 'load':
            if os.path.exists(picklename) is False:
                raise Exception("Load option is selected but no file exists.")

            else:
                with open(picklename, "rb") as output_file:
                    df = pickle.load(output_file)

            samples = df.sample(n=2, random_state=2).reset_index()
            #accounting for previous runs
            # samples = samples.drop(list(range(31)) +  [32, 50, 55, 57, 66, 70, 74])
    
        return samples

    def createOutput(samples, output_filename):
        #Prepare to populate output dataframe
        #check if it exists and append on to dataframe
        if os.path.exists(output_filename):
            # Create backup
            backupname = output_filename + '.bak'
            if os.path.exists(backupname):
                os.remove(backupname)
            os.system('cp {} {}'.format(output_filename, backupname))  

            with open(output_filename, "rb") as output_file:
                df_out = pickle.load(output_file)
        #otherwise make new
        else:
            df_out = pd.DataFrame(columns=samples.columns)
            df_out.insert(7, 'Run', 0)
            df_out.insert(8, 'NPR', 0)
            df_out.insert(9, 'NPi', 0)
            df_out.insert(10, 'runtime', 0)

        return df_out

    ### Load data (if no initial sample is provided)
    if samples is False: 
        samples = loadData(opt, picklename, samples)
    
    ### prepare the output data frame (to keep track of what sims are being run)
    df_out = createOutput(samples, output_filename)

    ### prepare STEPS params and call run_iter above
    for i in range(len(samples)):
        p = samples.iloc[i]

        mesh = "{}_{}_{}um".format(int(p['Lcell']), int(p['Ldomain']), int(p['mesh']))

        opt = {"general":{"NT":int(NT), "t_final" :int(p['t_final'])}, "opt":{"mesh": mesh,"unif":p['unif']}}
        # %VC = {'P0':{"D" : x[4], "T":1, "k_a" : k,"D" : x[4], "NP0" : x[3]}, "NR":NR}
        CC = {'P0':{"k_a" :p['k_a'],"D" : p['D'], "NP0" :p['NP0'] }}
        # stepsParams = {"category":opt, "cell":{"VC":VC, "CC":CC}}

        stepsParams = {"category":opt, "cell":{"CC":CC}}

        resi, timings = run_iter(stepsParams, Nruns, 'single_cell', [2,3])

        df_run = pd.DataFrame(columns=samples.columns)

        for n in range(Nruns):
            df_run = df_run.append(samples.iloc[i], ignore_index=True)

        # for n in range(Nruns):
        df_run['Run'] = range(Nruns)

        df_run['NPR'] = [resi[n][-1][0] for n in range(Nruns)]
        df_run['NPi'] = [resi[n][-1][1] for n in range(Nruns)]
        df_run['runtime'] = timings

        df_out = df_out.append(df_run, ignore_index=True)

        with open(output_filename, "wb") as output_file:
            pickle.dump(df_out, output_file)

        print('\r--------Number {} out of {} completed ({:.2g}% Complete)--------'.format(i+1, len(samples), 100*(i+1)/(len(samples))), end =" ")

    print('\nData generation stage complete')
    return df_out

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

def wellMixed(df_out, samples, NT =1000, Nruns = 10, output_filename  = 'df_out_wmxd_tmp'):
    
    def run_iter(stepsParams, N, sim, rctn):
        ssa = []
        for j in range(N):        
            score = run_sim.main(sim, stepsParams)
            ssa.append(score[1][:,0,rctn])
        return ssa

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

        df_run['index'] = samples.index[i]
        df_run['Run'] = range(Nruns)

        df_run['NPR'] = [resi[n][-1][0] for n in range(Nruns)]
        df_run['NPi'] = [resi[n][-1][1] for n in range(Nruns)]

        df_out = df_out.append(df_run, ignore_index=True)

        with open(output_filename, "wb") as output_file:
            pickle.dump(df_out, output_file)
    
        print('\r--------Number {} out of {} completed ({:.2g}% Complete)--------'.format(i+1, len(samples), 100*(i+1)/(len(samples))), end =" ")

    print('\nData generation of comparison wmxd scenario complete')

    return df_out

def singleCell(df_out, samples, NT=1000, Nruns=10, output_filename = 'df_out_single_cell_tmp'):

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

        df_run['index'] = int(samples.index[i])
        df_run['Run'] = range(Nruns)

        df_run['NPR'] = [resi[n][-1][0] for n in range(Nruns)]
        df_run['NPi'] = [resi[n][-1][1] for n in range(Nruns)]
        df_run['cost'] = timings

        df_out = df_out.append(df_run, ignore_index=True)

        with open(output_filename, "wb") as output_file:
            pickle.dump(df_out, output_file)

        print('\r--------Number {} out of {} completed ({:.2g}% Complete)--------'.format(i+1, len(samples), 100*(i+1)/(len(samples))), end =" ")

    print('\nData generation stage complete')
    return df_out

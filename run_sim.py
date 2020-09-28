#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time
import pickle, sys
import os.path, csv
import json
import random

import setup

import argparse

parser = argparse.ArgumentParser(description='Run STEPS simulation of stochastic reaction-diffusion in spatial domain.')
parser.add_argument("--p", default='wmxd', help="name of simulation to run", metavar='path')
parser.add_argument("--u", default=0, help="update configure file ", metavar='data')

def main(path, data):

    path = "{}_config.xml".format(path)
    sim_opt = setup.get_sim_data(path)

    if data != 0: 
        setup.update_config(data, path)
    # print("\nRunning STEPS simulation")
    
    sim_opt = setup.get_sim_data(path)

    fitness = []
    for run in range(int(sim_opt['opt'].N_runs)): 
        # prepare tissue data
        if sim_opt['opt'].tissue == 'wmxd':
            # wmxd of cells
            tissue = np.zeros(int(sim_opt['general'].N_cells))
            tissue_map  = {0: 'VC', 1: 'CC'}
        elif sim_opt['opt'].tissue == 'single_cell':
            # chamber of cells
            tissue = np.zeros(int(sim_opt['general'].N_cells))
            tissue_map  = {0: 'CC'}
        else:
            tissue = pickle.load( open( sim_opt['opt'].tissue, "rb" ) )
            section = random.randint(0,len(tissue)-1)
            tissue = tissue[section,:] 
            tissue_map  = {0: 'VC', 1 : 'CC', 2 : 'CSC'}
        
        # prepare cell data
        data = setup.get_cell_data(path)
        for n in range(len(tissue_map)):
            data[n]['N_cell'] = sum((tissue == n)*1)

        cells = setup.make_cells(data)

        tissue = setup.make_tissue(tissue, cells, tissue_map)

        # prepare simulation
        scenario = __import__(sim_opt['opt'].scenario)
        mdl = scenario.model(tissue)
        geom = scenario.geom(tissue, sim_opt)

        # run simulation
        start = time.time()
        treated_tissue, resi = scenario.solver(mdl, geom, tissue, sim_opt)
        end = time.time()
        time_taken = end - start
        
        # save output
        sim_str = sim_opt['general'].output_file

        # print('tissue section %s w/ fitness %.1g' % (section, setup.calc_fitness(tissue,treated_tissue)))

        #TODO: Add a save option 
        # file_name = 'result_%s_time_%.2g' % (sim_str, time_taken)

        # with open('output/'+ sim_str+'/'+ file_name + '.pickle', 'wb') as f:
        #     pickle.dump(resi, f)
        # fitness.append(setup.calc_fitness(tissue,treated_tissue))
    return (0, resi)

if __name__ == "__main__":

    args = parser.parse_args()
    path = args.p
    data = args.u
    if data != 0:
        data=json.loads(data)

    main(path,data)

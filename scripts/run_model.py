
import numpy as np
import pyemu
import pandas as pd
import flopy
import os
import pickle

# Access and modify parameters
pars = pd.read_csv('../pest/parameters.par').set_index('parname')
print(pars)

hk = [pars.loc['HK_1'].squeeze(), 
      pars.loc['HK_2'].squeeze(), 
      pars.loc['HK_3'].squeeze(), 
      pars.loc['HK_4'].squeeze()]

vk = [pars.loc['VK_1'].squeeze(), 
      pars.loc['VK_2'].squeeze(), 
      pars.loc['VK_3'].squeeze(), 
      pars.loc['VK_4'].squeeze()]

rch = [pars.loc['RCH_1'].squeeze()]

# Incorporate new parameters into model
class Parameters:
    def __init__(self, hk, vk, rch):     
        self.hk = hk
        self.vk = vk
        self.rch = rch
parameters = Parameters(hk, vk, rch)

# Load the simulation and flow model
sim = flopy.mf6.MFSimulation.load(sim_name = 'sim')
gwf = sim.get_model('pest_example')

# Load mesh
pickleoff = open('../modelfiles/mesh.pkl','rb')
mesh = pickle.load(pickleoff)
pickleoff.close()

# Load inputs
pickleoff = open('../modelfiles/inputs.pkl','rb')
inputs = pickle.load(pickleoff)
pickleoff.close()


def rewrite_model(gwf, inputs, mesh, parameters,):
    
    # Re-write recharge
    rec = []
    for icpl in range(mesh.ncpl): rec.append((icpl, parameters.rch))      
    rch_rec = {}      
    rch_rec[0] = rec

    # Assign k11, k22, k33 using np.take and array_kzones
    k11 = np.take(parameters.hk, mesh.array_kzones.astype(int))
    k22 = np.take(parameters.hk, mesh.array_kzones.astype(int))
    k33 = np.take(parameters.vk, mesh.array_kzones.astype(int))

    # Rewrite packages
    gwf.rch.stress_period_data = rch_rec
    gwf.rch.write()

    gwf.npf.k = k11
    gwf.npf.k22 = k22
    gwf.npf.k33 = k33
    gwf.npf.write()

sim = rewrite_model(gwf, mesh, inputs, parameters)
success, buff = sim.run_simulation(silent = False) 

# Get CSV output file from MODFLOW and convert to a text file
csv_file = inputs.modelname + "_observations.csv" # To write observation to
df = pd.read_csv(os.path.join(inputs.workspace, csv_file))
fname = '../pest/observations.txt'
with open(fname, 'w') as f:
    for value in df.iloc[0]:
        f.write(str(value) + '\n')
print(f'Observations written as a list to {fname}')

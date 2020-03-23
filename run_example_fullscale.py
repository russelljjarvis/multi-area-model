import numpy as np
import os

from multiarea_model import MultiAreaModel
from config import base_path

"""
Example script showing how to simulate the multi-area model
on a cluster.

We choose the same configuration as in
Fig. 3 of Schmidt et al. (2018).

"""

"""
Full model. Needs to be simulated with sufficient
resources, for instance on a compute cluster.
"""
d = {}
conn_params = {'g': -11.,
               'K_stable': os.path.join(base_path, 'K_stable.npy'),
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'input_params': input_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 10500.,
              'num_processes': 720,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params)
#p, r = M.theory.integrate_siegert()
#print("Mean-field theory predicts an average "
#      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
M.simulation.simulate()

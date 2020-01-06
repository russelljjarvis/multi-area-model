
"""
multiarea_model
==============

Simulation class of the multi-area model of macaque visual vortex by
Schmidt et al. (2018).


Classes
-------
Simulation : Loads a parameter file that specifies simulation
parameters for a simulation of the instance of the model. A simulation
is identified by a unique hash label.

"""

import json
import numpy as np
import os
import pprint
import shutil
import time

from .analysis_helpers import _load_npy_to_dict, model_iter
from config import base_path, data_path
from copy import deepcopy
from .default_params import nested_update, sim_params
from .default_params import check_custom_params
from dicthash import dicthash
from pygenn import genn_model, genn_wrapper
from scipy.stats import binom, norm
from .multiarea_helpers import extract_area_dict, create_vector_mask
try:
    from .sumatra_helpers import register_runtime
    sumatra_found = True
except ImportError:
    sumatra_found = False

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
# LIF neuron model
lif_model = genn_model.create_custom_neuron_class(
    "lif",
    param_names=[
        "C",                # Membrane capacitance
        "TauM",             # Membrane time constant [ms]
        "Vrest",            # Resting membrane potential [mV]
        "Vreset",           # Reset voltage [mV]
        "Vthresh",          # Spiking threshold [mV]
        "Ioffset",          # Offset current
        "TauRefrac",        # Refractory time [ms]
        "PoissonWeight",    # How much current each poisson spike adds [nA]
        "IpoissonTau"],     # Time constant of poisson spike integration [ms]],

    extra_global_params=[
        ("PoissonExpMinusLambda", "scalar")],

    var_name_types=[("V","scalar"), ("RefracTime", "scalar"), ("Ipoisson", "scalar")],
    derived_params=[
        ("ExpTC",                   genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
        ("Rmembrane",               genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("IpoissonExpDecay",        genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[8]))()),
        ("IpoissonInit",            genn_model.create_dpf_class(lambda pars, dt: pars[7] * (1.0 - np.exp(-dt / pars[8])) * (pars[8] / dt))()),
    ],

    sim_code="""
    scalar p = 1.0f;
    unsigned int numPoissonSpikes = 0;
    do
    {
        numPoissonSpikes++;
        p *= $(gennrand_uniform);
    } while (p > $(PoissonExpMinusLambda));
    $(Ipoisson) += $(IpoissonInit) * (scalar)(numPoissonSpikes - 1);
    if ($(RefracTime) <= 0.0)
    {
      scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson)) * $(Rmembrane)) + $(Vrest);
      $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else
    {
      $(RefracTime) -= DT;
    }
    $(Ipoisson) *= $(IpoissonExpDecay);
    """,

    reset_code="""
    $(V) = $(Vreset);
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)")

normal_clipped_model = genn_model.create_custom_init_var_snippet_class(
    "normal_clipped",
    param_names=["mean", "sd", "min", "max"],
    var_init_code="""
    scalar normal;
    do
    {
       normal = $(mean) + ($(gennrand_normal) * $(sd));
    } while (normal > $(max) || normal < $(min));
    $(value) = normal;
    """)

normal_clipped_delay_model = genn_model.create_custom_init_var_snippet_class(
    "normal_clipped_delay",
    param_names=["mean", "sd", "min", "max"],
    var_init_code="""
    scalar normal;
    do
    {
       normal = $(mean) + ($(gennrand_normal) * $(sd));
    } while (normal > $(max) || normal < $(min));
    $(value) = rint(normal / DT);
    """)
    
fixed_num_total_with_replacement_model = genn_model.create_custom_sparse_connect_init_snippet_class(
    "fixed_num_total_with_replacement",
    param_names=["total"],
    row_build_state_vars=[("x", "scalar", 0.0), ("c", "unsigned int", 0)],
    extra_global_params=[("preCalcRowLength", "unsigned int*")],
    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(binom.ppf(0.9999**(1.0 / num_pre), n=pars[0], p=float(num_post) / (num_pre * num_post))))(),
    calc_max_col_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(binom.ppf(0.9999**(1.0 / num_post), n=pars[0], p=float(num_pre) / (num_pre * num_post))))(),
    row_build_code="""
    const unsigned int rowLength = $(preCalcRowLength)[($(id_pre) * $(num_threads)) + $(id_thread)];
    if(c >= rowLength) {
       $(endRow);
    }
    const scalar u = $(gennrand_uniform);
    x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)(rowLength - c)));
    unsigned int postIdx = (unsigned int)(x * $(num_post));
    postIdx = (postIdx < $(num_post)) ? postIdx : ($(num_post) - 1);
    $(addSynapse, postIdx + $(id_post_begin));
    c++;
    """)

class Simulation:
    def __init__(self, network, sim_spec):
        """
        Simulation class.
        An instance of the simulation class with the given parameters.
        Can be created as a member class of a multiarea_model instance
        or standalone.

        Parameters
        ----------
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network to be simulated.
        params : dict
            custom simulation parameters that overwrite the
            default parameters defined in default_params.py
        """
        self.params = deepcopy(sim_params)
        if isinstance(sim_spec, dict):
            check_custom_params(sim_spec, self.params)
            self.custom_params = sim_spec
        else:
            fn = os.path.join(data_path,
                              sim_spec,
                              '_'.join(('custom_params',
                                        sim_spec)))
            with open(fn, 'r') as f:
                self.custom_params = json.load(f)['sim_params']

        nested_update(self.params, self.custom_params)

        self.network = network
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'network_label': self.network.label})

        print("Simulation label: {}".format(self.label))
        self.data_dir = os.path.join(data_path, self.label)
        try:
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'recordings'))
        except OSError:
            pass
        self.copy_files()
        print("Copied files.")
        d = {'sim_params': self.custom_params,
             'network_params': self.network.custom_params,
             'network_label': self.network.label}
        with open(os.path.join(self.data_dir,
                               '_'.join(('custom_params', self.label))), 'w') as f:
            json.dump(d, f)
        print("Initialized simulation class.")

        self.areas_simulated = self.params['areas_simulated']
        self.areas_recorded = self.params['recording_dict']['areas_recorded']
        self.T = self.params['t_sim']

    def __eq__(self, other):
        # Two simulations are equal if the simulation parameters and
        # the simulated networks are equal.
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        s = "Simulation {} of network {} with parameters:".format(self.label, self.network.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def copy_files(self):
        """
        Copy all relevant files for the simulation to its data directory.
        """
        files = [os.path.join('multiarea_model',
                              'data_multiarea',
                              'Model.py'),
                 os.path.join('multiarea_model',
                              'data_multiarea',
                              'VisualCortex_Data.py'),
                 os.path.join('multiarea_model',
                              'multiarea_model.py'),
                 os.path.join('multiarea_model',
                              'simulation.py'),
                 os.path.join('multiarea_model',
                              'default_params.py'),
                 os.path.join('config_files',
                              ''.join(('custom_Data_Model_', self.network.label, '.json'))),
                 os.path.join('config_files',
                              '_'.join((self.network.label, 'config')))]
        if self.network.params['connection_params']['replace_cc_input_source'] is not None:
            fs = self.network.params['connection_params']['replace_cc_input_source']
            if '.json' in fs:
                files.append(fs)
            else:  # Assume that the cc input is stored in one npy file per population
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                for it in fn_iter:
                    fp_it = (fs,) + it
                    fp_ = '{}.npy'.format('-'.join(fp_it))
                    files.append(fp_)
        for f in files:
            shutil.copy2(os.path.join(base_path, f),
                         self.data_dir)

    def prepare(self):
        """
        Prepare GeNN model.
        """
        self.model = genn_model.GeNNModel("float", "potjans_microcircuit",
                                          code_gen_log_level=genn_wrapper.info)
        self.model.dT = self.params['dt']
        self.model._model.set_merge_postsynaptic_models(True)
        self.model.timing_enabled = self.params['timing_enabled']
        self.model.default_var_location = genn_wrapper.VarLocation_DEVICE
        self.model.default_sparse_connectivity_location = genn_wrapper.VarLocation_DEVICE
        self.model._model.set_seed(self.params['master_seed'])
        
        quantile = 0.9999
        normal_quantile_cdf = norm.ppf(quantile)
        
        # Calculate max delay for 
        ex_delay_sd = self.network.params['delay_params']['delay_e'] * self.network.params['delay_params']['delay_rel']
        in_delay_sd = self.network.params['delay_params']['delay_i'] * self.network.params['delay_params']['delay_rel']
        max_ex_delay = self.network.params['delay_params']['delay_e'] + (ex_delay_sd * normal_quantile_cdf)
        max_in_delay = self.network.params['delay_params']['delay_i'] + (in_delay_sd * normal_quantile_cdf)
        
        self.max_inter_area_delay = max(max_ex_delay, max_in_delay)
        print("Max inter-area delay:%fms" % self.max_inter_area_delay)

        # If we should create cortico-cortico connections
        if not self.network.params['connection_params']['replace_cc']:
            self.max_intra_area_delay = 0
            for target_area in self.areas_simulated:
                # Loop source area though complete list of areas
                for source_area in self.network.area_list:
                    if target_area != source_area:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area in self.areas_simulated:
                            v = self.network.params['delay_params']['interarea_speed']
                            s = self.network.distances[target_area][source_area]
                            mean_delay = s / v

                            delay_sd = mean_delay * self.network.params['delay_params']['delay_rel']
                            self.max_intra_area_delay = max(self.max_intra_area_delay,
                                                            mean_delay + (delay_sd * normal_quantile_cdf))
            print("Max intra-area delay:%fms" % self.max_intra_area_delay)

    def create_areas(self):
        """
        Create all areas with their populations and internal connections.
        """
        self.areas = []
        for area_name in self.areas_simulated:
            a = Area(self, self.network, area_name)
            self.areas.append(a)

    def cortico_cortical_input(self):
        """
        Create connections between areas.
        """
        replace_cc = self.network.params['connection_params']['replace_cc']
        replace_non_simulated_areas = self.network.params['connection_params'][
            'replace_non_simulated_areas']
        if self.network.params['connection_params']['replace_cc_input_source'] is None:
            replace_cc_input_source = None
        else:
            replace_cc_input_source = os.path.join(self.data_dir,
                                                   self.network.params['connection_params'][
                                                       'replace_cc_input_source'])

        if not replace_cc and set(self.areas_simulated) != set(self.network.area_list):
            if replace_non_simulated_areas == 'het_current_nonstat':
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                non_simulated_cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
            elif replace_non_simulated_areas == 'het_poisson_stat':
                fn = self.network.params['connection_params']['replace_cc_input_source']
                with open(fn, 'r') as f:
                    non_simulated_cc_input = json.load(f)
            elif replace_non_simulated_areas == 'hom_poisson_stat':
                non_simulated_cc_input = {source_area_name:
                                          {source_pop:
                                           self.network.params['input_params']['rate_ext']
                                           for source_pop in
                                           self.network.structure[source_area_name]}
                                          for source_area_name in self.network.area_list}
            else:
                raise KeyError("Please define a valid method to"
                               " replace non-simulated areas.")

        if replace_cc == 'het_current_nonstat':
            fn_iter = model_iter(mode='single', areas=self.network.area_list)
            cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
        elif replace_cc == 'het_poisson_stat':
            with open(self.network.params['connection_params'][
                    'replace_cc_input_source'], 'r') as f:
                cc_input = json.load(f)
        elif replace_cc == 'hom_poisson_stat':
            cc_input = {source_area_name:
                        {source_pop:
                         self.network.params['input_params']['rate_ext']
                         for source_pop in
                         self.network.structure[source_area_name]}
                        for source_area_name in self.network.area_list}

        # Connections between simulated areas are not replaced
        if not replace_cc:
            for target_area in self.areas:
                # Loop source area though complete list of areas
                for source_area_name in self.network.area_list:
                    if target_area.name != source_area_name:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area_name in self.areas:
                            source_area = self.areas[self.areas.index(source_area_name)]
                            connect(self,
                                    target_area,
                                    source_area)
                        # Else, replace the input from source_area with the
                        # chosen method
                        else:
                            target_area.create_additional_input(replace_non_simulated_areas,
                                                                source_area_name,
                                                                non_simulated_cc_input[
                                                                    source_area_name])
        # Connections between all simulated areas are replaced
        else:
            for target_area in self.areas:
                for source_area in self.areas:
                    if source_area != target_area:
                        target_area.create_additional_input(replace_cc,
                                                            source_area.name,
                                                            cc_input[source_area.name])

    def simulate(self):
        """
        Create the network and execute simulation.
        Record used memory and wallclock time.
        """
        t0 = time.time()
        self.prepare()
        t1 = time.time()
        self.time_prepare = t1 - t0
        print("Prepared simulation in {0:.2f} seconds.".format(self.time_prepare))

        self.create_areas()
        t2 = time.time()
        self.time_network_local = t2 - t1
        print("Created areas and internal connections in {0:.2f} seconds.".format(
            self.time_network_local))

        self.cortico_cortical_input()
        t3 = time.time()
        self.time_network_global = t3 - t2
        print("Created cortico-cortical connections in {0:.2f} seconds.".format(
            self.time_network_global))

        if self.params['rebuild_model']:
            self.model.build()
            t4 = time.time()
            self.time_genn_build = t4 - t3
            print("Built GeNN model in {0:.2f} seconds.".format(self.time_genn_build))
        else:
            t4 = t3
           
        self.model.load()
        t5 = time.time()
        self.time_genn_load = t5 - t4
        print("Loaded GeNN model in {0:.2f} seconds.".format(self.time_genn_load))
        
        nest.Simulate(self.T)
        t6 = time.time()
        self.time_simulate = t6 - t5
        print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))
        self.logging()

    def register_runtime(self):
        if sumatra_found:
            register_runtime(self.label)
        else:
            raise ImportWarning('Sumatra is not installed, the '
                                'runtime cannot be registered.')


class Area:
    def __init__(self, simulation, network, name):
        """
        Area class.
        This class encapsulates a single area of the model.
        It creates all populations and the intrinsic connections between them.
        It provides an interface to allow connecting the area to other areas.

        Parameters
        ----------
        simulation : simulation
           An instance of the simulation class that specifies the
           simulation that the area is part of.
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network the area is part of.
        name : str
            Name of the area.
        """

        self.name = name
        self.simulation = simulation
        self.network = network
        self.neuron_numbers = network.N[name]
        self.synapses = extract_area_dict(network.synapses,
                                          network.structure,
                                          self.name,
                                          self.name)
        self.W = extract_area_dict(network.W,
                                   network.structure,
                                   self.name,
                                   self.name)
        self.W_sd = extract_area_dict(network.W_sd,
                                      network.structure,
                                      self.name,
                                      self.name)
        self.populations = network.structure[name]

        self.external_synapses = {}
        for pop in self.populations:
            self.external_synapses[pop] = self.network.K[self.name][pop]['external']['external']

        self.create_populations()
        self.connect_populations()
        print("created area {} with {} local nodes".format(self.name, self.num_local_nodes))

    def __str__(self):
        s = "Area {} with {} neurons.".format(
            self.name, int(self.neuron_numbers['total']))
        return s

    def __eq__(self, other):
        # If other is an instance of area, it should be the exact same
        # area This opens the possibility to have multiple instance of
        # one cortical areas
        if isinstance(other, Area):
            return self.name == other.name #and self.gids == other.gids
        elif isinstance(other, str):
            return self.name == other

    def create_populations(self):
        """
        Create all populations of the area.
        """
        neuron_params = self.network.params['neuron_params']
        v_init_params = {"mean": neuron_params['V0_mean'], "sd": neuron_params['V0_sd']}
        lif_init = {"RefracTime": 0.0, "Ipoisson": 0.0, "V": genn_model.init_var("Normal", v_init_params)}
        lif_params = {"C": neuron_params['single_neuron_dict']['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']['t_ref']}

        self.genn_pops = {}
        self.num_local_nodes = 0
        for pop in self.populations:
            assert neuron_params['neuron_model'] == 'iaf_psc_exp'

            pop_lif_params = deepcopy(lif_params)

            mask = create_vector_mask(self.network.structure, areas=[self.name], pops=[pop])
            pop_lif_params['Ioffset'] = self.network.add_DC_drive[mask][0]
            if self.network.params['input_params']['poisson_input']:
                pop_lif_params['PoissonWeight'] = self.network.W[self.name][pop]['external']['external']
                pop_lif_params['IpoissonTau'] = neuron_params['single_neuron_dict']['tau_syn_ex']
            else:
                K_ext = self.external_synapses[pop]
                W_ext = self.network.W[self.name][pop]['external']['external']
                tau_syn = self.network.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
                DC = K_ext * W_ext * tau_syn * 1.e-3 * \
                    self.network.params['rate_ext']
                pop_lif_params['Ioffset'] += DC

            # Use LIF model with Poisson input if poisson noise is used, otherwise standard LIF
            # **YUCK** really should use Poisson current source here
            pop_model = lif_model if self.network.params['input_params']['poisson_input'] else 'LIF'

            # Create GeNN population
            genn_pop = self.simulation.model.add_neuron_population(self.name + '_' + pop, 
                                                                   int(self.neuron_numbers[pop]), 
                                                                   pop_model, pop_lif_params, lif_init)
            genn_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE)

            # Set EGP containing lambda value for external input Poisson process
            if self.network.params['input_params']['poisson_input']:
                ext_input_rate = self.network.params['input_params']['rate_ext'] * self.external_synapses[pop]
                genn_pop.add_extra_global_param("PoissonExpMinusLambda",
                                                np.exp(-(ext_input_rate / 1000.0) * self.simulation.params['dt']))

            self.genn_pops[pop] = genn_pop

    def connect_populations(self):
        """
        Create connections between populations.
        """
        connect(self.simulation,
                self,
                self)

    def create_additional_input(self, input_type, source_area_name, cc_input):
        """
        Replace the input from a source area by the chosen type of input.

        Parameters
        ----------
        input_type : str, {'het_current_nonstat', 'hom_poisson_stat',
                           'het_poisson_stat'}
            Type of input to replace source area. The source area can
            be replaced by Poisson sources with the same global rate
            rate_ext ('hom_poisson_stat') or by specific rates
            ('het_poisson_stat') or by time-varying specific current
            ('het_current_nonstat')
        source_area_name: str
            Name of the source area to be replaced.
        cc_input : dict
            Dictionary of cortico-cortical input of the process
            replacing the source area.
        """
        assert False
        """
        synapses = extract_area_dict(self.network.synapses,
                                     self.network.structure,
                                     self.name,
                                     source_area_name)
        W = extract_area_dict(self.network.W,
                              self.network.structure,
                              self.name,
                              source_area_name)
        v = self.network.params['delay_params']['interarea_speed']
        s = self.network.distances[self.name][source_area_name]
        delay = s / v
        for pop in self.populations:
            for source_pop in self.network.structure[source_area_name]:
                syn_spec = {'weight': W[pop][source_pop],
                            'delay': delay}
                K = synapses[pop][source_pop] / self.neuron_numbers[pop]

                if input_type == 'het_current_nonstat':
                    curr_gen = nest.Create('step_current_generator', 1)
                    dt = self.simulation.params['dt']
                    T = self.simulation.params['t_sim']
                    assert(len(cc_input[source_pop]) == int(T))
                    nest.SetStatus(curr_gen, {'amplitude_values': K * cc_input[source_pop] * 1e-3,
                                              'amplitude_times': np.arange(dt,
                                                                           T + dt,
                                                                           1.)})
                    nest.Connect(curr_gen,
                                 tuple(
                                     range(self.gids[pop][0], self.gids[pop][1] + 1)),
                                 syn_spec=syn_spec)
                elif 'poisson_stat' in input_type:  # hom. and het. poisson lead here
                    pg = nest.Create('poisson_generator', 1)
                    nest.SetStatus(pg, {'rate': K * cc_input[source_pop]})
                    nest.Connect(pg,
                                 tuple(
                                     range(self.gids[pop][0], self.gids[pop][1] + 1)),
                                 syn_spec=syn_spec)
        """

def build_row_lengths(num_pre, num_post, num_sub_rows, num_connections):
    assert num_sub_rows > 0
    
    num_post_per_sub_row = (num_post + num_sub_rows - 1) // num_sub_rows
    num_post_remainder = num_post % num_post_per_sub_row
    
    remaining_connections = num_connections
    matrix_size = num_pre * num_post
    
    sub_row_lengths = np.empty(num_pre * num_sub_rows, dtype=np.uint32)
    for i in range(num_pre):
        last_pre = (i == (num_pre - 1))
        
        for j in range(num_sub_rows):
            last_sub_row = (j == (num_sub_rows - 1))
            
            if not last_pre or not last_sub_row:
                num_sub_row_neurons = (num_post_remainder 
                                       if num_post_remainder != 0 and last_sub_row 
                                       else num_post_per_sub_row)
                
                probability = float(num_sub_row_neurons) / float(matrix_size)
                
                # Sample row length;
                length = binom.rvs(remaining_connections, probability)
                
                # Update counters
                remaining_connections -= length
                matrix_size -= num_sub_row_neurons
                
                sub_row_lengths[(i * num_sub_rows) + j] = length
        
    # Insert remaining connections into last row
    sub_row_lengths[-1] = remaining_connections
    return sub_row_lengths
    
def connect(simulation,
            target_area,
            source_area):
    """
    Connect two areas with each other.

    Parameters
    ----------
    simulation : Simulation instance
        Simulation simulating the network containing the two areas.
    target_area : Area instance
        Target area of the projection
    source_area : Area instance
        Source area of the projection
    """
    num_sub_rows = (simulation.params['num_threads_per_spike'] 
                    if simulation.params['procedural_connectivity'] 
                    else 1)

    matrix_type = "PROCEDURAL_PROCEDURALG" if simulation.params['procedural_connectivity']  else "SPARSE_INDIVIDUALG"

    network = simulation.network
    synapses = extract_area_dict(network.synapses,
                                 network.structure,
                                 target_area.name,
                                 source_area.name)
    W = extract_area_dict(network.W,
                          network.structure,
                          target_area.name,
                          source_area.name)
    W_sd = extract_area_dict(network.W_sd,
                             network.structure,
                             target_area.name,
                             source_area.name)

    for target in target_area.populations:
        for source in source_area.populations:
            num_connections = int(synapses[target][source])
            conn_spec = {"total": num_connections}

            syn_weight = {"mean": W[target][source], "sd": W_sd[target][source]}
            exp_curr_params = {}
            
            if target_area == source_area:
                max_delay = simulation.max_inter_area_delay
                if 'E' in source:
                    mean_delay = network.params['delay_params']['delay_e']
                elif 'I' in source:
                    mean_delay = network.params['delay_params']['delay_i']
            else:
                max_delay = simulation.max_intra_area_delay
                v = network.params['delay_params']['interarea_speed']
                s = network.distances[target_area.name][source_area.name]
                mean_delay = s / v
            
            if 'E' in source:
                exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['tau_syn_ex']})
                syn_weight.update({'min': 0., 'max': float(np.finfo(np.float32).max)})
            else:
                exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['tau_syn_in']})
                syn_weight.update({'min': float(-np.finfo(np.float32).max), 'max': 0.})
            
            delay_sd = mean_delay * network.params['delay_params']['delay_rel']

            
            syn_delay = {'min': simulation.params['dt'],
                         'max': max_delay,
                         'mean': mean_delay,
                         'sd': delay_sd}
            syn_spec = {'g': genn_model.init_var(normal_clipped_model, syn_weight),
                        'd': genn_model.init_var(normal_clipped_delay_model, syn_delay)}

            # Add synapse population
            source_genn_pop = source_area.genn_pops[source]
            target_genn_pop = target_area.genn_pops[target]
            syn_pop = simulation.model.add_synapse_population(source_genn_pop.name + "_" + target_genn_pop.name, 
                matrix_type, genn_wrapper.NO_DELAY,
                source_genn_pop, target_genn_pop,
                "StaticPulseDendriticDelay", {}, syn_spec, {}, {},
                "ExpCurr", exp_curr_params, {},
                genn_model.init_connectivity(fixed_num_total_with_replacement_model, conn_spec))

            # Add extra global parameter with row lengths
            #syn_pop.add_connectivity_extra_global_param(
            #    'preCalcRowLength', build_row_lengths(source_genn_pop.size, target_genn_pop.size, num_sub_rows, num_connections))

            # Set max dendritic delay and span type
            syn_pop.pop.set_max_dendritic_delay_timesteps(int(round(max_delay / simulation.params['dt'])))

            if simulation.params['procedural_connectivity']:
                syn_pop.pop.set_span_type(genn_wrapper.SynapseGroup.SpanType_PRESYNAPTIC)
                syn_pop.pop.set_num_threads_per_spike(simulation.params['num_threads_per_spike'])

# -*- coding: utf-8 -*-
"""
@author: rsalisbury
File: V2_intercept.py
Started: March 26, '21
"""



# %% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt


import multiprocessing as mp
#print ("Number of processors: ", mp.cpu_count()) # 8 processors

# %% Import function files
import V4_simfcns_intercept as sim
import V4_graphingfcns_intercept as gr

# %% Run simulation

'''important for file path'''
version_no = '4_int_'
            
def generate_param_list(iterations, coeffs, 
                    phx_var, phq_var, x_var, phlen, max_q, mult_cap, periods, trick_params,
                    last_fold, directory, selected_models, sweep, sweep_variables, sweep_values, 
                    param_list = [], model_list = []):

    if sweep == True:
        for value in sweep_values:
            for sweep_var in sweep_variables:
                coeffs[sweep_var] = value
            
            true_alpha_0, true_idio_alpha, true_ind_beta = coeffs['true_alpha_0'], coeffs['true_idio_alpha'], coeffs['true_ind_beta']
            ent_alpha_0, ent_idio_alpha, ent_ind_beta = coeffs['ent_alpha_0'], coeffs['ent_idio_alpha'], coeffs['ent_ind_beta']
            inc_alpha_0, inc_idio_alpha, inc_ind_beta = coeffs['inc_alpha_0'], coeffs['inc_idio_alpha'], coeffs['inc_ind_beta']
            
            if selected_models['both_correct'] == True:
                # 0. BOTH CORRECT LIST:
                # Both the entrant & the incumbent have the same correct coefficients for demand and the correct structure.
                param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[true_alpha_0,true_idio_alpha,0,true_ind_beta],[true_alpha_0,true_idio_alpha,0,true_ind_beta]],
                              [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                               50,periods, periods,max_q,mult_cap,iterations,
                               trick_params, 0, 0, #trick_params, monop_firm, so_firm
                               'six_per_page', os.path.join(directory, 'cm=both_correct') ,'base_sr_',1])
                model_list.append('both_correct')
            
            if selected_models['both_incorrect'] == True:
                # 1. BOTH INCORRECT LIST:
                # Both the entrant & the incumbent have the same incorrect coefficients for demand AND THE INCORRECT STRUCTURE.
                param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta], [inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                              [[1,0,1,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                               50,periods, periods,max_q,mult_cap,iterations,
                               trick_params, 0, 0, #trick_params, monop_firm, so_firm
                               'six_per_page', os.path.join(directory, 'cm=both_incorrect') ,'base_sr_',1])
                model_list.append('both_incorrect')
            
            if selected_models['sr_base_duop'] == True:
                #BASE DUOP PARAM LIST, SELF-REFLECTIVE:
                param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                          [[1,1,0,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                           50,periods, periods,max_q,mult_cap,iterations,
                           trick_params, 0, 0, #trick_params, monop_firm, so_firm
                           'six_per_page', os.path.join(directory, 'cm=sr_base_duop') ,'base_sr_',1])
                model_list.append('sr_base_duop')
            
            if selected_models['so_base_duop'] == True:
                #BASE DUOP PARAM LIST, SOPHISTICATED:
                #FIRM 1 IS SOPHISTICATED.
                param_list.append([2, 'q','so', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                                  [[1,1,0,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                                  50,periods, periods,max_q,mult_cap,iterations,
                                  trick_params, 1, 0, #trick_params, monop_firm, so_firm
                                  'six_per_page', os.path.join(directory, 'cm=so_base_duop'),'base_so',2])
                model_list.append('so_base_duop')
            
            if selected_models['sr_monop_duop'] == True:
                #MONOP_DUO PARAM LIST, SELF-REFLECTIVE:
                param_list.append([2, 'q','sr', 'monop_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,inc_idio_alpha,0,inc_ind_beta]],
                                  [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                                   50,periods, periods,max_q,mult_cap,iterations,
                                   trick_params, 1, 0, #trick_params, so_firm
                                   'six_per_page', os.path.join(directory, 'cm=sr_monop_duop'),'monop_sr_',3])
                model_list.append('sr_monop_duop')
                
            if selected_models['so_monop_duop'] == True:
                #MONOP_DUOP PARAM LIST, SOPHISTICATED:
                #FIRM 1, THE DUOPOLIST, IS SOPHISTICATED.
                param_list.append([2, 'q','so', 'monop_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                                   [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,inc_idio_alpha,0,inc_ind_beta]],
                                   [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                                   50,periods, periods,max_q,mult_cap,iterations,
                                   trick_params, 1, 0,#for 'so': trick_params, monop_firm, so_firm
                                   'six_per_page', os.path.join(directory, 'cm=so_monop_duop'),'monop_so_',4])
                model_list.append('so_monop_duop')
    
    
    
    else:
        true_alpha_0, true_idio_alpha, true_ind_beta = coeffs['true_alpha_0'], coeffs['true_idio_alpha'], coeffs['true_ind_beta']
        ent_alpha_0, ent_idio_alpha, ent_ind_beta = coeffs['ent_alpha_0'], coeffs['ent_idio_alpha'], coeffs['ent_ind_beta']
        inc_alpha_0, inc_idio_alpha, inc_ind_beta = coeffs['inc_alpha_0'], coeffs['inc_idio_alpha'], coeffs['inc_ind_beta']
        
        if selected_models['both_correct'] == True:
            # 0. BOTH CORRECT LIST:
            # Both the entrant & the incumbent have the same correct coefficients for demand and the correct structure.
            param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[true_alpha_0,true_idio_alpha,0,true_ind_beta],[true_alpha_0,true_idio_alpha,0,true_ind_beta]],
                          [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                           50,periods, periods,max_q,mult_cap,iterations,
                           trick_params, 0, 0, #trick_params, monop_firm, so_firm
                           'six_per_page', os.path.join(directory, 'cm=both_correct') ,'base_sr_',1])
            model_list.append('both_correct')
        
        if selected_models['both_incorrect'] == True:
            # 1. BOTH INCORRECT LIST:
            # Both the entrant & the incumbent have the same incorrect coefficients for demand AND THE INCORRECT STRUCTURE.
            param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta], [inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                          [[1,0,1,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                           50,periods, periods,max_q,mult_cap,iterations,
                           trick_params, 0, 0, #trick_params, monop_firm, so_firm
                           'six_per_page', os.path.join(directory, 'cm=both_incorrect') ,'base_sr_',1])
            model_list.append('both_incorrect')
        
        if selected_models['sr_base_duop'] == True:
            #BASE DUOP PARAM LIST, SELF-REFLECTIVE:
            param_list.append([2, 'q','sr', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                      [[1,1,0,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                       50,periods, periods,max_q,mult_cap,iterations,
                       trick_params, 0, 0, #trick_params, monop_firm, so_firm
                       'six_per_page', os.path.join(directory, 'cm=sr_base_duop') ,'base_sr_',1])
            model_list.append('sr_base_duop')
        
        if selected_models['so_base_duop'] == True:
            #BASE DUOP PARAM LIST, SOPHISTICATED:
            #FIRM 1 IS SOPHISTICATED.
            param_list.append([2, 'q','so', 'base_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,0,inc_idio_alpha,inc_ind_beta]],
                              [[1,1,0,1],[1,0,1,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                              50,periods, periods,max_q,mult_cap,iterations,
                              trick_params, 1, 0, #trick_params, monop_firm, so_firm
                              'six_per_page', os.path.join(directory, 'cm=so_base_duop'),'base_so',2])
            model_list.append('so_base_duop')
        
        if selected_models['sr_monop_duop'] == True:
            #MONOP_DUO PARAM LIST, SELF-REFLECTIVE:
            param_list.append([2, 'q','sr', 'monop_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,inc_idio_alpha,0,inc_ind_beta]],
                              [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                               50,periods, periods,max_q,mult_cap,iterations,
                               trick_params, 1, 0, #trick_params, so_firm
                               'six_per_page', os.path.join(directory, 'cm=sr_monop_duop'),'monop_sr_',3])
            model_list.append('sr_monop_duop')
        
        if selected_models['so_monop_duop'] == True:
            #MONOP_DUOP PARAM LIST, SOPHISTICATED:
            #FIRM 1, THE DUOPOLIST, IS SOPHISTICATED.
            param_list.append([2, 'q','so', 'monop_duop',[true_alpha_0,true_idio_alpha,0,true_ind_beta],
                               [[ent_alpha_0,ent_idio_alpha,0,ent_ind_beta],[inc_alpha_0,inc_idio_alpha,0,inc_ind_beta]],
                               [[1,1,0,1],[1,1,0,1]],phlen,[1,phx_var,1,x_var,2,phq_var],
                               50,periods, periods,max_q,mult_cap,iterations,
                               trick_params, 1, 0,#for 'so': trick_params, monop_firm, so_firm
                               'six_per_page', os.path.join(directory, 'cm=so_monop_duop'),'monop_so_',4])
            model_list.append('so_monop_duop')
            
    return param_list, model_list


def run_through_parameters(param_list, model_list, selected_models, directory, version_no, last_fold,
                           sweep, sweep_variables, sweep_values, rerun_sim = False):
    
    csv_name = {}
    dir_list = {}
    initial_weights = {}
    
    param_index_num = 0
    for param in param_list:
        
        #Create a file to record results of the run.
        if not os.path.exists(directory):
            os.makedirs(directory)
        output_file = open(os.path.join(directory, 'output_file.txt'), 'w+')
        '''WORKING IN HERE '''
    
        ### choose an index parameter to identify each parameterization. ###
        # When there is no sweep, identify by the model:
            
        parameters_dict = {'true_alpha_0': param[4][0], 'true_idio_alpha': param[4][1], 'true_ind_beta': param[4][3],
                           'ent_alpha_0': param[5][0][0], 'ent_idio_alpha': param[5][0][1], 'ent_ind_beta': param[5][0][3],
                           'inc_alpha_0': param[5][1][0], 'inc_idio_alpha': param[5][1][1], 'inc_ind_beta': param[5][1][3],
                           'phlen': param[7], 'phx_var': param[8][1], 'x_var': param[8][3], 'phq_var': param[8][5],
                           'periods': param[10], 'max_q': param[11], 'mult_cap': param[12], 'iterations': param[13]}    
        
        model_index = model_list[param_index_num]
        if sweep == False:
            var_index = parameters_dict['true_idio_alpha']
            file_id = model_index + '_nosweep_' + str(var_index)
        else: #indexing the csv's when there is a sweep.
            var_index = parameters_dict[sweep_variables[0]]
            file_id = model_index + '_' + sweep_variables[0] + '_' + str(var_index)
        
        directory = param[18]
        filebegin = param[19]
        sim_csv_name = str(filebegin) + '_' +  file_id + '_sim_data.csv'
        
        if not os.path.exists(os.path.join(directory, sim_csv_name)) or rerun_sim == True:
            csv_name[var_index, model_index], initial_weights[var_index, model_index] = sim.run_simulation(*param, file_id)
        else:
            print('This csv exists: ' + str(sim_csv_name))
            
        gr.graph_sns(csv_name[var_index, model_index], initial_weights[var_index, model_index], *param)
        
        dir_list[model_index] = [os.path.join(os.getcwd(), 'MY_v' + str(version_no),
                last_fold, str([true_alpha_0,true_idio_alpha,0,true_ind_beta]), 'cm='+ model_index), [*param]]
        
        output_file.write('This param is finished running.')
        output_file.write('-' * 20)
        param_index_num += 1
        
    output_file.close()
    return csv_name, dir_list, initial_weights

################################################################################
###                          End of functions                                ###
################################################################################


############################## INTERCEPT MODEL #################################

# %% Entering parameters.

iterations = 20

### ENTER DEMAND FCN COEFFICIENTS ###

#True demand parameters
true_alpha_0 = 105 #standard alpha = 105
true_idio_alpha = -15 #standard idio_alpha = -5
true_ind_beta = -15

#the entrant's demand parameters (structurally correct)
ent_alpha_0 = 105
ent_idio_alpha = -15
ent_ind_beta = -15

#the monopolist's/incumbent's demand parameters (structurally incorrect)
inc_alpha_0 = 105
inc_idio_alpha = -5
inc_ind_beta = -15 #standard ind_beta = -15

coeffs = {'true_alpha_0': true_alpha_0, 'true_idio_alpha': true_idio_alpha, 'true_ind_beta': true_ind_beta,
          'ent_alpha_0': ent_alpha_0, 'ent_idio_alpha': ent_idio_alpha, 'ent_ind_beta': ent_ind_beta,
          'inc_alpha_0': inc_alpha_0, 'inc_idio_alpha': inc_idio_alpha, 'inc_ind_beta': inc_ind_beta}

#Other model parameters.
phx_var = 0.5 #SOMETIMES CHANGED TO REFLECT EARLIER PAPER. IN THIS VERSION PHX_VAR = 0.5. IN OLD VERSION, PHX_VAR = 0.3
phq_var = 0.5
x_var = 0.3
phlen = 20 #standard phlen = 20
max_q = 50 #SOMETIMES CHANGED TO REFLECT EARLIER PAPER. IN THIS VERSION MAX_Q = 50. IN OLD VERSION, MAX_Q = 5
mult_cap = 1 #multiplies the monopolist's capacity by this factor, so its capacity is max_q*mult_cap; 1 when no multiplier
periods = 100

#Trick parameters (currently not used in this model).
trick_type = 'none'
ta = 0
tamount_space = 0
tperiod_space = []
trick_len = 0
trick_freq = 0
trick_params = [trick_type, ta, tamount_space, tperiod_space, trick_len, trick_freq]


# %% Enter what types of models you want to run.

last_fold = 'V3_P1_' + str(iterations) + 'it_simplifying_1'
directory = os.path.join(os.getcwd(), 'MY_v' + str(version_no),
            last_fold,str([true_alpha_0,true_idio_alpha,0,true_ind_beta]))

selected_models = {'both_correct': True, 'both_incorrect': False,
        'sr_base_duop': True, 'so_base_duop': False, 
        'sr_monop_duop': False, 'so_monop_duop': False}

#####
# Summary of sweeps types that can be selected:
# 1. idiosyncratic alpha sweep. - this sweeps all 
# 2. incorrect idiosyncratic alpha sweep.
#####


sweep = True #possible values: True or False
sweep_variables = ['true_idio_alpha', 'ent_idio_alpha'] #note: if multiple variables are listed, they will be swept over together.
sweep_values = (-15, -10, -5)


# %% Creating the parameter list

param_list, model_list = generate_param_list(iterations, coeffs, 
                    phx_var, phq_var, x_var, phlen, max_q, mult_cap, periods, trick_params,
                    last_fold, directory, selected_models, sweep, sweep_variables, sweep_values)

# %% Running the simulation on the parameter list to create csv's.


csv_name, dir_list, initial_weights = run_through_parameters(param_list, model_list, selected_models, directory, version_no, last_fold,
                   sweep, sweep_variables, sweep_values, rerun_sim = False) #rerun_sim = True if you want sim csv files to be generated even if some with the same name.


df_dict = {}



### SWEEP GRAPHS!
if sweep == True:
    param_num = 0
    for param in param_list:
        parameters_dict = {'true_alpha_0': param[4][0], 'true_idio_alpha': param[4][1], 'true_ind_beta': param[4][3],
                           'ent_alpha_0': param[5][0][0], 'ent_idio_alpha': param[5][0][1], 'ent_ind_beta': param[5][0][3],
                           'inc_alpha_0': param[5][1][0], 'inc_idio_alpha': param[5][1][1], 'inc_ind_beta': param[5][1][3],
                           'phlen': param[7], 'phx_var': param[8][1], 'x_var': param[8][3], 'phq_var': param[8][5],
                           'periods': param[10], 'max_q': param[11], 'mult_cap': param[12], 'iterations': param[13]}   
        model_index = model_list[param_num]
        var_index = parameters_dict[sweep_variables[0]]
        
        name = csv_name[var_index, model_index]
        df_dict[var_index, model_index] = pd.read_csv(os.path.join(dir_list[model_index][0], name))
        param_num+=1
        
    ### WIP: this graph_sns_overlap fcn has not been adapted to the new dictionary system for determining the model or the nature of the sweep.###
    gr.graph_sns_overlap(df_dict, sweep_values, model_list, directory, dir_list, initial_weights, sweep_variables)

    del(df_dict)
    
    plt.close()
    plt.cla()
    plt.clf()

    print("Done graphing sweep over the incorrect firm's idiosync beta")    



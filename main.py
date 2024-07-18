#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:51:14 2024

@author: vitor
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plot.mplstyle')
import time
import torch
from tqdm import tqdm, trange
from models import MLP, HNN, DHNN
from scipy.integrate import solve_ivp
from structural_model import MultibodyBeam
from train import train, loss_plot
from utils import animate, pre_process, ObjectView

################################## USER INPUTS #################################
beam_dict = {'N': 10, # number of elements
             'L': 1, # length of the beam [m]
             'w': 0.1, # width of the beam [m] 
             'm0': 0.2, # linear mass density [kg/m] 
             'I0': 0.1, # mass moment of inertia per element [kg*m^2]
             'EIyy': 50, # out-of-plane flexural rigidity [N*m^2]
             'EIzz': 1e6, # in-plane flexural rigidity [N*m^2]
             'GJ': 50, # torsion rigidity [N*m^2]
             'c': 0.0, # damping ratio
             'y_cg': 0., # center of gravity cg position relative to reference line [m]
             'psi': 0, # nominal deflection angle [deg] around z-axis (inertial frame)
             'theta': 0, # nominal deflection angle [deg] around y-axis (inertial frame)
             'phi': 0, # nominal deflection angle [deg] around x-axis (inertial frame)
             'bending': True, # consider out-of-plane bending deformation mode
             'torsion': True, # consider torsion deformation mode
             'in_bending': False} # consider in-plane banding deformation mode
data_dict = {'dataset': 'nn_dataset.npy', # opts: 'create' or data file
             'ref_model': 'ROM', # reference model: 'FOM' or 'ROM'
             'r': 3, # number of modes (only used for ROM 'ref_model')
             'n_samples': 50, # number of samples
             'train_split': 0.8, # fraction of samples used for training
             'data_scaler': 'normalize', # scale dataset; opts: 'normalize', 'standardize' or None
             'noise': 0., # include noise into data
             'q_span': (-5, 5), # limits of q [deg]
             'p_span': (0, 0), # limits of p
             't_span': (0, 1), # simulation time span     
             't_step': 0.001, # simulation timestep
             'int_method': 'Radau'} # integration scheme to be used by the solve_ivp() function
nn_dict  =  {'train_nn': False, # if the nn models have to be trained
             'hidden_dim': [256], # number of neurons of each hidden layer
             'activation': 'elu', # activation function of each layer
             'learning_rate': 0.0001, # initial learning rate for the Adam optimizer
             'epochs': 20000, # number of training epochs 
             'batch_size': 64, # number of samples per training step
             'loss': 'L2', # loss function; opts: 'L2' or 'mse'
             'device': 'cpu'} # 'cpu' or 'cuda' (if GPUs available)
sim_dict = { 'model': ['FOM','ROM'], # models to be simulated
             'r': 2, # number of modes (only used for ROM model)
             'g': False, # presence of gravitational force
             'fe': True, # presence of external forces (edit in external_force.py)
             'x_fe': 'tip', # x-axis position (in global frame) of application of external forces relative to the element. opts: 'cm' if at center of mass or 'tip' if at the tip
             'y_fe': 0., # y-axis position [m] (in global frame) of application of external forces relative to the reference line of the element.
             'ic': 'null', # initial conditions, opts: 'random', 'equilibrium' or 'null'; for 'random' it is necessary to specify 'q_min', 'q_max', 'p_min', and 'p_max' in the function ic_fun() 
             'seed': None, # random seed to be used in case of 'random' ic
             'F_span': (0, 0, -10, 0, 0, 10), # tip force span [N] (only used if 'random' ic)
             't_span': (0, 2), # simulation time span     
             't_step': 0.002, # simulation timestep
             'int_method': 'Radau', # integration scheme to be used by the solve_ivp() function
             'animation': True} # create animation .gif file from the simulation solution
################################################################################
beam = MultibodyBeam(beam_dict, sim_dict) # initialize the object "beam"

#%% DATASET MANAGEMENT
np.random.seed(sim_dict['seed'])
match data_dict['dataset']:
    case 'create':
        q_data = np.deg2rad(data_dict['q_span'][0] + np.diff(data_dict['q_span'])*np.random.rand(data_dict['n_samples'], 2*beam.N))
        p_data = (data_dict['p_span'][0] + np.diff(data_dict['p_span'])*np.random.rand(data_dict['n_samples'], 2*beam.N))
        
        if data_dict['ref_model'] == 'ROM':
            beam.reduced_model()
            q_data, p_data = (beam.Phi_r.T @ q_data.T).T, (beam.Phi_r.T @ p_data.T).T 
        x0_data = np.hstack((q_data, p_data))
        #xdot_data = np.zeros((nn_dict['n_samples'], 2*beam.N))
        xdot_data = np.empty((0, x0_data.shape[1]))
        x_data = np.empty((0, x0_data.shape[1]))
        t_eval = np.linspace(data_dict['t_span'][0], data_dict['t_span'][1], int(np.diff(data_dict['t_span'])/data_dict['t_step']))
        for i in trange(data_dict['n_samples'], position=0, desc='Create dataset'):
            x = solve_ivp(beam.dynamics, data_dict['t_span'], x0_data[i], # integrate the reference structural model
                                        method=data_dict['int_method'], t_eval=t_eval,
                                        args=(data_dict['ref_model'],)) 
            x_data = np.append(x_data, x['y'].T, axis=0)
            xdot = np.zeros(x['y'].T.shape)
            for k in range(len(x['t'])):
                xdot[k] = beam.dynamics(x['t'][k], x['y'].T[k], data_dict['ref_model'])
            xdot_data = np.append(xdot_data, xdot, axis=0)
            # xdot_data[i] = beam.dynamics(0, x_data[i])
        data_raw = {'x': x_data, 
                    'dx': xdot_data} 
        data, x_scaler, dx_scaler = pre_process(data_raw, data_dict)
        np.save('data/nn_dataset', data_raw)
    case None:
        pass
    case _:
        data_raw = np.load('data/'+data_dict['dataset'], allow_pickle=True).item()
        data, x_scaler, dx_scaler = pre_process(data_raw, data_dict)
        

#%% TRAINING NEURAL NETWORKS
nn_dict['input_dim'] = 2*beam.N if data_dict['ref_model'] == 'FOM' else 2*data_dict['r']
nn_dict['output_dim'] = 2*beam.N if data_dict['ref_model'] == 'FOM' else 2*data_dict['r']
model, losses, t_train = {}, {}, {}
for m in [x for i,x in enumerate(sim_dict['model']) if x!='FOM' and x!='ROM']:
    if nn_dict['train_nn']:
        model[m] = globals()[m](ObjectView(nn_dict))
        t = time.time()
        losses[m] = train(model[m], data, ObjectView(nn_dict))
        t_train[m] = time.time() - t
        model[m].cpu()
        torch.save(model[m].state_dict(), m+'_trained')
    else:
        model[m] = globals()[m](ObjectView(nn_dict))
        model[m].load_state_dict(torch.load(m+'_trained'))
        model[m].eval()
        model[m].cpu()
loss_plot(losses) if nn_dict['train_nn'] else None


#%% SIMULATION
q0, p0 = beam.ic_fun(sim_dict['F_span']) # initial conditions
# q0, p0 = beam.ic_fun(data_dict['q_span'], data_dict['p_span']) # initial conditions
t_eval = np.linspace(sim_dict['t_span'][0], sim_dict['t_span'][1], # time vector
                     int(np.diff(sim_dict['t_span'])/sim_dict['t_step']))
args, x, q, p, r, rcm, rv, rm, E = {}, {}, {}, {}, {}, {}, {}, {}, {}
for m in sim_dict['model']: # time integration of the models
    pbar = tqdm(total = len(t_eval), position=0, desc='{} simulation'.format(m))
    args['pbar'], args['state'] = pbar, [t_eval[0], sim_dict['t_step']]
    if m == 'FOM':
        x[m] = solve_ivp(beam.dynamics, sim_dict['t_span'], np.concatenate((q0, p0)), 
                         method=sim_dict['int_method'], t_eval = t_eval, 
                          args=(m, args))
        q[m], p[m] = np.split(x[m]['y'], 2)
    elif m == 'ROM':
        beam.reduced_model()
        eta0, pr0 = beam.Phi_r.T @ q0, beam.Phi_r.T @ p0 
        x[m] = solve_ivp(beam.dynamics, sim_dict['t_span'], np.concatenate((eta0, pr0)), 
                         method=sim_dict['int_method'], t_eval=t_eval, 
                         args=(m, args))
        q[m], p[m] = np.split(x[m]['y'], 2)[0], np.split(x[m]['y'], 2)[1]
    else:
        q0_nn, p0_nn = (beam.Phi_r.T @ q0, beam.Phi_r.T @ p0) if data_dict['ref_model'] == 'ROM' else (q0, p0)  
        args['x_scaler'], args['dx_scaler'] = x_scaler, dx_scaler
        args['ref_model'] = data_dict['ref_model']
        x[m] = solve_ivp(beam.dynamics, sim_dict['t_span'], np.concatenate((q0_nn, p0_nn)), 
                         method=sim_dict['int_method'], t_eval=t_eval, 
                         args=(model[m], args)) 
        q[m], p[m] = np.split(x[m]['y'], 2)
    pbar.close()
    
    E[m] = beam.total_energy(x[m]['t'], np.concatenate((q[m], p[m])), m, data_dict['ref_model']) # total energy of the system
    
    r[m] = np.zeros((len(x[m]['t']), beam.N+1, 3))
    rcm[m] = np.zeros((len(x[m]['t']), beam.N, 3))
    rv_r = np.zeros((len(x[m]['t']), beam.N+1, 3))
    rv_l = np.zeros((len(x[m]['t']), beam.N+1, 3))
    rv_r[:,0,1], rv_l[:,0,1] = -beam.w/2, beam.w/2 
    for i in range(len(x[m]['t'])):
        r[m][i,1:] = beam.position(beam.Phi_r@q[m][:,i] if m == 'ROM' or (m != 'FOM' and data_dict['ref_model'] == 'ROM') else q[m][:,i], r_0=beam.r1_0).reshape(-1,3,order='F') # tip position of each element
        rcm[m][i,:] = beam.position(beam.Phi_r@q[m][:,i] if m == 'ROM' or (m != 'FOM' and data_dict['ref_model'] == 'ROM') else q[m][:,i], r_0=beam.r1_0).reshape(-1,3,order='F') # center of mass position of each element
        rv_r[i,1:] = beam.position(beam.Phi_r@q[m][:,i] if m == 'ROM' or (m != 'FOM' and data_dict['ref_model'] == 'ROM') else q[m][:,i], r_0=beam.rvr_0).reshape(-1,3,order='F') # right side vertices of each element
        rv_l[i,1:] = beam.position(beam.Phi_r@q[m][:,i] if m == 'ROM' or (m != 'FOM' and data_dict['ref_model'] == 'ROM') else q[m][:,i], r_0=beam.rvl_0).reshape(-1,3,order='F') # left side vertices of each element    
    rm[m] = np.concatenate((rv_r,rv_l), axis=2).reshape((rv_r.shape[0],2*(beam.N+1),3)) # boundary vertices of each element
    rv[m] = np.concatenate((rv_r, np.flip(rv_l,axis=1)), axis=1) # boundary vertices of the beam
   
    plt.figure(1)
    plt.plot(x[m]['t'], r[m][:,-1,0], label=m)
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.xlim(sim_dict['t_span'])
    plt.grid(linestyle='--')   
    plt.legend(loc='upper right', framealpha=.9)
    
    plt.figure(2)
    plt.plot(x[m]['t'], r[m][:,-1,1], label=m)
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.grid(linestyle='--')
    plt.xlim(sim_dict['t_span'])
    
    plt.figure(3)  
    plt.plot(x[m]['t'], r[m][:,-1,2], label=m)
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.grid(linestyle='--')
    plt.xlim(sim_dict['t_span'])   
    
    plt.figure(4)
    q_n = (beam.Phi_r@q[m] if m == 'ROM' else q[m].copy()) if beam.torsion else 0*q[m].copy()
    plt.plot(x[m]['t'], np.rad2deg(np.sum(q_n[beam.N:,:], axis=0)), label=m)
    plt.xlabel('t [s]')
    plt.ylabel('$\gamma$ [deg]')
    plt.grid(linestyle='--')
    plt.xlim(sim_dict['t_span'])   
    
    plt.figure(5, figsize=(16,6))
    plt.plot(x[m]['t'], E[m], label=m)
    plt.xlabel('t [s]')
    plt.ylabel('E$_\mathrm{total}$ [J]')
    plt.grid(linestyle='--')
    plt.legend(loc='upper right', framealpha=.9)
    plt.savefig('images/E.png', dpi=150, format='png', bbox_inches='tight', transparent=True)
plt.show()
animate(beam, x[m]['t'], q, (r, rcm, rv, rm), show_surf=True, show_mesh=True) if sim_dict['animation'] else None


# t0 = time.time()
# print('Simulation time: {:.3f}s'.format(time.time() - t0))


#%% Equilibrium solution:
# from scipy.optimize import fsolve
# q_eq = fsolve(beam.equilibrium, np.zeros(beam.N), args=('ROM')) # find the equilibrium solution
# r_eq = np.zeros((beam.N+1, 3))
# r_eq[1:] = beam.position(beam.Phi_r@q_eq, r_0 = beam.r1_0).reshape(-1, 3, order='F')
# print('Tip position at equilibrium condition:\nx: {:.3f}m\ny: {:.3f}m\nz: {:.3f}m'.format(r_eq[-1,0],r_eq[-1,1],r_eq[-1,2]))
# plt.plot(r_eq[:,0], r_eq[:,2],'o-',lw=5, markersize=10)
# plt.title('Equilibrium Solution')
# plt.xlabel('x [m]')
# plt.ylabel('z [m]')
# plt.xlim((0, 1.05*beam.L))
# plt.ylim((-beam.L, beam.L))
# plt.grid(linestyle='--')
# plt.show()


#%% Reference Models Validation
# val_data = np.load('data/validation_data.npy', allow_pickle=True).item()
# plt.figure(1)
# plt.plot(val_data['10sin20t']['t'], val_data['10sin20t']['x'], '--k', label='Reference')
# plt.xlabel('t [s]')
# plt.ylabel('x [m]')
# plt.xlim(sim_dict['t_span'])
# plt.grid(linestyle='--')
# plt.legend(loc='upper right', framealpha=0.9)
# plt.savefig('images/validation1_xfom.png', transparent=True, dpi=150, format='png')

# plt.figure(3)
# ax = plt.gca()
# axins = ax.inset_axes([0.51, 0.09, 0.45, 0.45], xlim=(0.6, 0.8), ylim=(0.045, 0.075))
# ax.plot(val_data['10sin20t']['t'], val_data['10sin20t']['z'], '--k', label='Reference')
# axins.grid(linestyle='--')
# axins.patch.set_alpha(0.9)
# plt.setp(axins.get_xticklabels(), backgroundcolor="white")
# plt.setp(axins.get_yticklabels(), backgroundcolor="white")
# axins.set_yticks([], minor=True)
# axins.tick_params(labelsize=18)
# plt.xlabel('t [s]')
# plt.ylabel('z [m]')
# plt.grid(linestyle='--')
# plt.xlim(sim_dict['t_span'])
# axins = ax.inset_axes([0.51, 0.09, 0.45, 0.45], xlim=(0.6, 0.8), ylim=(0.045, 0.075))
# axins.grid(linestyle='--')
# axins.patch.set_alpha(0.9)
# plt.setp(axins.get_xticklabels(), backgroundcolor="white")
# plt.setp(axins.get_yticklabels(), backgroundcolor="white")
# axins.set_yticks([], minor=True)
# axins.tick_params(labelsize=18)
# axins.plot(val_data['10sin20t']['t'], val_data['10sin20t']['z'], '--k')
# ax.indicate_inset_zoom(axins, edgecolor='black', alpha=1, lw=0.8) 
# plt.savefig('images/validation1_zfom.png',transparent=True, dpi=150, format='png')

#%%
# Static Solution - Tip Force:
# plt.figure()
# plt.plot(val_data['F_static'], val_data['tip_static'][:,0], 'o', markerfacecolor='none', markeredgewidth=2, markersize=15, label='FOM')
# # plt.plot(val_data['F_static'], val_data['tip_static_rom'][:,0], 'x', markeredgewidth=3, markersize=10, label='ROM')
# plt.plot(val_data['F_static'], val_data['tip_static_ref'][:,0], '--k', markersize=15, label='Reference')
# plt.xlabel('$\mathrm{F_z}$ [N]')
# plt.ylabel('x [m]')
# plt.grid(linestyle='--')
# plt.legend(loc='upper right', framealpha=0.9)
# plt.savefig('images/validation_static_xfom.png',transparent=True, dpi=150, format='png')

# plt.figure()
# plt.plot(val_data['F_static'], val_data['tip_static'][:,2], 'o', markerfacecolor='none', markeredgewidth=2, markersize=15, label='FOM')
# # plt.plot(val_data['F_static'], val_data['tip_static_rom'][:,2], 'x', markeredgewidth=3, markersize=10, label='ROM')
# plt.plot(val_data['F_static'], val_data['tip_static_ref'][:,2], '--k', markersize=15, label='Reference')
# plt.xlabel('$\mathrm{F_z}$ [N]')
# plt.ylabel('z [m]')
# plt.grid(linestyle='--')
# plt.savefig('images/validation_static_zfom.png',transparent=True, dpi=150, format='png')

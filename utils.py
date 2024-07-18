#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:44:42 2024

Description: 
@author: vitor
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.style.use('plot.mplstyle')


class ObjectView(object): # build "struct" variables from dictionary variables
    def __init__(self, d): 
        self.__dict__ = d
        
def DCM(n, mu_rad):                                                        
    if isinstance(n, int):
        match n:
            case 1:
                C = np.array([[       1.,                0.,               0.      ],
                              [       0.,          np.cos(mu_rad),   np.sin(mu_rad)],
                              [       0.,         -np.sin(mu_rad),   np.cos(mu_rad)]])
            case 2:
                C = np.array([[ np.cos(mu_rad),         0.,         -np.sin(mu_rad)],
                              [       0.,               1.,                0.      ],
                              [ np.sin(mu_rad),         0.,          np.cos(mu_rad)]])
            case 3:
                C = np.array([[ np.cos(mu_rad),   np.sin(mu_rad),         0.       ],
                              [-np.sin(mu_rad),   np.cos(mu_rad),         0.       ],
                              [       0.,               0.,               1.       ]])
    else:
        n = n/np.linalg.norm(n)
        C = (1 - np.cos(mu_rad))*(n@n.T) + np.cos(mu_rad)*np.eye(3) - np.sin(mu_rad)*skew(n);    
    return C

def skew(v):
    v_tilde = np.array([[   0.,    -v[2,0],    v[1,0]],
                        [ v[2,0],     0.,     -v[0,0]],
                        [-v[1,0],   v[0,0],      0. ]])
    return v_tilde

def complex_step(f, v):
    n_col = len(v)
    n_lin = len(f(v))
    delta = 1e-100
    A = np.zeros([n_lin, n_col])
    Delta = v + 1j*delta*np.eye(n_col)
    for i in range(0, n_col):
        A[:,i] = f(Delta[i]).imag/delta
    return A 

def animate(beam, t_vec, q, r, show_surf=False, show_mesh=False):    
    fig = plt.figure()  
    fig.set_tight_layout(True) 
    ax = fig.add_subplot(projection='3d')
    ax.tick_params(labelsize=15)
    ax.azim = 225
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo['grid']['linestyle'] = ':'
    ax.yaxis._axinfo['grid']['linestyle'] = ':'
    ax.zaxis._axinfo['grid']['linestyle'] = ':'
    ax.set(xlim3d=(0, 1.1*beam.L), ylim3d=(-beam.L, beam.L), zlim3d=(-beam.L, beam.L))
    ax.set_xlabel('x [m]', fontsize=18, labelpad=10)
    ax.set_ylabel('y [m]', fontsize=18, labelpad=10)
    ax.set_zlabel('z [m]', fontsize=18, labelpad=10)
    text = ax.text2D(0, 0, '', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    line_ea, point_cm, vert, mesh = {}, {}, {}, {}
    color = plt.cm.tab10(np.linspace(0,1,10))
    pbar = tqdm(total=len(t_vec), position=0, desc='Load animation')
    for j, m in enumerate(r[0]):
        q_n = (beam.Phi_r@q[m] if m == 'ROM' else q[m].copy()) if beam.torsion else 0*q[m].copy()
        line_ea[m], = ax.plot3D([], [], [], label=m if not show_surf else '_nolegend_')
        #point_cm[m] = ax.scatter([], [], [], marker='o')
        if show_surf:
            vert[m] = ax.add_collection3d(Poly3DCollection([], lw=1, edgecolor='k', facecolor=color[j], alpha=.3, label=m))
            vert[m]._facecolors2d = vert[m]._facecolor3d # workaround for an error of Poly3DCollection that doesn't let it show legend entries 
            vert[m]._edgecolors2d = vert[m]._edgecolor3d # workaround for an error of Poly3DCollection that doesn't let it show legend entries 
        if show_mesh:
            k, kk = 0, 0
            while k <= len(r[3][m][0]):
                mesh[m+'{:.0f}'.format(kk)], = ax.plot3D([], [], [], 'k', lw=.5)
                k, kk = k+2, kk+1
        plt.legend(loc='upper left', fontsize=15)  
    def frame_update(i, pbar):
        pbar.update() if i < len(t_vec)-1 else None
        for m in r[0]:
            line_ea[m].set_data_3d(r[0][m][i,:,0], r[0][m][i,:,1], r[0][m][i,:,2])  
            #point_cm[m]._offsets3d = (np.ma.ravel(r[1][m][i,:,0]), np.ma.ravel(r[1][m][i,:,1]), np.ma.ravel(r[1][m][i,:,2]))
            if show_surf:
                vert[m].set_verts([list(zip(r[2][m][i,:,0], r[2][m][i,:,1], r[2][m][i,:,2]))])
            if show_mesh:
                k, kk = 0, 0
                while k <= len(r[3][m][0]):
                    mesh[m+'{:.0f}'.format(kk)].set_data_3d(r[3][m][i,k:k+2,0], r[3][m][i,k:k+2,1], r[3][m][i,k:k+2,2])
                    k, kk = k+2, kk+1
        text.set_text('t = {:.3f}s\n\ntip x: {:.3f}m\ntip y: {:.3f}m\ntip z: {:.3f}m\ntip $\gamma$: {:.3f}$^\circ$'.format(t_vec[i], r[0][m][i,-1,0], r[0][m][i,-1,1], r[0][m][i,-1,2], np.rad2deg(np.sum(q_n[beam.N:,i], axis=0))))    
        text.set_position((0.8, 0.99))
        return text, line_ea, vert#, point_cm     
    plt.close()
    anim = FuncAnimation(fig, partial(frame_update, pbar=pbar), frames=len(t_vec)) 
    anim.save('BeamSimulation.gif', fps=30, dpi=150) 
    pbar.close()
    return
  
def pre_process(data_raw, nn_dict):     
    data = {}
    match nn_dict['data_scaler']: # normalizes/standardizes dataset
        case 'normalize':
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            dx_scaler = MinMaxScaler(feature_range=(0, 1))
            data_raw['x'] = x_scaler.fit_transform(data_raw['x'])
            data_raw['dx'] = dx_scaler.fit_transform(data_raw['dx'])
        case 'standardize':
            x_scaler = StandardScaler()
            dx_scaler = StandardScaler()
            data_raw['x'] = x_scaler.fit_transform(data_raw['x'])
            data_raw['dx'] = dx_scaler.fit_transform(data_raw['dx'])
        case _:
            x_scaler, dx_scaler = [], []
            
    n_train = round(nn_dict['n_samples']*nn_dict['train_split']) # train/validation split:
    split_index = int(n_train*len(data_raw['x'])/nn_dict['n_samples'])
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k + '_train'], split_data[k + '_val'] = data_raw[k][:split_index], data_raw[k][split_index:]
    data = split_data
    
    for i in range(data['x_train'].shape[1]): # add noise to the samples
        data['x_train'][:,i] += np.random.randn(*data['x_train'][:,i].shape)*nn_dict['noise']
        data['x_val'][:,i] += np.random.randn(*data['x_val'][:,i].shape)*nn_dict['noise']
    return data, x_scaler, dx_scaler

def update_pbar(t, args):
    last_t, dt = args['state']
    n = int((t - last_t)/dt)
    args['pbar'].update(n)
    args['state'][0] = last_t + dt*n
    return
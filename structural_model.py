#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:37:08 2024

@author: vitor
"""

import numpy as np
import torch
from sympy import symbols, Function, dsolve, lambdify
from scipy.optimize import fsolve
from scipy.linalg import block_diag
from utils import DCM, complex_step, update_pbar 
from external_force import external_force


class MultibodyBeam:
    def __init__(self, beam_properties, sim_properties):
        self.N = beam_properties['N']
        self.L = beam_properties['L']
        self.w = beam_properties['w']
        self.m0 = beam_properties['m0']
        self.I0 = beam_properties['I0']
        self.EIyy = beam_properties['EIyy']
        self.EIzz = beam_properties['EIzz']
        self.GJ = beam_properties['GJ']
        self.c = beam_properties['c']
        self.y_cg = beam_properties['y_cg']
        self.psi = beam_properties['psi']
        self.theta = beam_properties['theta']
        self.phi = beam_properties['phi']
        self.l = self.L/self.N
        self.bending = beam_properties['bending']
        self.in_bending = beam_properties['in_bending']
        self.torsion = beam_properties['torsion']
        self.init_beam()

        self.g = sim_properties['g']
        self.fe = sim_properties['fe']
        self.x_fe = sim_properties['x_fe']
        self.y_fe = sim_properties['y_fe']
        self.r = sim_properties['r']
        self.ic = sim_properties['ic']
        self.seed = sim_properties['seed'] 
        self.r_fe = self.r1_0.copy() if self.x_fe == 'tip' else self.rcm_0.copy()
        self.r_fe[self.N:2*self.N] += self.y_fe
        self.mass_matrix()
        self.stiffness_matrix()
        self.damping_matrix()
        self.g_force()

    def init_beam(self):
        r0_l, rcm_l, r1_l = np.zeros([self.N, 3]), np.zeros([self.N, 3]), np.zeros([self.N, 3])
        C_l0 = DCM(1, np.deg2rad(self.psi))@DCM(2, np.deg2rad(self.theta))@DCM(3, np.deg2rad(self.phi))
        r0_l[:,0] = np.linspace(0, self.L, self.N+1)[:-1]
        rcm_l[:,0] = r0_l[:,0] + 0.5*self.l
        rcm_l[:,1] = self.y_cg
        r1_l[:,0] = rcm_l[:,0] + 0.5*self.l
        self.rcm_0 = (C_l0.T@rcm_l.T).reshape(-1)
        self.r0_0 = (C_l0.T@r0_l.T).reshape(-1)
        self.r1_0 = (C_l0.T@r1_l.T).reshape(-1)
        self.rvr_0 = self.r1_0.copy()
        self.rvr_0[self.N:2*self.N] -= self.w/2
        self.rvl_0 = self.rvr_0.copy()
        self.rvl_0[self.N:2*self.N] *= -1
        #self.rv_0 = np.concatenate((rvr_0,np.flip(rvl_0, axis=0))) # boundary points of the beam
        return 

    def position(self, qvec, r_0=None):
        r_0 = self.rcm_0 if r_0 is None else r_0
        qvec_o, qvec_t = np.split(qvec, 2)
        # qvec_t = np.zeros(self.N) if not self.torsion else qvec_t
        # qvec_o = np.zeros(self.N) if not self.bending else qvec_o
        # qvec_i = np.zeros(self.N) if not self.in_bending else qvec_i
        C_d0 = np.array([(DCM(1, q0_t) @ DCM(2, q0_o)) for q0_o, q0_t in zip(np.cumsum(qvec_o), np.cumsum(qvec_t))])#np.array([DCM(2, q0) for q0 in np.cumsum(qvec)])
        delta_r = (r_0 - self.r0_0).reshape(-1, 3, order='F')
        delta_r0 = self.r0_0.reshape(-1, 3, order='F')[1]
        # r0_local = r0_0.reshape(-1, 3, order='F')[0]
        # r = np.zeros([len(delta_r), 3])
        r0_local = np.zeros([len(delta_r), 3], dtype=qvec.dtype)
        r0_local[1:] = np.cumsum([C @ delta_r0 for C in C_d0[:-1]], axis=0)
        r = [r_i + C_i @ del_i for r_i, C_i, del_i in zip(r0_local, C_d0, delta_r)]
        # r0_local = r0_local.at[1:].set(np.cumsum(np.array([C @ delta_r0 for C in C_d0[:-1]]), axis=0))
        # r = np.array([r0_local[i] + C_d0[i] @ delta_r[i] for i, _ in enumerate(C_d0)])
        # for i in range(0, len(q)):
        #     if i == 0:
        #         # r[i,:] = C_d0[i] @ delta_r[i]
        #         r = r.at[i].set(C_d0[i] @ delta_r[i])
        #     else:
        #         r0_local = r0_local + C_d0[i-1] @ delta_r0
        #         r = r.at[i].set(r0_local + C_d0[i] @ delta_r[i])
                # r[i,:] = r0_local + C_d0[i] @ delta_r[i]
        return np.array(r).T.reshape(-1)
    
    def jacobian(self, q_vec, r=None):
        J = complex_step(lambda q: self.position(q, r_0=r), q_vec)
        return J 
        
    def mass_matrix(self):
        self.M = np.diag(self.m0*self.l*np.ones(3*self.N))
        T = block_diag(np.zeros((self.N,self.N)),np.tril(np.ones((self.N,self.N))))
        self.I = T.T @ (self.I0*self.l*np.eye(2*self.N)) @ T
        return
    
    def stiffness_matrix(self):
        self.K = np.zeros((2*self.N, 2*self.N))
        y, z, phi = symbols('y', cls=Function), symbols('z', cls=Function), symbols('phi', cls=Function)
        x = symbols('x')
        z_sol = dsolve(-self.EIyy*z(x).diff(x, 4), z(x), 
                       ics={z(x).subs(x, 0): 0, 
                            z(x).diff(x).subs(x, 0): 0,
                            z(x).diff(x, 2).subs(x, self.L): 0,
                            z(x).diff(x, 3).subs(x, self.L): -1/self.EIyy})
        y_sol = dsolve(-self.EIzz*y(x).diff(x, 4), y(x), 
                       ics={y(x).subs(x, 0): 0, 
                            y(x).diff(x).subs(x, 0): 0,
                            y(x).diff(x, 2).subs(x, self.L): 0,
                            y(x).diff(x, 3).subs(x, self.L): -1/self.EIzz})
        phi_sol = dsolve(-self.GJ*phi(x).diff(x) + 1, phi(x), 
                         ics={phi(x).subs(x, 0): 0})
        tau_z = -self.EIyy*z_sol.rhs.diff(x, 2)
        z_sol = lambdify(x, z_sol.rhs)
        tau_z = lambdify(x, tau_z)
        tau_y = -self.EIzz*y_sol.rhs.diff(x, 2)
        y_sol = lambdify(x, y_sol.rhs)
        tau_y = lambdify(x, tau_y)
        tau_x = -self.GJ*phi_sol.rhs/x
        phi_sol = lambdify(x, phi_sol.rhs)
        tau_x = lambdify(x, tau_x)
        x_vec = np.insert(self.r1_0[:self.N], 0, 0.)
        z = z_sol(x_vec).tolist()
        tau_z = tau_z(x_vec).tolist()
        y = y_sol(x_vec).tolist()
        tau_y = tau_y(x_vec).tolist()
        phi = phi_sol(x_vec).tolist()
        tau_x = (tau_x(x_vec)*np.ones(self.N+1)).tolist()
        theta_z, theta_y, K_out, K_in, K_t = [0.], [0.], [], [], []
        for i in range(1, self.N+1):
            theta_z.append(np.arcsin(z[i] - z[i-1])/self.l)
            theta_y.append(np.arcsin(y[i] - y[i-1])/self.l)
            K_out.append(-tau_z[i-1]/(theta_z[i] - theta_z[i-1]))
            K_in.append(-tau_y[i-1]/(theta_y[i] - theta_y[i-1]))
            K_t.append(-tau_x[i-1]/(phi[i]-phi[i-1]))
        K_t[0] *= 2
        self.K[:self.N,:self.N] = np.diag(K_out)
        self.K[self.N:2*self.N,self.N:2*self.N] = np.diag(K_t)
        #self.K[2*self.N:3*self.N,2*self.N:3*self.N] = np.diag(K_in)
        return
    
    def damping_matrix(self):
        self.C = self.c*self.K
        self.C[self.N:2*self.N,self.N:2*self.N] *= 0.05
        return
    
    def g_force(self):
        self.Fg = np.zeros(3*self.N)
        self.Fg[2*self.N:] = -9.80665 * np.diag(self.M)[:self.N]
        return 
            
    def gen_force(self, t, J, q, F_tip=None):
        #Q = np.zeros(self.N)  
        F = external_force(self, t, F_tip)
            #tau = np.sum(tau[:,1])*np.ones(self.N)
        Q = J.T @ F #+ tau
        return Q
    
    def ic_fun(self, *args):
        match self.ic:
            case 'random':
                np.random.seed(self.seed)
                # q_min, q_max = np.deg2rad(args[0][0]), np.deg2rad(args[0][1])
                # p_min, p_max = args[1][0], args[1][1]
                # q0 = q_min + (q_max - q_min)*np.random.rand(self.N)
                # p0 = p_min + (p_max - p_min)*np.random.rand(self.N)
                F_span = args[0]
                Fx_tip = np.random.uniform(F_span[0], F_span[3], 1)
                Fy_tip = np.random.uniform(F_span[1], F_span[4], 1)
                Fz_tip = np.random.uniform(F_span[2], F_span[5], 1)
                q0 = fsolve(self.equilibrium, x0=np.zeros(2*self.N), args=(None, (Fx_tip, Fy_tip, Fz_tip)))
                p0 = np.zeros(2*self.N)
            case 'equilibrium':
                q0 = fsolve(self.equilibrium, x0=np.zeros(2*self.N)) 
                p0 = np.zeros(2*self.N)
            case 'null':
                q0 = np.zeros(2*self.N)
                p0 = np.zeros(2*self.N)
        x0 = q0, p0
        return x0
        
    def dynamics(self, t, x, model, *args):
        match model:
            case 'FOM':
                q, p = np.split(x, 2)
                J = self.jacobian(q)
                J_f = self.jacobian(q, r=self.r_fe)
                qdot = np.linalg.solve(J.T @ self.M @ J + self.I, p)
                pdot = -self.K @ q - self.C @ qdot 
                if self.fe: pdot += self.gen_force(t, J_f, q) 
                if self.g: pdot += J.T @ self.Fg  
                xdot = np.concatenate((qdot, pdot))
            case 'ROM':
                eta, pr = np.split(x, 2)
                q = self.Phi_r @ eta
                J = self.jacobian(q)
                J_f = self.jacobian(q, r=self.r_fe)
                etadot = np.linalg.solve(self.Phi_r.T @ (J.T @ self.M @ J + self.I) @ self.Phi_r, pr)
                prdot = - self.K_r @ eta - self.C_r @ etadot 
                if self.fe: prdot += self.Phi_r.T @ self.gen_force(t, J_f, q) 
                if self.g: prdot += self.Phi_r.T @ J.T @ self.Fg  
                xdot = np.concatenate((etadot, prdot))
            case _:
                q, p = np.split(x, 2)
                x_scaler, dx_scaler = args[0]['x_scaler'], args[0]['dx_scaler']
                x = x_scaler.transform(x.reshape(1,-1)) if x_scaler != [] else x.reshape(1,-1)
                x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
                xdot = model(x)['xdot_hat']
                xdot = dx_scaler.inverse_transform(xdot.data.numpy()).flatten() if dx_scaler != [] else xdot.data.numpy().flatten()
                J = self.jacobian(q) if args[0]['ref_model'] == 'FOM' else self.jacobian(self.Phi_r@q)
                J_f = self.jacobian(q, r=self.r_fe) if args[0]['ref_model'] == 'FOM' else self.jacobian(self.Phi_r@q, r=self.r_fe)
                if self.fe: xdot[len(xdot)//2:] += self.gen_force(t, J_f, q) if args[0]['ref_model'] == 'FOM' else self.Phi_r.T @ self.gen_force(t, J_f, q)
                if self.g: xdot[len(xdot)//2:] += J.T @ self.Fg if args[0]['ref_model'] == 'FOM' else self.Phi_r.T @ J.T @ self.Fg
        if args and 'pbar' in args[0]: update_pbar(t, args[0]) # updates progress bar
        return xdot
    
    def total_energy(self, t_vec, x_vec, model, ref_model):
        E = np.empty(x_vec.shape[1])
        for i in range(x_vec.shape[1]):
            q, p = np.split(x_vec[:, i], 2)
            if model == 'ROM' or (model != 'FOM' and ref_model == 'ROM'): 
                eta, pr = q, p 
                J = self.jacobian(self.Phi_r @ eta)
                E[i] = 0.5*(pr @ np.linalg.solve(self.Phi_r.T @ (J.T @ self.M @ J + self.I) @ self.Phi_r, pr)) + 0.5*(eta @ self.K_r @ eta) 
            else:
                J = self.jacobian(q)
                E[i] = 0.5*(p @ np.linalg.solve(J.T @ self.M @ J + self.I, p)) + 0.5*(q @ self.K @ q) 
        return E
        
    def equilibrium(self, q, model=None, F_tip=None):
        if model == 'ROM':
            Q_e = self.gen_force(0, self.jacobian(self.Phi_r@q, r=self.r_fe), self.Phi_r @ q, F_tip)
            Q_g = self.jacobian(self.Phi_r@q).T @ self.Fg
            trim_fun = self.Phi_r.T@(Q_g + Q_e) - self.K_r @ q
        else:
            Q_e = self.gen_force(0, self.jacobian(q, r=self.r_fe), q, F_tip)
            Q_g = self.jacobian(q).T @ self.Fg
            trim_fun = Q_g + Q_e - self.K @ q
        return trim_fun
    
    def reduced_model(self):
        q_eq = fsolve(self.equilibrium, np.zeros(2*self.N)) 
        J_eq = self.jacobian(q_eq)
        eigval, eigvec = np.linalg.eig(np.linalg.solve(J_eq.T @ self.M @ J_eq + self.I, self.K))
        Phi = eigvec[:, np.argsort(eigval)] # sort modes in ascending order of frequencies
        self.Phi_r = Phi[:, :self.r]
        self.K_r = self.Phi_r.T @ self.K @ self.Phi_r 
        self.C_r = self.Phi_r.T @ self.C @ self.Phi_r 
        return
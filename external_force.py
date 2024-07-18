#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:29:38 2024

@author: vitor
"""
import numpy as np


def external_force(beam, t, F_tip=None):
    F = np.zeros((beam.N, 3))
    if t > 0 and t <= 2:
        F[-1] = np.array([0, 0, 10*np.sin(50*t)])
    
    if F_tip is not None: # only called when ic_fun() produces random ic
        F[-1] = F_tip
    #delr_fe = (self.position(q, r_0=self.r_fe) - self.position(q))
    #tau = np.cross(delr_fe.reshape(-1, 3, order='F'), F.reshape(-1, 3, order='F'))
    return F.T.flatten()#, tau
# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, model_args):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(model_args.hidden_dim)):
            if i == 0:
                self.layers.append(nn.Linear(model_args.input_dim, model_args.hidden_dim[i]))     
            else:
                self.layers.append(nn.Linear(model_args.hidden_dim[i-1], model_args.hidden_dim[i]))
            # self.layers.append(nn.Dropout(0.1))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(model_args.hidden_dim[-1], model_args.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return {'xdot_hat': x}


class HNN(nn.Module): 
  def __init__(self, model_args):
    super(HNN, self).__init__()  # Inherit the methods of the Module constructor
    self.model_args = model_args
    self.model_args.output_dim = 1
    self.mlp = MLP(self.model_args)  # Instantiate an instance of our baseline model.
    
  def forward(self, x, t=None):
    inputs = torch.cat([x, t], axis=-1) if t is not None else x
    H_hat = self.mlp(inputs)['xdot_hat']  # Bx2 Get the scalars from our baseline model
    grad_H_hat = torch.autograd.grad(H_hat.sum(), x, create_graph=True)[0]
    dHdq, dHdp = torch.split(grad_H_hat, grad_H_hat.shape[-1]//2, dim=1)
    Xdot_hat = torch.cat([dHdp, -dHdq], axis=-1)
    return {'xdot_hat': Xdot_hat, 'H_hat': H_hat}


class DHNN(nn.Module):
  def __init__(self, model_args):
    super(DHNN, self).__init__()  # Inherit the methods of the Module constructor
    model_args.output_dim = 1
    self.mlp_h = MLP(model_args)  # Instantiate an MLP for learning the conservative component
    self.mlp_d = MLP(model_args)  # Instantiate an MLP for learning the dissipative component
    
  def forward(self, x, t=None, as_separate=False): 
    inputs = torch.cat([x, t], axis=-1) if t is not None else x
    D_hat = self.mlp_d(inputs)['xdot_hat']
    H_hat = self.mlp_h(inputs)['xdot_hat']
    
    irr_component = torch.autograd.grad(D_hat.sum(), x, create_graph=True)[0]  # Take their gradients
    rot_component = torch.autograd.grad(H_hat.sum(), x, create_graph=True)[0]
    # For H, we need the symplectic gradient, and therefore we split our tensor into 2 and swap the chunks.
    dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1]//2, dim=1)
    dDdq, dDdp = torch.split(irr_component, irr_component.shape[-1]//2, dim=1)
    # q_dot_hat, p_dot_hat = dHdp, -dHdq
    Xdot_rot_hat = torch.cat([dHdp, -dHdq], axis=-1)
    Xdot_irr_hat = torch.cat([dDdq, dDdp], axis=-1)
    if as_separate:
        return Xdot_irr_hat, Xdot_rot_hat  # Return the two fields seperately, or return the composite field. 

    return {'xdot_hat': Xdot_irr_hat + Xdot_rot_hat, 'H_hat': H_hat, 'D_hat': D_hat}


#%% Tensorflow Version
    
# import tensorflow as tf
# from keras.layers import Dense
# import time

# class MLP(tf.keras.Model): 
#     def __init__(self, model_args):
#       super(MLP, self).__init__()
#       self.layer = []
#       for i in range(len(model_args.hidden_dim)):
#           if i == 0:
#               self.layers.append(Dense(model_args.hidden_dim[i], activation=model_args.activation[i], 
#                                        input_dim=model_args.input_dim, trainable=True))
#           else:
#               self.layer.append(Dense(model_args.hidden_dim[i], activation=model_args.activation[i], 
#                                       trainable=True))
#       self.layer.append(Dense(model_args.output_dim, trainable=True))
      
#     def call(self, x):
#         for layer in self.layer:
#             x = layer(x)
#         return x


# class HNN(tf.keras.Model):
#     def __init__(self, model_args):
#         super(HNN, self).__init__()
#         model_args.output_dim = 1
#         self.mlp = MLP(model_args)
  
#     def call(self, x):
#         with tf.GradientTape() as tape:
#             tape.watch(x)
#             H_hat = self.mlp(x)[...,0]
#             # tf.print(tf.shape(H))
#             H_hat = tf.reduce_sum(H_hat)
#         # tf.print(tf.shape(H))
#         H_grad = tape.gradient(H_hat, x)
#         # tf.print(H_grad)
#         dHdp, dHdq = tf.split(H_grad, num_or_size_splits=2, axis=-1)
        
#         q_dot_hat, p_dot_hat = dHdp, -dHdq
#         X_dot_hat = tf.concat([p_dot_hat, q_dot_hat], axis=-1)
#         # tf.print(tf.shape(H_hat))
#         return X_dot_hat


# class DHNN(tf.keras.Model):
#     def __init__(self, model_args):
#         super(DHNN, self).__init__()
#         model_args.output_dim = 1
#         self.mlp_h = MLP(model_args)
#         self.mlp_d = MLP(model_args)
    
#     def call(self, x):
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(x)
#             H_hat = self.mlp_h(x)[...,0]
#             # H_hat = tf.reduce_sum(H_hat)
#             D_hat = self.mlp_d(x)[...,0]
#             # D_hat = tf.reduce_sum(D_hat)
#             # H = tf.reduce_sum(H)
#         # tf.print(tf.shape(H_hat))
#         H_hat_grad = tape.gradient(H_hat, x)
#         D_hat_grad = tape.gradient(D_hat, x)
#         del tape
#         # tf.print(tf.shape(H_grad))
#         dHdp, dHdq = tf.split(H_hat_grad, num_or_size_splits=2, axis=1)
#         q_dot_hat, p_dot_hat = dHdp, -dHdq
#         X_dot_rot_hat = tf.concat([p_dot_hat, q_dot_hat], axis=1)
#         X_dot_hat = tf.add(X_dot_rot_hat, D_hat_grad)        
#         return X_dot_hat
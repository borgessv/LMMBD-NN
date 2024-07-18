import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('plot.mplstyle')


class L2Loss:
    def __call__(self, predicted, target):
        # pdot_pred, qdot_pred = xdot_pred.split(xdot_pred.shape[-1]//2, dim=1)
        # pdot_target, qdot_target = xdot_target.split(xdot_target.shape[-1]//2, dim=1)
        return (predicted-target).pow(2).mean()#(torch.linalg.vector_norm(pdot_pred - pdot_target, ord=2, dim=1) + torch.linalg.vector_norm(qdot_pred - qdot_target, ord=2, dim=1)).mean() #(predicted - target).pow(2).mean()
    
    
def train(model, data, nn_args):
    device = torch.device(nn_args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_args.learning_rate)
    loss_function = L2Loss() if nn_args.loss == 'L2' else torch.nn.MSELoss()
    
    # Convert the data to PyTorch tensors
    model.to(nn_args.device)
    x_train = torch.tensor(data['x_train'], dtype=torch.float32, requires_grad=True, device=device)
    dx_train = torch.tensor(data['dx_train'], dtype=torch.float32, device=device)
    x_val = torch.tensor(data['x_val'], dtype=torch.float32, requires_grad=True, device=device)
    dx_val = torch.tensor(data['dx_val'], dtype=torch.float32, device=device)
    
    train_losses = []
    val_losses = []
    
    # Training loop:
    for epoch in range(nn_args.epochs):
        model.train()
        
        indices = torch.randperm(len(x_train))
        x_train_shuffled = x_train[indices]
        dx_train_shuffled = dx_train[indices]
        
        for i in range(0, len(x_train_shuffled), nn_args.batch_size):
            x_batch = x_train_shuffled[i:i+nn_args.batch_size]
            dx_batch = dx_train_shuffled[i:i+nn_args.batch_size]
            
            # Forward pass
            Xdot_hat = model(x_batch)['xdot_hat']
            loss = loss_function(Xdot_hat, dx_batch)
            
            # Backward pass and optimization
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Evaluate model on validation set
        model.eval()
        # with torch.no_grad():
        Xdot_hat_val = model(x_val)['xdot_hat']
        val_loss = loss_function(Xdot_hat_val, dx_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        # Print progress
        print(f"Training {model.__class__.__name__} model -- Epoch {epoch+1}/{nn_args.epochs}, Loss: {loss.item():.4e}, Val Loss: {val_loss.item():.4e}")
           
    return {'train': train_losses, 'val': val_losses}
    
def loss_plot(losses):
    fig, ax = plt.subplots()
    for m in losses:
        plt.plot([*range(1, len(losses[m]['train'])+1)], losses[m]['train'], label='Train: '+m)
        plt.plot([*range(1, len(losses[m]['train'])+1)], losses[m]['val'], label='Validation: '+m)
    ax.set_yscale('log') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(which='both')
    plt.savefig('images/loss_plot.png', dpi=150, format='png', bbox_inches='tight', transparent=True)
    plt.show()
    
    
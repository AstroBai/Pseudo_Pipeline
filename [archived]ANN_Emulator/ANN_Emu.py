import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import yaml
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm

class ANN_Emu(nn.Module):
    """
    ANN Emulator with configurable layers, activation, dropout, weight decay,
    automatic data loading, input/output size detection, and optional PCA for output dimensionality reduction.
    """
    def __init__(self, config_path='config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config  
        ann_param = config['ANNParam']
        hidden_sizes = ann_param['S_Hidden']
        activation_name = ann_param['Activation'].lower()
        dropout_rate = ann_param.get('Dropout', 0.0)  # Add dropout rate from config

        # === Load data ===
        data_config = config['DataParam']
        # Load parameter and data tensors
        Train_X = torch.tensor(np.loadtxt(data_config['Training_ParamPath'], skiprows=1), dtype=torch.float32)
        self.Train_X_original = Train_X.clone()
        self.Train_Y_original = torch.tensor(np.load(data_config['Training_DataPath']), dtype=torch.float32)[:,:,:-1]
        Trial_X = torch.tensor(np.loadtxt(data_config['Trial_ParamPath'], skiprows=1), dtype=torch.float32)
        self.Trial_X_original = Trial_X.clone()
        self.Trial_Y_original = torch.tensor(np.load(data_config['Trial_DataPath']), dtype=torch.float32)
        # Scale input parameters if needed
        if ann_param['Scale_Params']:
            scaler = MinMaxScaler()
            Train_X = torch.tensor(scaler.fit_transform(Train_X), dtype=torch.float32)
            Trial_X = torch.tensor(scaler.transform(Trial_X), dtype=torch.float32)      
            # save scaler, create file path if not exists
            if not os.path.exists(config['OutputParam']['Model_SavePath']):
                os.makedirs(config['OutputParam']['Model_SavePath'])
            scaler_path = os.path.join(config['OutputParam']['Model_SavePath'], 'input_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            print(f"Input scaler saved to {scaler_path}")
        # Log-transform output if needed
        ks = self.Train_Y_original[0,0,:].numpy()
        if data_config['Do_Log']:
            Train_Y = torch.log(self.Train_Y_original[1,:,:])
            Trial_Y = torch.log(self.Trial_Y_original[1,:,:])
        else:
            Train_Y = self.Train_Y_original[1,:,:]
            Trial_Y = self.Trial_Y_original[1,:,:]

        # === PCA for output dimensionality reduction ===
        self.use_pca = ann_param['Use_PCA']
        self.n_pca_components = ann_param['N_PCA_Components']
        if self.use_pca:
            self.pca = PCA(n_components=self.n_pca_components)
            Train_Y_np = Train_Y.numpy()
            self.pca.fit(Train_Y_np)
            Train_Y_pca = self.pca.transform(Train_Y_np)
            Trial_Y_pca = self.pca.transform(Trial_Y.numpy())
            Train_Y = torch.tensor(Train_Y_pca, dtype=torch.float32)
            Trial_Y = torch.tensor(Trial_Y_pca, dtype=torch.float32)
            output_size = self.n_pca_components
            # print PCs
            print("PCA components:", self.pca.components_)
            # print coefficients for first 5 training samples
            print("PCA coefficients for first 5 training samples:", Train_Y[:5,:])
            # save PCA object
            pca_path = os.path.join(config['OutputParam']['Model_SavePath'], 'pca.pkl')
            joblib.dump(self.pca, pca_path)
            print(f"PCA object saved to {pca_path}")
        else:
            output_size = Train_Y.shape[1]

        input_size = Train_X.shape[1]

        # Save data and sizes
        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Trial_X = Trial_X
        self.Trial_Y = Trial_Y
        self.ks = ks
        self.input_size = input_size
        self.output_size = output_size

        super(ANN_Emu, self).__init__()
        # Build network layers with dropout
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if activation_name == 'relu':
                layers.append(nn.ReLU())
            elif activation_name == 'tanh':
                layers.append(nn.Tanh())
            elif activation_name == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_name == 'gelu':
                layers.append(nn.GELU())
            elif activation_name == 'silu':
                layers.append(nn.SiLU())
            elif activation_name == 'linear':
                pass  # No activation
            else:
                raise ValueError(f"Unknown activation: {activation_name}")
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_data_loaders(self):
        """Return PyTorch DataLoader objects for training and trial sets."""
        train_dataset = TensorDataset(self.Train_X, self.Train_Y)
        train_loader = DataLoader(train_dataset, shuffle=True)
        trial_dataset = TensorDataset(self.Trial_X, self.Trial_Y)
        trial_loader = DataLoader(trial_dataset, shuffle=False)
        return train_loader, trial_loader

    def train_model(self, train_loader, trial_loader=None):
        """Train the model using config parameters and optional early stopping."""
        ann_param = self.config['ANNParam']
        output_param = self.config['OutputParam']
        num_epochs = ann_param['Max_Epochs']
        learning_rate = ann_param['Initial_LR']
        gamma = ann_param['Gamma']
        step_size = ann_param['Step_Size']
        min_delta = float(ann_param['Min_Delta'])
        patience = float(ann_param['Patience']) 
        weight_decay = float(ann_param['Weight_Decay'])  # Add weight decay from config
        save_path = os.path.join(output_param['Model_SavePath'], 'ann_emu.pth')
        plot_loss = ann_param['Plot_Loss']

        device = self.config['ANNParam']['Device']
        print(f'Using device: {device}')
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        waited = 0
        loss_training = []
        loss_trial = []

        for epoch in range(int(num_epochs)):
            self.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(train_loader.dataset)
            scheduler.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                loss_training.append(epoch_loss)
                if trial_loader is not None:
                    self.eval()
                    with torch.no_grad():
                        t_loss = 0.0
                        for t_inputs, t_targets in trial_loader:
                            t_inputs, t_targets = t_inputs.to(device), t_targets.to(device)
                            t_outputs = self.forward(t_inputs)
                            t_loss += criterion(t_outputs, t_targets).item() * t_inputs.size(0)
                        t_loss /= len(trial_loader.dataset)
                        loss_trial.append(t_loss)
                    self.train()
                delta = abs(loss_training[-1] - loss_training[-2]) if len(loss_training) > 1 else float('inf')
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Delta: {delta}')
                # Early stopping
                if delta < min_delta:
                    waited += 1
                    if waited >= patience:
                        print("Early stopping")
                        break
                else:
                    waited = 0

                # Plot loss curve
                if plot_loss:
                    plt.figure()
                    plt.plot(range(1, len(loss_training)+1), loss_training, label='Training Loss')
                    if trial_loader is not None:
                        plt.plot(range(1, len(loss_trial)+1), loss_trial, label='Trial Loss')
                    plt.xlabel('Epochs (x{})'.format(10))
                    plt.ylabel('Loss')
                    plt.yscale('log')
                    plt.legend()
                    plt.title('Loss over Epochs')
                    plt.grid()
                    plt.savefig(os.path.join(output_param['Model_SavePath'], 'loss_curve.png'))
                    plt.close()

        # Save the trained model
        torch.save(self.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def predict(self, x):
        """Predict output for input x. If PCA is used, inverse transform the result."""
        # load scaler
        scaler_path = os.path.join(self.config['OutputParam']['Model_SavePath'], 'input_scaler.pkl')
        scaler = joblib.load(scaler_path)
        x = scaler.transform(x)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device)
            predictions = self.forward(x).cpu().numpy()
            if self.use_pca:
                predictions = self.pca.inverse_transform(predictions)
                if self.config['DataParam']['Do_Log']:
                    predictions = np.exp(predictions)  # reverse log-transform
        return predictions

# Example usage:
if __name__ == "__main__":
    emu = ANN_Emu(config_path='config.yaml')
    train_loader, trial_loader = emu.get_data_loaders()
    emu.train_model(train_loader, trial_loader)

    # Draw prediction vs true for trial set and errors in 1 figure
    predictions = emu.predict(emu.Trial_X_original.numpy())
    true_values = emu.Trial_Y_original[1,:,:].numpy()
    print("Predictions:", predictions)
    print("True Values:", true_values)  
    num_points = true_values.shape[0]
    plt.figure(figsize=(10, 8))
    errors = predictions - true_values
    for i in range(num_points):
        plt.plot(emu.ks, errors[i, :] / true_values[i, :])
    plt.xscale('log')    
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Relative Error')
    plt.title('Prediction Errors')
    plt.grid()

    plt.tight_layout()
    outdir = emu.config['OutputParam']['Model_SavePath']
    plt.savefig(os.path.join(outdir, 'prediction_errors.png'))
    plt.close()
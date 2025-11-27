# Plotting the distribution of the nodes of training set and trial set
import numpy as np
import matplotlib.pyplot as plt


# read parameters

Train_ParamPath = '/home/jbai/cosmology/ann_gp/Pseudo_Pipeline/PseudoEmulator/Training_Set/Nodes/Seed1Mx1.0_CPFinal_BFMixed-DataUltimate_Nodes200_Dim13_NoHeader.dat'
Trial_ParamPath = '/home/jbai/cosmology/ann_gp/Pseudo_Pipeline/PseudoEmulator/Trial_Set/Nodes/Seed2Mx1.0_CPFinal_BFMixed-DataUltimate12.5_Nodes200_Dim13_NoHeader.dat'

Train_Params = np.loadtxt(Train_ParamPath)
Trial_Params = np.loadtxt(Trial_ParamPath)

# # Om_m, Om_b, H, ns, As, 8 Weights... (13 in total)

param_names = [r'$\omega_m$', r'$\omega_b$', r'$h$', r'$n_s$', r'$A_s$', r'$w^1$' , r'$w^2$', r'$w^3$', r'$w^4$', r'$w^5$', r'$w^6$', r'$w^7$', r'$w^8$']

num_params = Train_Params.shape[1]

# Plotting corner plot, training with gray, trial with red
fig, axes = plt.subplots(num_params, num_params, figsize=(20, 20))
for i in range(num_params):
    for j in range(num_params):
        ax = axes[i, j]
        if i == j:
            # Diagonal: 1D histogram
            ax.hist(Train_Params[:, i], bins=30, color='gray', alpha=0.5, label='Training Set')
            ax.hist(Trial_Params[:, i], bins=30, color='red', alpha=0.5, label='Trial Set')

        elif i > j:
            # Lower triangle: 2D histogram
            ax.scatter(Train_Params[:, j], Train_Params[:, i], c='Grey', alpha=0.5, s=1, label='Training Set')
            ax.scatter(Trial_Params[:, j], Trial_Params[:, i], c='Red', alpha=0.5, s=1, label='Trial Set')
        else:
            # Upper triangle: empty
            ax.axis('off')

        if i == num_params - 1:
            ax.set_xlabel(param_names[j], fontsize=12)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        if j == 0:
            ax.set_ylabel(param_names[i], fontsize=12)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

plt.tight_layout()
plt.savefig('Parameter_Distribution_Train_Trial.png', dpi=300)                        
plt.close()
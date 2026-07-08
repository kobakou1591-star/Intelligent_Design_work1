import numpy as np
import torch
from torch.utils.data import Dataset

class MRRDataset(Dataset):
    def __init__(self, num_samples, tau1_range, tau2_range, alpha_range, FSR, wl:torch.Tensor):
        self.num_samples = num_samples
        self.wl = wl
        self.tau1_range = tau1_range
        self.tau2_range = tau2_range
        self.alpha_range = alpha_range
        self.FSR = FSR

        if tau1_range[0] == tau1_range[1]:
            self.tau1 = torch.full((num_samples, 1), tau1_range[0]) # for fixed tau1
        else:
            self.tau1 = torch.distributions.Uniform(tau1_range[0], tau1_range[1]).sample((num_samples, 1))

        if tau2_range[0] == tau2_range[1]:
            self.tau2 = torch.full((num_samples, 1), tau2_range[0]) # for fixed tau2
        else:
            self.tau2 = torch.distributions.Uniform(tau2_range[0], tau2_range[1]).sample((num_samples, 1))
        
        if alpha_range[0] == alpha_range[1]:
            self.alpha = torch.full((num_samples, 1), alpha_range[0]) # for fixed alpha
        else:
            self.alpha = torch.distributions.Uniform(alpha_range[0], alpha_range[1]).sample((num_samples, 1))
        
        # Calculate Complex Transfer Function:
        c0=299792458
        n_effL=2*torch.pi*c0/self.FSR
        theta=2*torch.pi*n_effL/self.wl
        Input = 1-2 * self.alpha * self.tau1 * self.tau2 * torch.cos(theta) + (self.alpha * self.tau1 * self.tau2)**2
        Output_th = self.tau1**2 -2 * self.alpha * self.tau1 * self.tau2 * torch.cos(theta) + (self.alpha * self.tau2 )**2
        Output_dr = (1-self.tau1**2) * (1-self.tau2**2) * self.alpha

        self.T_th = Output_th / Input
        self.T_dr = Output_dr / Input
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        T_th = self.T_th[idx]
        T_dr = self.T_dr[idx]
        tau1 = self.tau1[idx]
        tau2 = self.tau2[idx]
        alpha = self.alpha[idx]

        circuit_parameters = torch.cat([tau1, tau2, alpha], dim=1)

        return T_th, T_dr, circuit_parameters

if __name__ == "__main__":

    wl = torch.linspace(1549.5e-9, 1550.5e-9, 100)
    dataset = MRRDataset(num_samples=1000, tau1_range=[0.1, 0.9], tau2_range=[0.1, 0.9], alpha_range=[0.1, 0.9], FSR=50e9, wl=wl)
    idx = [0, 1, 2]
    T_th_actual, T_dr_actual, circuit_parameters = dataset[idx]
    tau1 = circuit_parameters[:, 0]      # tau1 values
    tau2 = circuit_parameters[:, 1]      # tau2 values
    alpha = circuit_parameters[:, 2]    # alpha values

    for i in idx:
        print(f"T_th: {T_th_actual[i]}, T_dr: {T_dr_actual[i]}, tau1: {tau1[i]}, tau2: {tau2[i]}, alpha: {alpha[i]}")

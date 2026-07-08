import numpy as np
import torch
from torch.utils.data import Dataset

# class RCDataset(Dataset):
#     def __init__(self, num_samples, R_range, C_range, fs:torch.Tensor):
#         """
#         Args:
#             num_samples: Number of data points to generate
#             R_range: [min_R, max_R]
#             C_range: [min_C, max_C]
#             fs: The frequency array (Hz) to evaluate H for

#             For fixed R, use R_range = [R, R].
#         """
#         self.num_samples = num_samples
#         self.fs = fs
#         self.ws = 2 * torch.pi * self.fs
#         self.R_range = R_range
#         self.C_range = C_range

#         # R の生成
#         if R_range[0] == R_range[1]:
#             self.R = torch.full((num_samples, 1), R_range[0]) # for fixed R
#         else:
#             self.R = torch.distributions.Uniform(R_range[0], R_range[1]).sample((num_samples, 1))

#         # C の生成
#         if C_range[0] == C_range[1]:
#             self.C = torch.full((num_samples, 1), C_range[0]) # for fixed C
#         else:
#             self.C = torch.distributions.Uniform(C_range[0], C_range[1]).sample((num_samples, 1))

#         # Calculate Complex Transfer Function: H(jw) = 1 / (1 + jwRC)
#         # Shape: (num_samples, len(fs))
#         jwRC = 1j * self.ws * self.R * self.C
#         self.H = 1 / (1 + jwRC)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         # To match the expected format for plotting/analysis:
#         # Returns (H, R, C)
#         H = self.H[idx]   # Complex frequency response
#         r = self.R[idx]   # Resistance value
#         c = self.C[idx]   # Capacitance value

#         circuit_parameters = torch.cat([r, c], dim=1)

#         return H, circuit_parameters

class MRRDataset(Dataset):
    def __init__(self, num_samples, tau1_range, tau2_range, alphas_range, FSR, wl:torch.Tensor):
        self.num_samples = num_samples
        self.wl = wl
        self.tau1_range = tau1_range
        self.tau2_range = tau2_range
        self.alphas_range = alphas_range
        self.FSR = FSR

        if tau1_range[0] == tau1_range[1]:
            self.tau1 = torch.full((num_samples, 1), tau1_range[0]) # for fixed tau1
        else:
            self.tau1 = torch.distributions.Uniform(tau1_range[0], tau1_range[1]).sample((num_samples, 1))

        if tau2_range[0] == tau2_range[1]:
            self.tau2 = torch.full((num_samples, 1), tau2_range[0]) # for fixed tau2
        else:
            self.tau2 = torch.distributions.Uniform(tau2_range[0], tau2_range[1]).sample((num_samples, 1))
        
        if alphas_range[0] == alphas_range[1]:
            self.alphas = torch.full((num_samples, 1), alphas_range[0]) # for fixed alphas
        else:
            self.alphas = torch.distributions.Uniform(alphas_range[0], alphas_range[1]).sample((num_samples, 1))
        
        # Calculate Complex Transfer Function:
        c0=299792458
        n_effL=2*torch.pi*c0/self.FSR
        theta=2*torch.pi*n_effL/self.wl
        Input = 1-2 * self.alpha * self.tau1 * self.tau2 * torch.cos(theta) + (self.alpha * self.tau1 * self.tau2)**2
        Output_th = self.tau1**2 -2 * self.alpha * self.tau1 * self.tau2 * torch.cos(theta) + (self.alpha * self.tau1 )**2
        Output_dr = (1-self.tau1**2) * (1-self.alpha**2) * self.tau2

        self.T_th = Output_th / Input
        self.T_dr = Output_dr / Input
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        T_th = self.T_th[idx]
        T_dr = self.T_dr[idx]
        tau1 = self.tau1[idx]
        tau2 = self.tau2[idx]
        alphas = self.alphas[idx]

        circuit_parameters = torch.cat([tau1, tau2, alphas], dim=1)

        return T_th, T_dr, circuit_parameters

if __name__ == "__main__":

    wl = torch.linspace(1549.5e-9, 1550.5e-9, 100)
    dataset = MRRDataset(num_samples=1000, tau1_range=[0.1, 0.9], tau2_range=[0.1, 0.9], alphas_range=[0.1, 0.9], FSR=50e9, wl=wl)
    idx = [0, 1, 2]
    T_th_actual, T_dr_actual, circuit_parameters = dataset[idx]
    tau1 = circuit_parameters[:, 0]      # tau1 values
    tau2 = circuit_parameters[:, 1]      # tau2 values
    alphas = circuit_parameters[:, 2]    # alphas values

    for i in idx:
        print(f"T_th: {T_th_actual[i]}, T_dr: {T_dr_actual[i]}, tau1: {tau1[i]}, tau2: {tau2[i]}, alphas: {alphas[i]}")

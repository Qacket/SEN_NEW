import numpy as np
import torch

class A_VAE_Loss(torch.nn.Module):
    def __init__(self):
        super(A_VAE_Loss, self).__init__()
        self.mseloss = torch.nn.MSELoss()

    def KL_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def forward(self, mu, log_var, recon_x, x):

        KLD = self.KL_loss(mu, log_var)
        BCE = self.mseloss(recon_x, x)
        return KLD, BCE


class T_VAE_Loss(torch.nn.Module):

    def __init__(self):
        super(T_VAE_Loss, self).__init__()
        self.nlloss = torch.nn.NLLLoss()

    def KL_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def reconstruction_loss(self, x_hat_param, x):
        x = x.view(-1).contiguous()
        x_hat_param = x_hat_param.view(-1, x_hat_param.size(2))
        recon = self.nlloss(x_hat_param, x)
        return recon

    def forward(self, mu, log_var, x_hat_param, x):
        kl_loss = self.KL_loss(mu, log_var)
        recon_loss = self.reconstruction_loss(x_hat_param, x)
        return kl_loss, recon_loss
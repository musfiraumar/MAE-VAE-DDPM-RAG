import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self,d=16):
        super().__init__()
        self.d=d
        self.enc_conv=nn.Sequential(nn.Conv2d(1,32,4,stride=2,padding=1),nn.ReLU(),nn.Conv2d(32,64,4,stride=2,padding=1),nn.ReLU(),nn.Flatten())
        self.fc_mu=nn.Linear(64*7*7,d)
        self.fc_logvar=nn.Linear(64*7*7,d)
        self.dec_fc=nn.Linear(d,64*7*7)
        self.dec_conv=nn.Sequential(nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),nn.ConvTranspose2d(32,1,4,stride=2,padding=1))
    def encode(self,x):
        h=self.enc_conv(x)
        return self.fc_mu(h),self.fc_logvar(h)
    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+std*eps
    def decode(self,z):
        h=self.dec_fc(z)
        h=h.view(-1,64,7,7)
        return self.dec_conv(h)
    def forward(self,x):
        mu,logvar=self.encode(x)
        z=self.reparameterize(mu,logvar)
        return self.decode(z),mu,logvar
def kl_divergence(mu,logvar):
    return -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(),dim=1)
def elbo_terms(logits,x,mu,logvar):
    recon=F.binary_cross_entropy_with_logits(logits,x,reduction="none").sum(dim=(1,2,3))
    kl=kl_divergence(mu,logvar)
    return recon,kl
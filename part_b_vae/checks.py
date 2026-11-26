import torch
from model import VAE,kl_divergence
mu0,lv0=torch.zeros(8,16),torch.zeros(8,16)
print("KL at mu=0, logvar=0:",kl_divergence(mu0,lv0))
assert torch.allclose(kl_divergence(mu0,lv0),torch.zeros(8))
mu=torch.randn(10000,16)*3
lv=torch.randn(10000,16)*2
kl=kl_divergence(mu,lv)
print("min KL over 10000 random cases:",kl.min().item())
assert (kl>=-1e-5).all()
m=VAE(16)
x=torch.rand(4,1,28,28)
logits,mu_,lv_=m(x)
print("logits",tuple(logits.shape),"mu",tuple(mu_.shape),"logvar",tuple(lv_.shape))
import sys,json,os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append("/content/starter")
from uhat import load_uhat_characters
from viz import save_grid
from model import VAE
torch.manual_seed(0)
device="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("/content/results",exist_ok=True)
_,_,names=load_uhat_characters(root="/content/starter/data/uhat",split="train")
x_te,l_te,_=load_uhat_characters(root="/content/starter/data/uhat",split="test",class_names=names)
model=VAE(16).to(device)
model.load_state_dict(torch.load("/content/results/vae.pt",map_location=device))
model.eval()
log=json.load(open("/content/results/vae_log.json"))
epochs=range(1,len(log["train"])+1)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,key,title in zip(axes,["recon","kl","total"],["Reconstruction (BCE)","KL","Negative ELBO"]):
    ax.plot(epochs,[e[key] for e in log["train"]],label="train")
    ax.plot(epochs,[e[key] for e in log["test"]],label="test")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats per image")
    ax.legend()
fig.tight_layout()
fig.savefig("/content/results/vae_loss_terms.png",dpi=120)
plt.close(fig)
with torch.no_grad():
    idx=torch.randperm(len(x_te))[:16]
    x=x_te[idx].to(device)
    mu,logvar=model.encode(x)
    recon=torch.sigmoid(model.decode(mu))
    save_grid(torch.cat([x,recon]),"/content/results/vae_reconstructions.png",nrow=16)
    z=torch.randn(64,16,device=device)
    samples=torch.sigmoid(model.decode(z))
    save_grid(samples,"/content/results/vae_samples.png",nrow=8)
    i=0
    j=int((l_te!=l_te[i]).nonzero()[0])
    pair=x_te[[i,j]].to(device)
    mu_pair,_=model.encode(pair)
    alphas=torch.linspace(0,1,10,device=device).view(10,1)
    z_line=(1-alphas)*mu_pair[0]+alphas*mu_pair[1]
    interp=torch.sigmoid(model.decode(z_line))
    save_grid(interp,"/content/results/vae_interpolation.png",nrow=10)
    print("interpolating",names[int(l_te[i])],"->",names[int(l_te[j])])
print("final test:",log["test"][-1])
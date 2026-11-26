import sys,time,json
import torch
from torch.utils.data import DataLoader,TensorDataset
sys.path.append("../starter")
from uhat import load_uhat_characters
from model import VAE,elbo_terms
torch.manual_seed(0)
device="cuda" if torch.cuda.is_available() else "cpu"
x_tr,l_tr,names=load_uhat_characters(root="../starter/data/uhat",split="train")
x_te,l_te,_=load_uhat_characters(root="../starter/data/uhat",split="test",class_names=names)
print("train",tuple(x_tr.shape),"test",tuple(x_te.shape),"classes",len(names))
train_loader=DataLoader(TensorDataset(x_tr),batch_size=128,shuffle=True)
test_loader=DataLoader(TensorDataset(x_te),batch_size=128,shuffle=False)
model=VAE(d=16).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
n_params=sum(p.numel() for p in model.parameters())
print("parameters:",n_params)
def run_epoch(loader,train):
    model.train(train)
    sums={"recon":0.0,"kl":0.0,"total":0.0}
    n=0
    with torch.set_grad_enabled(train):
        for (x,) in loader:
            x=x.to(device)
            logits,mu,logvar=model(x)
            recon,kl=elbo_terms(logits,x,mu,logvar)
            loss=(recon+kl).mean()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sums["recon"]+=recon.sum().item()
            sums["kl"]+=kl.sum().item()
            sums["total"]+=(recon+kl).sum().item()
            n+=x.size(0)
    return {k:v/n for k,v in sums.items()}
log={"train":[],"test":[]}
start=time.time()
for epoch in range(1,31):
    tr=run_epoch(train_loader,train=True)
    te=run_epoch(test_loader,train=False)
    log["train"].append(tr)
    log["test"].append(te)
    print(f"epoch {epoch:2d} | train recon {tr['recon']:.2f} kl {tr['kl']:.2f} total {tr['total']:.2f} | test recon {te['recon']:.2f} kl {te['kl']:.2f} total {te['total']:.2f}")
elapsed=time.time()-start
log["params"]=n_params
log["seconds"]=elapsed
log["gpu"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
torch.save(model.state_dict(),"../results/vae.pt")
with open("../results/vae_log.json","w") as f:
    json.dump(log,f,indent=2)
print("done in",round(elapsed),"seconds")
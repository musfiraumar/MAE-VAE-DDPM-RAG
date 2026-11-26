import torch
from transformers import AutoTokenizer,AutoModel
ENCODER_NAME="sentence-transformers/all-MiniLM-L6-v2"
def load_encoder(device):
    tok=AutoTokenizer.from_pretrained(ENCODER_NAME)
    enc=AutoModel.from_pretrained(ENCODER_NAME).to(device).eval()
    return tok,enc
@torch.no_grad()
def embed_texts(texts,tok,enc,device,batch_size=128,max_length=256):
    out=[]
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        t=tok(batch,padding=True,truncation=True,max_length=max_length,return_tensors="pt").to(device)
        hidden=enc(**t).last_hidden_state
        mask=t["attention_mask"].unsqueeze(-1).float()
        mean=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1e-9)
        mean=mean/mean.norm(dim=1,keepdim=True)
        out.append(mean.cpu())
        if (i//batch_size)%50==0:
            print(f"  embedded {min(i+batch_size,len(texts))}/{len(texts)}")
    return torch.cat(out).numpy()

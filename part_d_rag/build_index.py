import sys,os,json,time
import numpy as np
import torch
sys.path.append("../starter")
from bioasq import load_corpus
from chunking import build_chunks
from embedding import load_encoder,embed_texts
device="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("../results/index",exist_ok=True)
corpus=load_corpus()
texts,doc_ids=build_chunks(corpus)
print("abstracts:",len(corpus))
print("chunks   :",len(texts))
tok,enc=load_encoder(device)
t0=time.time()
emb=embed_texts(texts,tok,enc,device)
secs=time.time()-t0
print("embedding matrix:",emb.shape,emb.dtype,f"({secs:.0f}s)")
np.save("../results/index/chunk_embeddings.npy",emb)
json.dump({"abstracts":len(corpus),"chunks":len(texts),"embed_seconds":secs},open("../results/index/index_info.json","w"))

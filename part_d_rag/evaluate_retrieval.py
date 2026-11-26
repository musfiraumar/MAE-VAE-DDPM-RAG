import sys,json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append("../starter")
from bioasq import load_corpus,load_questions
from chunking import build_chunks
from embedding import load_encoder,embed_texts
from search import Index
from metrics import retrieval_metrics,recall_at_k,hit_at_k
device="cuda" if torch.cuda.is_available() else "cpu"
corpus=load_corpus()
qs=load_questions("test",corpus)
texts,chunk_doc_ids=build_chunks(corpus)
emb=np.load("../results/index/chunk_embeddings.npy")
assert emb.shape[0]==len(texts),"index does not match the chunks: rebuild it"
index=Index(emb,chunk_doc_ids)
print("abstracts:",len(corpus),"| chunks:",len(texts),"| test questions:",len(qs))
print("mean relevant abstracts per question:",round(float(np.mean([len(q["relevant_ids"]) for q in qs])),3))
tok,enc=load_encoder(device)
q_vecs=embed_texts([q["question"] for q in qs],tok,enc,device)
rankings=[]
for i in range(0,len(qs),64):
    ids,_=index.search_docs(q_vecs[i:i+64],k=20)
    rankings.extend(ids.tolist())
relevants=[set(q["relevant_ids"]) for q in qs]
table4=retrieval_metrics(rankings,relevants,ks=(1,5,10))
for name,value in table4.items():
    print(f"{name:10s} {value:.4f}")
ks=list(range(1,21))
recall_curve=[float(np.mean([recall_at_k(r,rel,k) for r,rel in zip(rankings,relevants)])) for k in ks]
plt.figure(figsize=(6,4))
plt.plot(ks,recall_curve,marker="o")
plt.xticks(ks[::2])
plt.ylim(0,1)
plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Retrieval: Recall@k on BioASQ test questions")
plt.grid(alpha=0.3)
plt.savefig("../results/rag_recall_at_k.png",dpi=120,bbox_inches="tight")
per_question=[{"qid":q["qid"],"top20":r,"hit10":hit_at_k(r,rel,10)} for q,r,rel in zip(qs,rankings,relevants)]
json.dump({"table4":table4,"recall_curve":recall_curve,"per_question":per_question},open("../results/retrieval_results.json","w"))
print("questions with Hit@10 = 0:",sum(1 for p in per_question if p["hit10"]==0))

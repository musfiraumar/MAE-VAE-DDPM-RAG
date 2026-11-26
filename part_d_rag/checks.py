import sys
import numpy as np
import torch
sys.path.append("../starter")
from bioasq import load_corpus,load_questions
from chunking import build_chunks,chunk_words
from embedding import load_encoder,embed_texts
from search import Index
from metrics import retrieval_metrics
device="cuda" if torch.cuda.is_available() else "cpu"
corpus=load_corpus()
texts,doc_ids=build_chunks(corpus)
longest=max(len(t.split()) for t in texts)
print("longest chunk (words):",longest)
assert longest<=150
pairs=bad=0
for text in corpus.values():
    ch=chunk_words(text)
    for a,b in zip(ch[:-1],ch[1:]):
        pairs+=1
        bad+=a.split()[-30:]!=b.split()[:30]
print(f"overlap check: {pairs} consecutive pairs, {bad} violations")
assert bad==0
index_emb=np.load("../results/index/chunk_embeddings.npy")
assert index_emb.shape[0]==len(texts)
index=Index(index_emb,doc_ids)
assert len(index.doc_ids)==len(set(doc_ids.tolist()))
print("chunks per abstract are contiguous: OK")
norms=np.linalg.norm(index_emb,axis=1)
print(f"embedding norms: min {norms.min():.6f}  max {norms.max():.6f}")
assert np.allclose(norms,1.0,atol=1e-4)
tok,enc=load_encoder(device)
short="Aspirin inhibits platelet aggregation."
alone=embed_texts([short],tok,enc,device)[0]
padded=embed_texts([short,texts[0]],tok,enc,device)[0]
print("cosine(alone, padded):",float(alone@padded))
assert float(alone@padded)>0.9999
lens=[len(x) for x in tok(texts[:2000])["input_ids"]]
print(f"chunks longer than 256 tokens (first 2000): {np.mean(np.array(lens)>256):.1%}")
qs=load_questions("test",corpus)
other=list(corpus.keys())[:50]
rankings,relevants=[],[]
for q in qs:
    rel=set(q["relevant_ids"])
    rankings.append(sorted(rel)+[d for d in other if d not in rel])
    relevants.append(rel)
perfect=retrieval_metrics(rankings,relevants,ks=(1,5,10))
print("perfect retriever:",perfect)
assert all(abs(v-1.0)<1e-9 for v in perfect.values())

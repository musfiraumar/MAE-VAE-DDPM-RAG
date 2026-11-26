import numpy as np
def hit_at_k(ranked,relevant,k):
    return 1.0 if len(set(ranked[:k])&relevant)>=1 else 0.0
def recall_at_k(ranked,relevant,k):
    return len(set(ranked[:k])&relevant)/min(k,len(relevant))
def mrr_at_10(ranked,relevant):
    for rank,doc in enumerate(ranked[:10],start=1):
        if doc in relevant:
            return 1.0/rank
    return 0.0
def retrieval_metrics(rankings,relevants,ks=(1,5,10)):
    out={}
    for k in ks:
        out[f"Hit@{k}"]=float(np.mean([hit_at_k(r,rel,k) for r,rel in zip(rankings,relevants)]))
        out[f"Recall@{k}"]=float(np.mean([recall_at_k(r,rel,k) for r,rel in zip(rankings,relevants)]))
    out["MRR@10"]=float(np.mean([mrr_at_10(r,rel) for r,rel in zip(rankings,relevants)]))
    return out

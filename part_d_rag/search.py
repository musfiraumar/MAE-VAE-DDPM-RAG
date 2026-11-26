import numpy as np
class Index:
    def __init__(self,emb,chunk_doc_ids):
        self.emb=emb
        self.chunk_doc_ids=chunk_doc_ids
        change=np.r_[True,chunk_doc_ids[1:]!=chunk_doc_ids[:-1]]
        self.starts=np.flatnonzero(change)
        self.doc_ids=chunk_doc_ids[self.starts]
    def search_docs(self,q_vecs,k=20):
        scores=q_vecs@self.emb.T
        doc_scores=np.maximum.reduceat(scores,self.starts,axis=1)
        order=np.argsort(-doc_scores,axis=1)[:,:k]
        return self.doc_ids[order],np.take_along_axis(doc_scores,order,axis=1)
    def search_chunks(self,q_vec,k=5):
        scores=self.emb@q_vec
        order=np.argsort(-scores)[:k]
        return order,scores[order]

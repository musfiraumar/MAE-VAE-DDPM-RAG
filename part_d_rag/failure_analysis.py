import sys,json,random
import numpy as np
import torch
sys.path.append("../starter")
from bioasq import load_corpus,load_questions
from chunking import build_chunks
from embedding import load_encoder,embed_texts
from search import Index
from generation import load_generator,answer_question,cited_numbers,said_idk
device="cuda" if torch.cuda.is_available() else "cpu"
corpus=load_corpus()
qs=load_questions("test",corpus)
texts,chunk_doc_ids=build_chunks(corpus)
index=Index(np.load("../results/index/chunk_embeddings.npy"),chunk_doc_ids)
res=json.load(open("../results/retrieval_results.json"))["per_question"]
assert [r["qid"] for r in res]==[q["qid"] for q in qs]
fails=[i for i,r in enumerate(res) if not (set(r["top20"][:10])&set(qs[i]["relevant_ids"]))]
print(len(fails),"of",len(qs),"questions failed (no relevant abstract in the top 10)")
first,rest=[i for i in fails if i<100],[i for i in fails if i>=100]
rng=random.Random(0)
rng.shuffle(first)
rng.shuffle(rest)
chosen=(first+rest)[:3]
enc_tok,enc=load_encoder(device)
gen_tok,gen=load_generator(device)
def short(t,n=60):
    return " ".join(t.split()[:n])+" ..."
with open("../results/failure_cases.md","w") as f:
    for i in chosen:
        q=qs[i]
        qv=embed_texts([q["question"]],enc_tok,enc,device)
        ids,sc=index.search_docs(qv,k=len(index.doc_ids))
        ids,sc=ids[0].tolist(),sc[0]
        pos={d:r for r,d in enumerate(ids,start=1)}
        best_rank=min(pos[d] for d in q["relevant_ids"])
        idx,_=index.search_chunks(qv[0],k=5)
        passages=[texts[j] for j in idx]
        ans=answer_question(q["question"],passages,gen_tok,gen,device)
        f.write(f"## Question (test index {i}, qid {q['qid']})\n{q['question']}\n\n")
        f.write(f"**Reference answer:** {q['answer']}\n\n")
        f.write(f"**Best rank of a relevant abstract:** {best_rank} of {len(ids)}\n\n")
        for d in q["relevant_ids"][:2]:
            f.write(f"- relevant abstract {d} (rank {pos[d]}): {short(corpus[d])}\n")
        f.write("\n**Top 3 retrieved:**\n")
        for r in range(3):
            f.write(f"- abstract {ids[r]} (score {sc[r]:.3f}): {short(corpus[ids[r]],40)}\n")
        f.write(f"\n**Generator answer:** {ans}\n\ncited: {cited_numbers(ans)}  |  said IDK: {said_idk(ans)}\n\n")
        for k,j in enumerate(idx,start=1):
            f.write(f"- passage [{k}] (abstract {int(chunk_doc_ids[j])}): {short(texts[j],40)}\n")
        f.write("\n---\n\n")
print("written to ../results/failure_cases.md")

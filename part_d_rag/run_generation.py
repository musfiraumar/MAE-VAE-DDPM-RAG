import sys,json
import numpy as np
import torch
sys.path.append("../starter")
from bioasq import load_corpus,load_questions
from chunking import build_chunks
from embedding import load_encoder,embed_texts
from search import Index
from generation import load_generator,answer_question,cited_numbers,said_idk
device="cuda" if torch.cuda.is_available() else "cpu"
N_QUESTIONS,TOP_CHUNKS=100,5
corpus=load_corpus()
qs=load_questions("test",corpus)[:N_QUESTIONS]
texts,chunk_doc_ids=build_chunks(corpus)
emb=np.load("../results/index/chunk_embeddings.npy")
assert emb.shape[0]==len(texts)
index=Index(emb,chunk_doc_ids)
enc_tok,enc=load_encoder(device)
gen_tok,gen=load_generator(device)
with open("../results/rag_answers.jsonl","w") as f:
    for n,q in enumerate(qs,start=1):
        q_vec=embed_texts([q["question"]],enc_tok,enc,device)[0]
        idx,scores=index.search_chunks(q_vec,k=TOP_CHUNKS)
        retrieved=[{"passage_no":r+1,"chunk_index":int(i),"abstract_id":int(chunk_doc_ids[i]),"score":float(s),"text":texts[i]} for r,(i,s) in enumerate(zip(idx,scores))]
        answer=answer_question(q["question"],[p["text"] for p in retrieved],gen_tok,gen,device)
        rec={"qid":q["qid"],"question":q["question"],"reference_answer":q["answer"],"relevant_ids":q["relevant_ids"],"retrieved":retrieved,"answer":answer,"cited":cited_numbers(answer),"idk":said_idk(answer)}
        f.write(json.dumps(rec)+"\n")
        f.flush()
        print(f"[{n}/{N_QUESTIONS}] {q['question'][:70]}")

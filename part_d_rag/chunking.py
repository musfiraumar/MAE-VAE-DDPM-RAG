import numpy as np
def chunk_words(text,max_words=150,overlap=30):
    words=text.split()
    if not words:
        return []
    step=max_words-overlap
    chunks=[]
    start=0
    while True:
        piece=words[start:start+max_words]
        chunks.append(" ".join(piece))
        if start+max_words>=len(words):
            break
        start+=step
    return chunks
def build_chunks(corpus):
    texts,doc_ids=[],[]
    for doc_id,text in corpus.items():
        for c in chunk_words(text):
            texts.append(c)
            doc_ids.append(doc_id)
    return texts,np.array(doc_ids,dtype=np.int64)

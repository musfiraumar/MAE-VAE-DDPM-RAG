import re
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
GEN_NAME="Qwen/Qwen2.5-1.5B-Instruct"
NOTICE="For educational use only. Not medical advice."
IDK_SENTENCE="I don't know based on the provided passages."
SYSTEM_PROMPT=("You are a careful biomedical research assistant. Answer the question using ONLY the "+"numbered passages provided. After each statement, cite the passage number(s) that support "+"it in square brackets, for example [1] or [2][4]. Do not use any knowledge from outside the "+"passages. If the passages do not contain the answer, reply exactly: "+IDK_SENTENCE+" Keep the answer short and factual.")
IDK_PATTERNS=["i don't know","i do not know","do not contain","does not contain","doesn't contain","don't contain","not contain the answer","cannot be answered","no information"]
def load_generator(device):
    tok=AutoTokenizer.from_pretrained(GEN_NAME)
    dtype=torch.float16 if device=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(GEN_NAME,torch_dtype=dtype).to(device).eval()
    return tok,model
def build_messages(question,passages):
    context="\n\n".join(f"[{i+1}] {p}" for i,p in enumerate(passages))
    user=f"Passages:\n{context}\n\nQuestion: {question}"
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}]
@torch.no_grad()
def answer_question(question,passages,tok,model,device):
    prompt=tok.apply_chat_template(build_messages(question,passages),tokenize=False,add_generation_prompt=True)
    inputs=tok(prompt,return_tensors="pt").to(device)
    out=model.generate(**inputs,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0)
    new_tokens=out[0,inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens,skip_special_tokens=True).strip()
def cited_numbers(answer):
    nums=[]
    for group in re.findall(r"\[([\d,\s]+)\]",answer):
        for part in group.split(","):
            if part.strip().isdigit():
                nums.append(int(part))
    return nums
def said_idk(answer):
    a=answer.lower().replace("’","'")
    return any(p in a for p in IDK_PATTERNS)

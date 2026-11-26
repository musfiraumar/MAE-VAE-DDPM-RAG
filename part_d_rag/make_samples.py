import json,random
recs=[json.loads(line) for line in open("../results/rag_answers.jsonl")]
ret={r["qid"]:r for r in json.load(open("../results/retrieval_results.json"))["per_question"]}
idk=[r for r in recs if r["idk"]]
failed=[r for r in recs if ret[r["qid"]]["hit10"]==0]
print("IDK answers:",len(idk),"| retrieval failures among the 100:",len(failed))
chosen=[]
if idk:
    chosen.append(idk[0])
for r in failed:
    if r not in chosen:
        chosen.append(r)
        break
others=[r for r in recs if r not in chosen]
chosen+=random.Random(0).sample(others,5-len(chosen))
with open("../results/samples.md","w") as f:
    f.write("> For educational use only. Not medical advice.\n\n")
    for n,r in enumerate(chosen,start=1):
        tags=[]
        if ret[r["qid"]]["hit10"]==0:
            tags.append("retrieval failed")
        if r["idk"]:
            tags.append("model said it does not know")
        f.write(f"## Sample {n} {'('+', '.join(tags)+')' if tags else ''}\n\n")
        f.write(f"**Question:** {r['question']}\n\n**Answer:** {r['answer']}\n\n**Cited passages:**\n\n")
        for num in sorted(set(r["cited"])):
            if 1<=num<=len(r["retrieved"]):
                p=r["retrieved"][num-1]
                ok="relevant" if p["abstract_id"] in r["relevant_ids"] else "not relevant"
                f.write(f"- [{num}] abstract {p['abstract_id']} ({ok}): {p['text']}\n")
        f.write(f"\n**Reference answer:** {r['reference_answer']}\n\n---\n\n")

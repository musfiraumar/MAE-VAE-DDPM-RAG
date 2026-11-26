import json,re
from rouge_score import rouge_scorer
recs=[json.loads(line) for line in open("../results/rag_answers.jsonl")]
scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
scores=[]
for r in recs:
    clean=re.sub(r"\[[\d,\s]+\]","",r["answer"])
    scores.append(scorer.score(r["reference_answer"],clean)["rougeL"].fmeasure)
rouge_l=sum(scores)/len(scores)
correct=total=no_cite=0
for r in recs:
    relevant=set(r["relevant_ids"])
    nums=set(r["cited"])
    if not nums:
        no_cite+=1
    for n in nums:
        total+=1
        if 1<=n<=len(r["retrieved"]) and r["retrieved"][n-1]["abstract_id"] in relevant:
            correct+=1
precision=correct/total if total else float("nan")
idk_rate=sum(r["idk"] for r in recs)/len(recs)
print(f"ROUGE-L F             : {rouge_l:.4f}")
print(f"citation precision    : {precision:.4f}  ({correct}/{total} cited passages)")
print(f"'I don't know' rate   : {idk_rate:.2%}  ({sum(r['idk'] for r in recs)}/{len(recs)})")
print(f"answers with no citation: {no_cite}/{len(recs)}")
json.dump({"rougeL":rouge_l,"citation_precision":precision,"idk_rate":idk_rate,"no_citation_answers":no_cite},open("../results/generation_metrics.json","w"),indent=2)

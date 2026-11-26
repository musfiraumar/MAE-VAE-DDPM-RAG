import sys,csv,json,random,os
SEED=2026
recs=[json.loads(line) for line in open("../results/rag_answers.jsonl")]
picked=sorted(random.Random(SEED).sample(range(len(recs)),20))
mode=sys.argv[1]
if mode=="make":
    with open("../results/rag_human_eval_sheet.md","w") as f:
        for n,i in enumerate(picked,start=1):
            r=recs[i]
            f.write(f"## Item {n}  (qid {r['qid']})\n\n**Question:** {r['question']}\n\n")
            for p in r["retrieved"]:
                f.write(f"[{p['passage_no']}] (abstract {p['abstract_id']}) {p['text']}\n\n")
            f.write(f"**Model answer:** {r['answer']}\n\n**Reference answer:** {r['reference_answer']}\n\n---\n\n")
    for m in (1,2):
        with open(f"../results/rag_marks_member{m}.csv","w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["qid","supported","correct"])
            for i in picked:
                w.writerow([recs[i]["qid"],"",""])
    print("sheet written; each member fills supported/correct with 1 or 0 in their own CSV")
elif mode=="merge":
    def read(path):
        return {int(r["qid"]):r for r in csv.DictReader(open(path))}
    m1,m2=read("../results/rag_marks_member1.csv"),read("../results/rag_marks_member2.csv")
    path="../results/human_eval.csv"
    rows=list(csv.DictReader(open(path))) if os.path.exists(path) else []
    for r in rows:
        r.setdefault("part","C")
    rows=[r for r in rows if r["part"]!="D"]
    for i in picked:
        q=recs[i]["qid"]
        rows.append({"part":"D","qid":q,"supported_member1":m1[q]["supported"],"supported_member2":m2[q]["supported"],"correct_member1":m1[q]["correct"],"correct_member2":m2[q]["correct"]})
    fields=["part","id","true_class","member1_guess","member2_guess","member1_correct","member2_correct","agree","qid","supported_member1","supported_member2","correct_member1","correct_member2"]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,restval="")
        w.writeheader()
        w.writerows(rows)
    d=[r for r in rows if r["part"]=="D"]
    for key in ("supported_member1","supported_member2","correct_member1","correct_member2"):
        print(f"{key}: {sum(int(r[key]) for r in d if r[key].strip()!='')}/20")

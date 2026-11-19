"""(10-19) PLM-FGQA (PerceptionLM fine-grained human video QA) -> stage-1 manifest, the way LLaVA-Video rows are written:
one row per downloaded segment, its questions (up to 5, open answers as given) as turns, the evidence placeholder in the first.
Re-run as the download (logs/zz_fgqa_dl2.sh) progresses; only segments whose prep file exists are written.
-> manifests/stage1/fgqa.parquet  [video (relative to prep/fgqa_384_15fps), prep_exists, kind, q_types, n_turns, conversations]"""
import sys, os, json, random, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training")
import pandas as pd
from jepa_llm.data.chat_format import VIS
R = "/data/llama_vjepa_lm"; PREP = f"{R}/prep/fgqa_384_15fps"
d = pd.read_parquet(f"{R}/data/PLM-Video-Human/fgqa/plm_fgqa_train.parquet", columns=["segment_id", "question", "answer", "metadata"])
meta = list(d.metadata.map(dict)); d["src"] = [m["source_dataset"] for m in meta]; d["q_type"] = [m.get("q_type", "") for m in meta]
d = d[d.src.isin(["coin", "youcook2", "crosstask", "ht100m"])]
have = {s: {f[:-4] for f in os.listdir(f"{PREP}/{s}") if f.endswith(".mp4")} for s in os.listdir(PREP)} if os.path.isdir(PREP) else {}
d = d[[sid in have.get(s, ()) for s, sid in zip(d.src, d.segment_id)]]
rng = random.Random(0); rows = []
for (s, sid), g in d.groupby(["src", "segment_id"]):
    qa = list(zip(g.question, g.answer, g.q_type)); rng.shuffle(qa)
    for k in range(0, len(qa), 5):
        part = qa[k:k + 5]; conv = []
        for j, (q, a, _) in enumerate(part):
            conv += [{"role": "user", "content": (VIS + "\n" if j == 0 else "") + str(q).strip()}, {"role": "assistant", "content": str(a).strip()}]
        rows.append({"video": f"{s}/{sid}.mp4", "prep_exists": True, "kind": "fgqa", "source": s, "q_types": ",".join(sorted({t for *_, t in part})),
                     "n_turns": len(part), "conversations": json.dumps(conv)})
out = pd.DataFrame(rows); out.to_parquet(f"{R}/manifests/stage1/fgqa.parquet")
print("rows", len(out), "segments", out.video.nunique(), "questions", int(out.n_turns.sum()), "by source", out.source.value_counts().to_dict())
qt = collections.Counter(t for x in out.q_types for t in x.split(",")); print("question types (rows containing)", qt.most_common(12))

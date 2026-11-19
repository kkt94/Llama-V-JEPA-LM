"""(10-19) PLM-FGQA download list: one line per YouTube source video with the segments its questions use.
Order: coin, youcook2, crosstask (small), then ht100m by number of questions per video (most first). Ego4D / EgoExo4D are left out
(license + credentials). -> dl/fgqa/list.tsv  [source, video_id, n_questions, segments "seg_id:start:end,..."]"""
import pandas as pd, collections
d = pd.read_parquet("/data/llama_vjepa_lm/data/PLM-Video-Human/fgqa/plm_fgqa_train.parquet", columns=["metadata", "segment_id"])
M = pd.DataFrame(list(d.metadata.map(dict))); M["segment_id"] = d.segment_id
M = M[M.source_dataset.isin(["coin", "youcook2", "crosstask", "ht100m"])]
nq = M.groupby("source_video_id").size()
seg = M.drop_duplicates("segment_id")
rows = []
for (src, vid), g in seg.groupby(["source_dataset", "source_video_id"]):
    rows.append((src, vid, int(nq[vid]), ",".join(f"{r.segment_id}:{r.source_start_time:.3f}:{r.source_end_time:.3f}" for r in g.itertuples())))
order = {"coin": 0, "youcook2": 1, "crosstask": 2, "ht100m": 3}
rows.sort(key=lambda r: (order[r[0]], -r[2]))
import os; os.makedirs("/data/llama_vjepa_lm/dl/fgqa", exist_ok=True)
with open("/data/llama_vjepa_lm/dl/fgqa/list.tsv", "w") as f:
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
c = collections.Counter(r[0] for r in rows); q = collections.Counter(); [q.__setitem__(r[0], q[r[0]] + r[2]) for r in rows]
print("videos", dict(c), "questions", dict(q), "segments", sum(len(r[3].split(",")) for r in rows))

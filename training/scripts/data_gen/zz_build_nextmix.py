"""(10-21) manifests for the next run (fixed mix):
 llava_clean   llava minus every row whose video id is also a public-benchmark video (Charades in MVBench-STAR / TVBench / FAVOR)
 anetqa        LLaVA-Video-178K ActivityNet-QA (open-ended) rows whose video is already in prep/llava_384_15fps
 vidln_uvo     ViDLN UVO sparse + dense narratives (VideoChat-Flash), vidln_ovis (frame folders)
 htstep        HT-Step event count / relationship / understanding (VideoChat-Flash long-video subsets, frame folders)
 ocr_*_train   OCR sets; TextVQA: 1000 images held out -> manifests/eval/pilot/textvqa_ocr.jsonl (open answers, exact match)"""
import json, os, glob, random
import pandas as pd
R = "/data/llama_vjepa_lm"; M = f"{R}/manifests/stage1"; VIS = "<|jepa_evidence|>"
def vid(p): return os.path.splitext(os.path.basename(str(p)))[0]
def conv(c):
    out = []
    for k, t in enumerate(c):
        role = "user" if t["from"] in ("human", "user") else "assistant"
        v = t["value"].replace("<image>\n", "").replace("<image>", "").replace("<video>\n", "").replace("<video>", "").strip()
        if k == 0: v = f"{VIS}\n{v}"
        out.append({"role": role, "content": v})
    return json.dumps(out)
def save(rows, name):
    d = pd.DataFrame(rows); d.to_parquet(f"{M}/{name}.parquet"); print(name, len(d), "rows", d.video.nunique(), "videos", flush=True); return d
bench = set()
for f in glob.glob(f"{R}/manifests/eval/pilot/*.jsonl"):
    b = os.path.basename(f)[:-6]
    if "heldout" in b or b.startswith("videomme"): continue
    bench |= {vid(json.loads(l)["video"]) for l in open(f)}
ll = pd.read_parquet(f"{M}/llava.parquet"); keep = ~ll.video.map(vid).isin(bench)
print("llava rows", len(ll), "removed (benchmark videos)", int((~keep).sum()), "videos", ll[~keep].video.nunique())
ll[keep].reset_index(drop=True).to_parquet(f"{M}/llava_clean.parquet")
P = f"{R}/prep/llava_384_15fps"; anet = {vid(p): os.path.relpath(p, P) for p in glob.glob(P + "/academic_source/activitynet/**/*.mp4", recursive=True)}
rows = []
for f in sorted(glob.glob(f"{R}/data/LLaVA-Video-178K/*activitynetqa*/*.json")):
    for x in json.load(open(f)):
        k = vid(x["video"])
        if k in anet and k not in bench:
            rows.append({"video": anet[k], "prep_exists": True, "kind": "oe", "source": "anetqa", "n_turns": len(x["conversations"]) // 2, "conversations": conv(x["conversations"])})
save(rows, "anetqa")
A = f"{R}/data/VideoChat-Flash/annotations/video"; S = f"{R}/data/VideoChat-Flash/shortvideo_subset"; F = f"{R}/data/vcf_frames"
for name, ann, root in (("vidln_uvo", ["caption_vidln_uvo_sparse-vidln-uvo_sparse_train_6k.json", "caption_vidln_uvo_dense-vidln-uvo_dense_train_1k.json"],
                         [f"{S}/caption_vidln_uvo_sparse-vidln-uvo_sparse_train_6k", f"{S}/caption_vidln_uvo_dense-vidln-uvo_dense_train_1k"]),
                        ("vidln_ovis", ["caption_vidln_ovis-vidln-ovis_train_1k.json"], [f"{F}/caption_vidln_ovis-vidln-ovis_train_1k"])):
    rows = []
    for a, r in zip(ann, root):
        for x in json.load(open(f"{A}/{a}")):
            p = f"{r}/{x['video']}"
            if os.path.exists(p) and vid(p) not in bench:
                rows.append({"video": p, "prep_exists": True, "kind": "cap", "source": name, "n_turns": len(x["conversations"]) // 2, "conversations": conv(x["conversations"])})
    save(rows, name)
rows = []
for a, sub in (("htstep_eventcount-longvideo_annos-htstep_eventcount_2k_2k.json", "htstep_eventcount_2k"), ("htstep_eventrelationship-longvideo_annos-htstep_eventrelationship_1k_1k.json", "htstep_eventrelationship_1k"),
               ("htstep_eventunderstanding-longvideo_annos-htstep_eventunderstanding_1k_1k.json", "htstep_eventunderstanding_1k")):
    for x in json.load(open(f"{A}/{a}")):
        p = f"{F}/{sub}/{x['video']}"
        if os.path.isdir(p):
            rows.append({"video": p, "prep_exists": True, "kind": sub.split("_")[1], "source": "htstep", "n_turns": len(x["conversations"]) // 2, "conversations": conv(x["conversations"])})
save(rows, "htstep")
t = pd.read_parquet(f"{M}/ocr_textvqa.parquet"); random.seed(0); ho = set(random.sample(sorted(t.video.unique()), 1000))
t[~t.video.isin(ho)].reset_index(drop=True).to_parquet(f"{M}/ocr_textvqa_train.parquet")
ev = []
for _, r in t[t.video.isin(ho)].iterrows():
    c = json.loads(r.conversations)
    for k in range(0, len(c), 2):
        q = c[k]["content"].replace(VIS, "").strip(); a = c[k + 1]["content"].strip()
        ev.append({"bench": "textvqa_ocr", "id": f"tvqa_{r.video}_{k // 2}", "video": f"{R}/data/ocr/textvqa/img/{r.video}", "question": q, "options": None,
                   "answer_index": None, "answer_text": a, "group_id": f"tvqa_{r.video}_{k // 2}", "split": "holdout", "start_sec": None, "end_sec": None, "extra": {"image": True}})
open(f"{R}/manifests/eval/pilot/textvqa_ocr.jsonl", "w").writelines(json.dumps(x) + "\n" for x in ev)
print("textvqa train rows", int((~t.video.isin(ho)).sum()), "| eval questions", len(ev))

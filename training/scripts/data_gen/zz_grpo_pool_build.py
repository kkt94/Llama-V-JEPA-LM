"""(10-30 13:10) GRPO question pool for the next run (on the round-3 SFT 2B): multiple-choice items with VERIFIED answers (human annotations or
simulator ground truth) from our training manifests, balanced over reasoning TYPES, eval videos excluded (basename match against every eval manifest).
First MC turn of each row; clip = prep_root/video (training copy), t0/t1. Types and sources:
  direction  object / camera / hand / turn direction      tq4_odir/cdir/dir2_ssv2, tq4_cnat_sav, epic_turn, egocam_mc_epic, camera_clevrer, camera_chexo, v17_finemotion_dir__ssv2, egodex_sel
  speed      relative / changing speed, tempo              tq4_rspeed_ssv2/sav, tq4_dspeed_ssv2/sav, v17_finemotion_speed__ssv2, v16_tempo__epic
  order      first / last / before / after / sequence      tq4_order_ssv2, tq4_order_nat, chego_order, chego_seq, epic_order, epic_seq, holo_seq, fmt_before_*, fmt_after_*, v17_order2__charades, fmt_breakdown_*
  temporal   when in the clip                              loc_nat, loc_comp, v17_oops_loc3, v17_oops_half
  count      repetitions / numbers                         countix, ssv2_count, rep_loop, v15_ndistinct_k700_act
  state      end state / completion / success / mistake    tq4_state_ssv2, chego_state, chego_done, ssv2_real, holo_mist, robo_success_egodex
  physics    collisions / prediction / counterfactual      clevrer, phys_fastest, phys_first, phys_predict, nv_side, plausibility
  predict    what happens next                             next_action_holo, next_turn_epic, star_train
  general    PerceptionTest-style QA                       pt_train_mc
  narrative  (optional, flagged) Video-Holmes TRAIN split (233 films, 0 overlap with its test films)
Per type at most CAP items, spread evenly over its sources. -> runs/cot/grpo_pool.jsonl"""
import json, re, os, glob, random, collections, sys
import pandas as pd, yaml
R = "/data/llama_vjepa_lm"; CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 6000; rng = random.Random(0)
c = yaml.safe_load(open(f"{R}/training/configs/single/nxs14_Q.yaml")); DS = {s["name"]: s for s in c["data"]["sources"]}
TYPES = {
 "direction": ["tq4_odir_ssv2", "tq4_cdir_ssv2", "tq4_dir2_ssv2", "tq4_cnat_sav", "epic_turn", "egocam_mc_epic", "camera_clevrer", "camera_chexo", "v17_finemotion_dir__ssv2", "egodex_sel"],
 "speed": ["tq4_rspeed_ssv2", "tq4_rspeed_sav", "tq4_dspeed_ssv2", "tq4_dspeed_sav", "v17_finemotion_speed__ssv2", "v16_tempo__epic"],
 "order": ["tq4_order_ssv2", "tq4_order_nat", "chego_order", "chego_seq", "epic_order", "epic_seq", "holo_seq", "fmt_before_chego", "fmt_before_epic", "fmt_before_holo",
           "fmt_after_chego", "fmt_after_epic", "fmt_after_holo", "v17_order2__charades", "fmt_breakdown_epic", "fmt_breakdown_holo"],
 "temporal": ["loc_nat", "loc_comp", "v17_oops_loc3", "v17_oops_half"],
 "count": ["countix", "ssv2_count", "rep_loop", "v15_ndistinct_k700_act"],
 "state": ["tq4_state_ssv2", "chego_state", "chego_done", "ssv2_real", "holo_mist", "robo_success_egodex"],
 "physics": ["clevrer", "phys_fastest", "phys_first", "phys_predict", "nv_side", "plausibility"],
 "predict": ["next_action_holo", "next_turn_epic", "star_train"],
 "general": ["pt_train_mc"]}
ev = {os.path.splitext(os.path.basename(json.loads(l)["video"]))[0] for f in glob.glob(f"{R}/manifests/eval/pilot/*.jsonl") for l in open(f) if json.loads(l).get("video")}
MC = re.compile(r"^(.*?)\n((?:[A-J]\. .*\n?)+)Answer with the option's letter", re.S)
def first_mc(conv):
    c0 = json.loads(conv) if isinstance(conv, str) else conv
    if len(c0) < 2: return None
    u = c0[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c0[1]["content"].strip(); m = MC.match(u)
    if not m or not re.match(r"^[A-J]\.?$", a): return None
    opts = [re.sub(r"^[A-J]\. ", "", l) for l in m.group(2).strip().split("\n")]; g = "ABCDEFGHIJ".index(a[0])
    return (m.group(1).strip(), opts, g) if g < len(opts) and len(opts) >= 2 else None
out, stat = [], collections.Counter()
for ty, srcs in TYPES.items():
    srcs = [s for s in srcs if s in DS]; per = max(1, CAP // len(srcs)); got = []
    for s in srcs:
        d = pd.read_parquet(DS[s]["manifest"]); rows = d.to_dict("records"); rng.shuffle(rows); n0 = len(got)
        for r in rows:
            if len(got) - n0 >= per: break
            if os.path.splitext(os.path.basename(str(r["video"])))[0] in ev: stat["eval_excluded"] += 1; continue
            if "prep_exists" in r and not bool(r["prep_exists"]): continue
            x = first_mc(r["conversations"])
            if not x: continue
            q, opts, g = x
            got.append({"type": ty, "src": s, "video": r["video"], "prep_root": DS[s].get("prep_root"), "t0": r.get("t0"), "t1": r.get("t1"), "question": q, "options": opts, "gold": g})
        stat[f"{ty}:{s}"] = len(got) - n0
    out += got; print(ty, len(got), flush=True)
h = json.load(open(f"{R}/bench/Video-Holmes/train_Video-Holmes.json")); hv = f"{R}/bench/Video-Holmes/videos_cropped"; nh = 0
for r in h:
    ks = sorted(r["Options"]); 
    if ks != list("ABCDEFGHIJ"[:len(ks)]) or r["Answer"] not in ks or not os.path.exists(f"{hv}/{r['video ID']}.mp4"): continue
    out.append({"type": "narrative", "src": "holmes_train", "video": f"{r['video ID']}.mp4", "prep_root": hv, "t0": None, "t1": None, "question": r["Question"], "options": [r["Options"][k] for k in ks], "gold": ks.index(r["Answer"])}); nh += 1
print("narrative (holmes train, optional)", nh)
pd.DataFrame(out).to_json(f"{R}/runs/cot/grpo_pool.jsonl", orient="records", lines=True)
print("total", len(out), "| eval-excluded", stat["eval_excluded"]); [print(" ", k, v) for k, v in sorted(stat.items()) if ":" in k]

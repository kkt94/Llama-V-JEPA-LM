"""One-line summary of a light-set scores.json (rough per-checkpoint performance)."""
import json, sys
s = json.load(open(sys.argv[1]))
g = lambda b: s.get(f"light:{b}", {})
mvp, ip, tb, pt = g("mvp_mini"), g("intphys2"), g("temporalbench_short_qa"), g("perception_test")
f = lambda v: "-" if v is None else f"{v:.1f}"
ipacc = (ip.get("main", {}).get("all", {}) or {}).get("accuracy") if ip else None
ptacc = pt.get("top1", pt.get("accuracy")) if pt else None
if isinstance(ptacc, dict): ptacc = ptacc.get("all")
parts = [f"MVP pair/single {f(mvp.get('pair_accuracy'))}/{f(mvp.get('single_accuracy'))}", f"IntPhys2 {f(ipacc)}",
         f"TB bin/MBA {f(tb.get('binary_accuracy'))}/{f(tb.get('multiple_binary_accuracy'))}", f"PT {f(ptacc)}"]
for n in ("plaus", "use_pairs_ssv2", "use_pairs_lt", "use_pairs_clevrer"):
    h = g(f"{n}_heldout")
    parts.append(f"{n.replace('use_pairs_', '')} {f(h.get('single_accuracy'))}/{f(h.get('pair_accuracy'))}")
print(" | ".join(parts))

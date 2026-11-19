"""(10-29, CoT stage data A) rationales for runs/cot/A_items.jsonl written by a text LLM (vLLM, one GPU) from the item's ground-truth facts.
Teacher: Qwen3-8B, no-think template, T 0.7 / top-p 0.8. Kept only if the text ends with 'Answer: <gold letter>' and does not leak the
annotation wording (facts / label / annotation / clip / metadata). Resumable (item index), chunks of 1024.
usage: <vllm python> -I zz_cot_a_gen.py <k> <n> [limit]   -> runs/cot/A_rationales_<k>of<n>.jsonl"""
import sys, os, json, re


def main():
    R = "/data/llama_vjepa_lm"; k, n = int(sys.argv[1]), int(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    items = [json.loads(l) for l in open(f"{R}/runs/cot/A_items.jsonl")]
    idx = list(range(k, len(items), n))[:lim]
    out = f"{R}/runs/cot/A_rationales_{k}of{n}.jsonl"
    done = {json.loads(l)["i"] for l in open(out)} if os.path.exists(out) else set()
    idx = [i for i in idx if i not in done]
    from transformers import AutoTokenizer
    M = f"{R}/models/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
    tok.chat_template = open(f"{R}/training/configs/templates/qwen3_nothink.jinja").read()
    L = "ABCDEFGHIJ"
    def prompt(it):
        opts = "\n".join(f"{L[j]}. {o}" for j, o in enumerate(it["options"]))
        u = ("You write the short reasoning a careful viewer gives after watching a video, before answering a multiple-choice question.\n\n"
             f"What really happens in the video (ground truth):\n" + "\n".join("- " + f for f in it["facts"]) + "\n\n"
             f"Question: {it['question']}\n{opts}\n\nCorrect answer: {L[it['gold']]}\n\n"
             "Write 1-3 sentences of step-by-step reasoning as the viewer. Only restate what the ground truth says, phrased as what is seen in "
             "the video (what happens, in which order, which direction, how many times), then say which option this matches and, if useful, "
             "why the other options do not match those events. Do NOT add any other objects, actions, causes, places, colours or descriptions. "
             "Do not mention the ground truth, annotations, labels, clips or that you were told the answer. End with a final line exactly: "
             "Answer: " + L[it["gold"]])
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    print(f"A gen {k}/{n}: {len(idx)} items to write ({len(done)} done)", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=M, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=k)
    sp = SamplingParams(temperature=0.3, top_p=0.8, top_k=20, max_tokens=300, seed=k)
    bad = re.compile(r"ground truth|annotation|label|\bclip|metadata|was told|i was given", re.I)
    GEN = set("""video shows show shown seen see sees first then next after before finally begins begin starts start ends end later earlier while
      during moves move moving moved direction left right upward downward upwards downwards toward towards away from into onto across over under
      camera person someone object objects thing something option options answer correct therefore which since because matches match matching
      does doesn other others rule rules ruled happens happen happened order sequence times time count number once twice three four five
      this that these those there their they them with without only also both each step steps action actions event events question asks
      describes described description clearly visible appears appear viewer watching watch performed performs perform performing being been
      would could should will none same different instead rather than does done what when where how many much does into back forth
      seconds second longer location position positioned present scenario continues continue continuing movement motion indicating indicates
      until remains remain stays stay frame located visible view scene shown entire throughout beginning middle final last initially
      hand hands gets goes comes takes puts picks holds places uses opens closes turns moves moving along further still again now
      consistent described describes indicated showing shows depicts involves involving performed occurs occur occurring""".split())
    with open(out, "a") as f:
        for s in range(0, len(idx), 1024):
            ch = idx[s:s + 1024]
            for i, o in zip(ch, llm.generate([prompt(items[i]) for i in ch], sp)):
                t = o.outputs[0].text.strip(); it = items[i]
                m = re.search(r"Answer:\s*([A-J])\s*$", t)
                allowed = set(re.findall(r"[a-z]{4,}", (it["question"] + " " + " ".join(it["options"]) + " " + " ".join(it["facts"])).lower())) | GEN
                novel = [w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in allowed and w.rstrip("s") not in allowed and w + "s" not in allowed]
                ok = bool(m) and m.group(1) == L[it["gold"]] and not bad.search(t) and o.outputs[0].finish_reason == "stop" and len(set(novel)) <= 3
                f.write(json.dumps({"i": i, "src": it["src"], "ok": ok, "novel": sorted(set(novel)), "text": t}) + "\n")
            f.flush(); print(f"A gen {k}: {min(s + 1024, len(idx))}/{len(idx)}", flush=True)
    print("A gen DONE", flush=True)


if __name__ == "__main__":
    main()

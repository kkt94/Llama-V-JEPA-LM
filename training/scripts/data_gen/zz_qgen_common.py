"""(10-24) Shared helpers for the open-data question builders (Charades-Ego, EPIC-KITCHENS-100, Kinetics-700, Countix): answers come from human
labels or from a known transformation (time reversal), never from a model. Rows in the stage-1 manifest format of manifests/stage1/tq4/*.parquet
(video, prep_exists, root2, tq, aug, t0, t1, conversations); conversations as tq4: '<|jepa_evidence|>' + question + lettered options + the
answer-letter instruction, 30% of items with a second turn whose options are reordered (the answer must follow the content, not a slot)."""
import json, numpy as np

TAIL = "\nAnswer with the option's letter from the given choices directly."
L = "ABCDEFGHIJ"


class Gen:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def pick(self, xs):
        return xs[int(self.rng.integers(len(xs)))]

    def shuffle(self, opts, k):
        """options with the answer at index k -> shuffled options, new answer index"""
        order = self.rng.permutation(len(opts)); return [opts[i] for i in order], int(np.nonzero(order == k)[0][0])

    def render(self, q, opts, k, second=None):
        second = (self.rng.random() < 0.3) if second is None else second
        u = f"<|jepa_evidence|>\n{q}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + TAIL
        conv = [{"role": "user", "content": u}, {"role": "assistant", "content": L[k] + "."}]
        if second and len(opts) > 1:
            p = self.rng.permutation(len(opts))
            while (p == np.arange(len(opts))).all(): p = self.rng.permutation(len(opts))
            o2 = [opts[i] for i in p]; k2 = int(np.nonzero(p == k)[0][0])
            conv += [{"role": "user", "content": self.pick(["Same question, options in another order:", "Again, with the options reordered:", "Once more:"])
                      + f"\n{q}\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(o2)) + TAIL}, {"role": "assistant", "content": L[k2] + "."}]
        return json.dumps(conv)

    def row(self, video, t0, t1, q, opts, k, aug="", exists=True):
        opts, k = self.shuffle(opts, k)
        return dict(video=video, prep_exists=bool(exists), root2=None, tq=None, aug=aug, t0=float(t0), t1=float(t1), conversations=self.render(q, opts, k))


def report(name, rows):
    import collections
    ans = collections.Counter(json.loads(r["conversations"])[1]["content"] for r in rows)
    print(f"{name}: {len(rows)} rows | answer letters {dict(sorted(ans.items()))} | aug {dict(collections.Counter(r['aug'] for r in rows))}", flush=True)
    if rows: print("   example:", rows[0]["conversations"][:330].replace("\n", " / "), flush=True)


def count_options(n, rng, lo=0, width=4, k=4):
    """(10-25, nx5 audit) k distinct counts with n at a uniformly drawn rank among them (wrong counts within +-width of n, >= lo). The old
    'nearby wrong counts' (Countix: c-2, c-1, c+1, c+2, c+3, 2c; SSv2: 1..8 within 3) put the gold at rank 2 of 4 in 55-59 % of the rows and
    almost never largest - nx5 learned that shortcut (MotionBench repetition count picks: rank 2 53 %, largest 0 %). -> [str(n), wrong...]"""
    below = list(range(max(lo, n - width), n)); above = list(range(n + 1, n + width + 1))
    ranks = [r for r in range(k) if r <= len(below) and k - 1 - r <= len(above)]
    r = int(rng.choice(ranks))
    vals = list(rng.choice(below, r, replace=False)) + list(rng.choice(above, k - 1 - r, replace=False))
    return [str(n)] + [str(int(v)) for v in vals]

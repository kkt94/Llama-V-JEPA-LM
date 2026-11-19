"""Turn a conversation into LLM inputs with a block of visual evidence positions and answer-only labels (spec 4.2).

- The chat template is rendered as text with a placeholder string where the evidence goes; the text on each side is
  tokenised with add_special_tokens=False (the Llama template already emits <|begin_of_text|>; letting the tokenizer
  add it again produces a double BOS).
- Evidence positions get a dummy token id and are flagged in `vis_mask`; the model overwrites their embeddings.
- Labels cover exactly each assistant turn's span as rendered by the template (content + end-of-turn token), found by
  rendering message prefixes. Templates that are not prefix-stable are rejected rather than guessed.
- Over-long conversations drop trailing turns (whole QA pairs), never cut an answer.
"""
import os

VIS = "<|jepa_evidence|>"
IGNORE = -100


class TemplateError(ValueError):
    pass


def setup_tokenizer(tok, model_cfg):
    """(10-20) model.chat_template: a jinja file that replaces the tokenizer's template. Qwen3's own template renders the empty
    <think></think> block only on the LAST assistant turn, so it is not prefix-stable for multi-turn data; the file used for Qwen3
    (configs/templates/qwen3_nothink.jinja) gives every assistant turn the non-thinking block (the format of enable_thinking=False)."""
    path = (model_cfg or {}).get("chat_template")
    if path:
        tok.chat_template = open(path).read()
    return tok


def _render(tok, messages, gen, chat_kwargs):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, **chat_kwargs)


def _assistant_spans(tok, messages, full, chat_kwargs):
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        before = _render(tok, messages[:i], True, chat_kwargs)
        through = _render(tok, messages[: i + 1], False, chat_kwargs)
        if not (through.startswith(before) and full.startswith(through)):
            raise TemplateError("chat template is not prefix-stable; cannot locate assistant spans")
        spans.append((len(before), len(through)))
    return spans


def place_vis(content: str) -> str:
    """JEPA_VIS_LAST=1 (config data.vis_last): the evidence placeholder goes to the END of its user turn, so the question /
    instruction is read before the video and the LLM's own layers process the video tokens knowing what is asked
    (default: placeholder where the data put it, i.e. before the question)."""
    if os.environ.get("JEPA_VIS_LAST") == "1" and VIS in content:
        rest = content.replace(VIS, "").strip()
        return f"{rest}\n{VIS}" if rest else VIS
    return content


def build_example(tok, messages, n_vis: int, max_len: int, chat_kwargs=None):
    """messages: [{"role": "user"|"assistant"|"system", "content": str}], VIS appears exactly once (first user turn).
    Returns dict(input_ids, labels, vis_mask, n_turns) or None if even the first QA pair does not fit."""
    if os.environ.get("JEPA_MEMORY") == "1":     # (10-15) video read as memory by every layer: no video slots in the sequence
        n_vis = 0
    chat_kwargs = chat_kwargs or {}
    assert sum(m["content"].count(VIS) for m in messages) == 1, "exactly one evidence placeholder expected"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # (10-23) packed rows: an assistant message may carry "seg" = the original row it came from; every answer token gets that id
    # ("seg" output, -1 outside answers) so the loss can average per original row. Rendered without the key.
    segs_all = [int(m.get("seg", 0)) for m in messages if m["role"] == "assistant"]
    msgs = [{k: v for k, v in {**m, "content": place_vis(m["content"])}.items() if k != "seg"} for m in messages]
    if os.environ.get("JEPA_STRIP_THINK") == "1":   # (11-05) no thinking mode in the backbone (Llama): drop the think markers, keep the reasoning text
        msgs = [dict(m, content=m["content"].replace("\n</think>\n\n", "\n\n").replace("</think>", "").replace("<think>", "").strip()) if m["role"] == "assistant" else m for m in msgs]
    while msgs:
        full = _render(tok, msgs, False, chat_kwargs)
        spans = _assistant_spans(tok, msgs, full, chat_kwargs)
        pre, post = full.split(VIS)
        pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
        enc = tok(post, add_special_tokens=False, return_offsets_mapping=True)
        shift = len(pre) + len(VIS)
        post_labels, post_seg = [], []
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            a, b = a + shift, b + shift
            k = next((j for j, (s, e) in enumerate(spans) if b > a and s <= a and b <= e), None)
            post_labels.append(tid if k is not None else IGNORE); post_seg.append(segs_all[k] if k is not None else -1)
        ids = pre_ids + [pad] * n_vis + enc["input_ids"]
        if len(ids) <= max_len and any(x != IGNORE for x in post_labels):
            return {
                "input_ids": ids,
                "labels": [IGNORE] * (len(pre_ids) + n_vis) + post_labels,
                "vis_mask": [0] * len(pre_ids) + [1] * n_vis + [0] * len(enc["input_ids"]),
                "n_turns": sum(m["role"] == "assistant" for m in msgs),
                "seg": [-1] * (len(pre_ids) + n_vis) + post_seg,
            }
        # drop the last user/assistant pair and retry
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        if last_user == min(i for i, m in enumerate(msgs) if m["role"] == "user"):
            return None
        msgs = msgs[:last_user]
    return None


def build_example_multi(tok, messages, n_vis: int, max_len: int, chat_kwargs=None):
    """(10-15) as build_example, with one evidence placeholder in EACH user turn (several clips in one sequence, in order);
    every assistant turn is labelled. -> dict(input_ids, labels, vis_mask) or None if it does not fit."""
    if os.environ.get("JEPA_MEMORY") == "1":     # (10-15) video read as memory by every layer: no video slots in the sequence
        n_vis = 0
    chat_kwargs = chat_kwargs or {}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    msgs = [{**m, "content": place_vis(m["content"])} for m in messages]
    full = _render(tok, msgs, False, chat_kwargs)
    spans = _assistant_spans(tok, msgs, full, chat_kwargs)
    ids, labels, vis, pos = [], [], [], 0
    for k, part in enumerate(full.split(VIS)):
        if k:
            ids += [pad] * n_vis; labels += [IGNORE] * n_vis; vis += [1] * n_vis; pos += len(VIS)
        enc = tok(part, add_special_tokens=False, return_offsets_mapping=True)
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            a, b = a + pos, b + pos
            ids.append(tid); vis.append(0); labels.append(tid if b > a and any(s <= a and b <= e for s, e in spans) else IGNORE)
        pos += len(part)
    if len(ids) > max_len or all(x == IGNORE for x in labels):
        return None
    return {"input_ids": ids, "labels": labels, "vis_mask": vis, "n_turns": sum(m["role"] == "assistant" for m in msgs)}


SEM, LOC = "<|jepa_sem|>", "<|jepa_loc|>"


def build_pass1(tok, question: str, n_overview: int, chat_kwargs=None):
    """Stage-2 pass 1 (spec 4.1): user turn = question, then [SEM] [overview tokens] [LOC]. No labels.
    SEM sits before the overview, so under causal attention the semantic choice depends on the question only; LOC sees
    question and overview. Returns dict(input_ids, vis_mask, sem_mask, loc_mask)."""
    chat_kwargs = chat_kwargs or {}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    text = tok.apply_chat_template([{"role": "user", "content": f"{question}\n{SEM}{VIS}{LOC}"}], tokenize=False,
                                   add_generation_prompt=False, **chat_kwargs)
    a, rest = text.split(SEM)
    b, rest = rest.split(VIS)
    c, d = rest.split(LOC)
    ids, vis, sem, loc = [], [], [], []
    for piece, n_slot, kind in ((a, 1, "sem"), (b, n_overview, "vis"), (c, 1, "loc"), (d, 0, None)):
        t = tok(piece, add_special_tokens=False)["input_ids"]
        ids += t; vis += [0] * len(t); sem += [0] * len(t); loc += [0] * len(t)
        if n_slot:
            ids += [pad] * n_slot
            vis += [int(kind == "vis")] * n_slot; sem += [int(kind == "sem")] * n_slot; loc += [int(kind == "loc")] * n_slot
    return {"input_ids": ids, "vis_mask": vis, "sem_mask": sem, "loc_mask": loc}


def build_generation_prompt(tok, user_content: str, n_vis: int, chat_kwargs=None):
    """User turn with one evidence placeholder + the assistant header (add_generation_prompt), no labels.
    Returns dict(input_ids, vis_mask)."""
    chat_kwargs = chat_kwargs or {}
    if os.environ.get("JEPA_MEMORY") == "1":     # (10-15) video read as memory by every layer: no video slots in the sequence
        n_vis = 0
    assert user_content.count(VIS) == 1
    user_content = place_vis(user_content)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    text = tok.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False,
                                   add_generation_prompt=True, **chat_kwargs)
    pre, post = text.split(VIS)
    a = tok(pre, add_special_tokens=False)["input_ids"]
    b = tok(post, add_special_tokens=False)["input_ids"]
    return {"input_ids": a + [pad] * n_vis + b, "vis_mask": [0] * len(a) + [1] * n_vis + [0] * len(b)}

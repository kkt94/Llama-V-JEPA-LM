"""Video decoding / frame sampling and prompt building for JEPA-LLM (identical to the evaluation path of the model).

Video: `num_frames` (32) frame times at the slot centres of the clip, t_i = t0 + (i + 0.5) * (t1 - t0) / 32, rounded to the
nearest source frame; frames are resized (short side 384, bicubic with antialias, only if needed), centre-cropped to
384 x 384 and ImageNet-normalised. Frame times (seconds) are kept: the model receives them.
Prompt: one user turn "<evidence>\\n{question}" (multiple choice: options as "A. ...", then the instruction line), in the
chat template the model was trained with (CHAT_TEMPLATES[config.chat_template_name]; Qwen3: no system prompt; Llama 3: the
template's own system header with its fixed date line, no system text); the evidence positions are replaced by the 1024 video tokens.
"""
import numpy as np
import torch
import torch.nn.functional as F

VIS = "<|jepa_evidence|>"
IGNORE = -100
LETTERS = "ABCDEFGHIJ"
MC_SUFFIX = "Answer with the option's letter from the given choices directly."
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# the chat templates the models were trained and evaluated with (config.chat_template_name; no system prompt)
CHAT_TEMPLATES = {
    # plain Qwen3 turns, no thinking block (Qwen3-4B-Instruct-2507)
    "qwen3_instruct": (
        "{%- for message in messages -%}\n"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}\n"
    ),
    # Qwen3 hybrid-thinking models in non-thinking mode: every assistant turn starts with an empty think block (Qwen3-1.7B)
    "qwen3_nothink": (
        "{%- for message in messages -%}\n"
        "{{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
        "{%- if message['role'] == 'assistant' -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}\n"
        "{{- message['content'] + '<|im_end|>\\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}\n"
    ),
    # Llama 3.2 / 3.1 Instruct (Meta's templates; the 'Today Date' line is fixed to the template default "26 Jul 2024",
    # as in training and evaluation - no date_string is passed)
    "llama32_instruct": '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n',
    "llama31_instruct": '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n',
}
CHAT_TEMPLATE = CHAT_TEMPLATES["qwen3_instruct"]   # default (backward compatible name)


# ---- video ----------------------------------------------------------------------------------------------------------
def sample_times(duration: float, num_frames: int) -> np.ndarray:
    """uniform sampling: num_frames distinct times at the slot centres of [0, duration]"""
    return (np.arange(num_frames) + 0.5) * duration / num_frames


def load_video(path: str, num_frames: int = 32, size: int = 384, start_sec=None, end_sec=None):
    """-> (video [3, T, size, size] float32 ImageNet-normalised, frame_times [T] float32 seconds).
    start_sec / end_sec: optional visible span (seconds); a missing bound means the start / end of the video."""
    try:
        import decord
    except ImportError as e:  # pip install eva-decord (the `decord` module)
        raise ImportError("JEPA-LLM video loading needs decord: pip install eva-decord==0.6.1") from e
    vr = decord.VideoReader(path, num_threads=1)
    fps = float(vr.get_avg_fps())
    n = len(vr)
    if start_sec is not None or end_sec is not None:
        t0 = max(0.0, float(start_sec or 0.0))
        t1 = min(n / fps, float(end_sec) if end_sec is not None else n / fps)
        times = t0 + sample_times(max(t1 - t0, 0.0), num_frames)
    else:
        times = sample_times(n / fps, num_frames)
    idx = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, n - 1)
    frames = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy()).permute(3, 0, 1, 2).float() / 255   # [3, T, H, W]
    return preprocess_frames(frames, size), torch.as_tensor(idx / fps, dtype=torch.float32)


def preprocess_frames(frames: torch.Tensor, size: int = 384) -> torch.Tensor:
    """frames [3, T, H, W] in 0..1 -> short side `size` (bicubic, antialias; only if needed), centre crop, ImageNet norm."""
    h, w = frames.shape[-2:]
    if min(h, w) != size:
        s = size / min(h, w)
        frames = F.interpolate(frames.transpose(0, 1), size=(round(h * s), round(w * s)), mode="bicubic",
                               antialias=True, align_corners=False).clamp(0, 1).transpose(0, 1)
        h, w = frames.shape[-2:]
    x = int(min(max(round(0.5 * w - size / 2), 0), w - size))
    y = int(min(max(round(0.5 * h - size / 2), 0), h - size))
    frames = frames[:, :, y:y + size, x:x + size]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return (frames - mean) / std


# ---- prompts --------------------------------------------------------------------------------------------------------
def mc_prompt(question: str, options) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}\n{MC_SUFFIX}"


def letter_ids(tok, n: int):
    ids = [tok(LETTERS[i], add_special_tokens=False)["input_ids"] for i in range(n)]
    assert all(len(x) == 1 for x in ids), "option letters must be single tokens"
    return [x[0] for x in ids]


def _render(tok, messages, gen):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def build_scoring_example(tok, user_content: str, n_vis: int):
    """The user turn + the assistant answer "A", rendered by the chat template; labels mark the assistant span
    (as in training). The option scores are the logits at the position that predicts the FIRST label token.
    -> dict(input_ids, labels, vis_mask)"""
    assert user_content.count(VIS) == 1
    msgs = [{"role": "user", "content": user_content}, {"role": "assistant", "content": "A"}]
    full = _render(tok, msgs, False)
    before, through = _render(tok, msgs[:1], True), _render(tok, msgs, False)
    if not (through.startswith(before) and full.startswith(through)):
        raise ValueError("chat template is not prefix-stable")
    s, e = len(before), len(through)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pre, post = full.split(VIS)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    enc = tok(post, add_special_tokens=False, return_offsets_mapping=True)
    shift = len(pre) + len(VIS)
    post_labels = [tid if (b > a and s <= a + shift and b + shift <= e) else IGNORE
                   for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": pre_ids + [pad] * n_vis + enc["input_ids"],
            "labels": [IGNORE] * (len(pre_ids) + n_vis) + post_labels,
            "vis_mask": [0] * len(pre_ids) + [1] * n_vis + [0] * len(enc["input_ids"])}


def build_generation_prompt(tok, user_content: str, n_vis: int):
    """User turn with one evidence placeholder + the assistant header. -> dict(input_ids, vis_mask)"""
    assert user_content.count(VIS) == 1
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    text = _render(tok, [{"role": "user", "content": user_content}], True)
    pre, post = text.split(VIS)
    a = tok(pre, add_special_tokens=False)["input_ids"]
    b = tok(post, add_special_tokens=False)["input_ids"]
    return {"input_ids": a + [pad] * n_vis + b, "vis_mask": [0] * len(a) + [1] * n_vis + [0] * len(b)}

# Llama-V-JEPA-LM: Technical Report

**Video-language models that read dynamics beyond static, text-aligned representations.**

Llama-V-JEPA-LM joins a frozen **V-JEPA 2.1 world-model video encoder** to a language model (Qwen3 or Llama, 1B to 8B) and trains it in **one stage** with **counterfactual supervision**. The models read how a scene changes (direction, speed, order, camera motion, state) and lead every size class on eleven video benchmarks with **4.53M public training samples** and **32 visual tokens per frame**.

Weights: [Llama-V-JEPA-LM collection on Hugging Face](https://huggingface.co/collections/kkt94/llama-v-jepa-lm-6af41fb2a548b444c6255d63)

---

## 1. Highlights

- **First in every size class.** The 4B model averages **72.14** on twelve scores from eleven video benchmarks. The best open 4B model scores 61.08 and the best open 8B model 64.79. The 2B model averages **66.02** against 52.45 for the best open 2B model.
- **Strongest on dynamics.** The 4B model scores TOMATO 53.5 (open 4B models at most 36.5), Vinoground video 66.8 (at most 33.8) and SSv2 92.2 (at most 68.3).
- **Ahead of Meta's V-JEPA video LLMs.** On the six benchmarks Meta reports for its V-JEPA 2 / 2.1 video LLMs (Llama-3.1-8B), our 8B models average 70.0 and 68.2 against 63.3 and 62.8. They use a twentieth of the training data and a ninth of the visual tokens.
- **Generalizes beyond its training constructions.** On six held-out sets the models average 94.5–96.3. The sets cover unseen speed ratios, an unseen source corpus, unseen combinations and an unseen simulator. Open models score 43.5–55.5 (chance 33.3).
- **Efficient.** Training is one stage of 229–648 H100-hours. Each clip becomes 1,024 visual tokens at any resolution, so the first token takes 70–182 ms on one H100.

## 2. Model zoo

| Model | Language model | Encoder (frozen) | Total params | Avg (12 scores) | Held-out (6 sets) |
|---|---|---|---:|---:|---:|
| [Llama-V-JEPA-LM-Llama3.2-1B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Llama3.2-1B) | Llama-3.2-1B-Instruct | V-JEPA 2.1 ViT-L | 1.55 B | 62.21 | 94.5 |
| [Llama-V-JEPA-LM-Qwen3-1.7B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Qwen3-1.7B) | Qwen3-1.7B | V-JEPA 2.1 ViT-L | 2.03 B | 66.02 | 95.0 |
| [Llama-V-JEPA-LM-Llama3.2-3B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Llama3.2-3B) | Llama-3.2-3B-Instruct | V-JEPA 2.1 ViT-g | 4.25 B | 68.97 | 96.0 |
| [Llama-V-JEPA-LM-Qwen3-4B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Qwen3-4B) | Qwen3-4B-Instruct-2507 | V-JEPA 2.1 ViT-g | 5.06 B | **72.14** | **96.3** |
| [Llama-V-JEPA-LM-Llama3.1-8B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Llama3.1-8B) | Llama-3.1-8B-Instruct | V-JEPA 2.1 ViT-g | 9.07 B | 70.51 | 96.1 |
| [Llama-V-JEPA-LM-Qwen3-8B](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Qwen3-8B) | Qwen3-8B | V-JEPA 2.1 ViT-g | 9.23 B | 70.11 | 95.9 |

Each model card has a quick start, the full per-benchmark table of its size class, inference cost and licence details.

## 3. Idea

**Captions and QA do not make a model read dynamics.** World-model encoders such as V-JEPA learn how scenes change from video alone. They are usually joined to a language model with the image–text recipe: captions and question–answer pairs over several stages and many millions of samples. But a caption or question names only a few attributes of a clip, and matching them lowers the loss. The model therefore learns to fetch what the text asks about, not to read the change.

We measured this on held-out speed questions. A linear probe reads the answer from frozen V-JEPA features at 86.7%, and at 83.5% from the bridge of a model trained on captions and QA. That model still answers at chance (33.7%).

**Where dynamics lives in the encoder.** A frame-wise, text-aligned encoder such as SigLIP embeds each frame on its own. Reversing a clip only reorders its tokens, so the arrow of time is in no token's content. A predictive encoder behaves differently. Reversing, mirroring or re-timing a clip moves its tokens along a change direction that points the same way across videos, and it does so 20–100× more consistently for V-JEPA than for SigLIP.

**Counterfactual supervision** uses that structure. A clip and a transformed copy are asked the same question, and their answers differ by construction. Only the change can tell them apart, so scene content cancels from the answer. The rows come from pixel transformations of public videos, with no new annotation, and are added to an ordinary caption and QA mixture.

## 4. Training

**Architecture.**
- **Encoder:** frozen V-JEPA 2.1, ViT-L for the 1B and 2B models and ViT-g for 3B–8B. It reads 32 frames.
- **Bridge:** a spectral bridge maps each clip to 1,024 tokens. It applies a fixed PCA rotation and rescaling computed once from encoder statistics, then a trainable MLP.
- **Language model:** fully fine-tuned.

**One stage.** After a short bridge-only warm-up (0.6–0.8% of the steps), the bridge and the language model are trained together in a single pass. There are no separate alignment, pre-training or instruction stages. The pass covers 4.53M rows drawn at fixed per-source weights from 175 public sources (4.60M for the ViT-L models).

**Counterfactual data.** 33 sources and 509k rows, about 6% of the drawn samples, in six families:

| Family | Constructions |
|---|---|
| Sign flip | clip reversed or mirrored; camera move rendered or reversed; two clips, each optionally reversed |
| Order | two events joined in sequence or stacked; which segment matches a two-action caption |
| Position | several clips joined, one holds the queried action: beginning / middle / end |
| Magnitude | relative speed; acceleration / deceleration; slow motion vs. time-lapse |
| Count | one action repeated *k* times; number of distinct activities |
| Stop / cut | clip cut before the action completes; physically impossible simulator edits |

**Balancing.** Within each source every scene appears with each answer equally often in expectation. The generator also balances how long the correct option is relative to the others, and how likely each option looks to the text-only language model. It decorrelates answers from clip length and timing.

**Compute.**

| Model | H100s | Global batch | Steps | Samples | H100-hours |
|---|---:|---:|---:|---:|---:|
| Llama3.2-1B | 28 | 168 | 27,382 | 4.60M | 303 |
| Qwen3-1.7B | 14 | 168 | 27,382 | 4.60M | 229 |
| Llama3.2-3B | 30 | 300 | 15,094 | 4.53M | 429 |
| Qwen3-4B | 30 | 150 | 30,187 | 4.53M | 499 |
| Llama3.1-8B | 30 | 150 | 30,187 | 4.53M | 596 |
| Qwen3-8B | 30 | 150 | 30,187 | 4.53M | 648 |

Open video LLMs of these sizes use several stages and 9M–172M samples, with 64–256 visual tokens per frame.

## 5. Results

**Protocol.** All models, ours and open, are evaluated by the same harness:
- 32 frames per clip
- the same items and prompts
- option-letter scoring

**Scores.** MVP and the two Vinoground scores use pair accuracy, TemporalBench the binary split, and SSv2 multiple choice over action labels. Avg is the mean of the twelve scores; its 95% bootstrap interval is ±0.7 for every model. Bold marks the best in each class.

MVB: MVBench, PT: PerceptionTest, TVB: TVBench, TC: TempCompass, TOM: TOMATO, MoB: MotionBench, NQA: NExT-QA, MVP: Minimal Video Pairs, TB: TemporalBench, SSv2: Something-Something v2, VG-T / VG-V: Vinoground text / video.

### 2B class

| Model | MVB | PT | TVB | TC | TOM | MoB | NQA | MVP | TB | SSv2 | VG-T | VG-V | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-V-JEPA-LM-Qwen3-1.7B** | **69.5** | **75.3** | **70.7** | **73.5** | **50.9** | 54.3 | 74.0 | **54.6** | 63.9 | **91.6** | **63.2** | 50.8 | **66.02** |
| **Llama-V-JEPA-LM-Llama3.2-1B** | 66.3 | 71.9 | 67.9 | 71.6 | 48.2 | 49.5 | 70.2 | 48.5 | 57.8 | 89.6 | 51.2 | **53.8** | 62.21 |
| Qwen3-VL-2B | 61.0 | 65.1 | 49.8 | 67.4 | 30.3 | **57.5** | 77.1 | 32.3 | 65.9 | 59.2 | 36.2 | 27.6 | 52.45 |
| InternVL3.5-2B | 66.8 | 64.2 | 50.9 | 65.7 | 31.0 | 56.2 | 76.1 | 28.8 | **71.3** | 57.0 | 40.0 | 21.0 | 52.42 |
| PLM-1B | 68.0 | 69.8 | 48.2 | 60.5 | 24.5 | 51.7 | **79.8** | 30.9 | 58.1 | 61.7 | 33.2 | 29.6 | 51.33 |
| Apollo-1.5B | 61.1 | 63.9 | 47.9 | 60.3 | 27.8 | 51.1 | 78.8 | 26.0 | 55.8 | 56.0 | 38.8 | 22.4 | 49.16 |
| VideoLLaMA3-2B | 57.9 | 62.9 | 44.0 | 64.1 | 24.0 | 56.3 | 75.4 | 23.1 | 64.1 | 50.0 | 38.0 | 25.2 | 48.75 |
| Gemma 4 E2B | 39.8 | 46.3 | 36.4 | 51.5 | 18.7 | 41.1 | 51.4 | 8.1 | 61.4 | 37.0 | 24.0 | 22.4 | 36.51 |

### 4B class

| Model | MVB | PT | TVB | TC | TOM | MoB | NQA | MVP | TB | SSv2 | VG-T | VG-V | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-V-JEPA-LM-Qwen3-4B** | 72.1 | 78.2 | **74.8** | **79.9** | 53.5 | 59.1 | 79.3 | **58.1** | **75.7** | **92.2** | **76.0** | **66.8** | **72.14** |
| **Llama-V-JEPA-LM-Llama3.2-3B** | 70.7 | 74.3 | 70.3 | 76.5 | **55.9** | 54.2 | 77.0 | 56.3 | 67.3 | 91.3 | 71.0 | 62.8 | 68.97 |
| Molmo2-4B | **74.2** | **81.4** | 65.5 | 70.8 | 36.5 | **62.3** | **85.1** | 45.4 | 72.8 | 57.0 | 48.2 | 33.8 | 61.08 |
| PLM-3B | 73.0 | 76.0 | 56.6 | 67.0 | 36.3 | 58.9 | 84.0 | 46.2 | 66.2 | 68.3 | 55.0 | 31.4 | 59.91 |
| InternVL3.5-4B | 70.9 | 69.1 | 60.0 | 71.1 | 32.1 | 59.0 | 79.8 | 41.4 | 73.2 | 64.1 | 53.6 | 28.0 | 58.52 |
| Qwen3-VL-4B | 68.0 | 72.0 | 59.2 | 70.9 | 36.3 | 61.5 | 80.9 | 39.6 | 68.9 | 64.8 | 47.6 | 31.0 | 58.39 |
| Apollo-3B | 62.0 | 67.1 | 50.9 | 63.0 | 29.0 | 55.0 | 80.9 | 27.4 | 60.1 | 64.7 | 37.4 | 25.8 | 51.94 |
| Gemma 4 E4B | 46.5 | 52.0 | 41.5 | 57.4 | 22.8 | 49.4 | 64.7 | 16.5 | 61.0 | 43.8 | 26.4 | 24.0 | 42.17 |

### 8B class

| Model | MVB | PT | TVB | TC | TOM | MoB | NQA | MVP | TB | SSv2 | VG-T | VG-V | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-V-JEPA-LM-Llama3.1-8B** | 71.9 | 75.0 | 71.8 | 77.6 | 55.5 | 55.9 | 79.0 | 57.6 | 72.3 | 92.3 | 74.2 | **63.0** | **70.51** |
| **Llama-V-JEPA-LM-Qwen3-8B** | 72.8 | 77.8 | **75.3** | **79.4** | **56.5** | 59.1 | 81.2 | **58.1** | **73.5** | **93.8** | **75.2** | 38.6 | 70.11 |
| PLM-8B | **76.0** | 80.6 | 62.7 | 70.9 | 42.6 | 62.7 | **85.7** | 49.0 | 71.2 | 67.9 | 63.4 | 44.8 | 64.79 |
| Molmo2-8B | 75.6 | **81.9** | 64.1 | 72.1 | 35.4 | 61.4 | 85.4 | 43.4 | 70.8 | 55.2 | 49.2 | 33.8 | 60.69 |
| Qwen3-VL-8B | 68.8 | 73.0 | 58.2 | 73.6 | 36.5 | **62.9** | 82.1 | 41.8 | 70.8 | 65.9 | 51.0 | 35.0 | 59.97 |
| InternVL3.5-8B | 71.0 | 72.7 | 62.7 | 71.0 | 33.6 | 58.1 | 81.5 | 38.8 | 72.1 | 66.7 | 47.2 | 28.4 | 58.65 |
| VideoLLaMA3-7B | 65.6 | 69.7 | 49.2 | 69.3 | 29.4 | 57.8 | 81.5 | 29.9 | 70.5 | 59.1 | 44.8 | 28.6 | 54.62 |

### Against Meta's V-JEPA video LLMs

Meta has not released these video LLMs, only the encoders, so their numbers are quoted from Meta's papers. The six benchmarks are the ones both of Meta's papers report.

| Model | Visual tokens | Training samples | MVB | PT | TVB | TC | TOM | MVP | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 (Llama-3.1-8B) | 9,216 | 88.5M | 73.5 | 84.0 | 60.6 | 76.9 | 40.3 | 44.5 | 63.3 |
| V-JEPA 2.1 (Llama-3.1-8B) | – | 72.5M | 75.4 | 83.1 | 62.8 | 74.0 | 38.0 | 43.2 | 62.8 |
| **Llama-V-JEPA-LM-Qwen3-8B** | 1,024 | 4.53M | 72.8 | 77.8 | 75.3 | 79.4 | 56.5 | 58.1 | **70.0** |
| **Llama-V-JEPA-LM-Llama3.1-8B** | 1,024 | 4.53M | 71.9 | 75.0 | 71.8 | 77.6 | 55.5 | 57.6 | **68.2** |

### Generalization beyond the training constructions

**Held-out sets.** Six held-out sets test only factors that never appear in training:
- unseen playback-speed ratios (1.6, 2.5, 4.0)
- speed and camera direction on Kinetics-700, a corpus never used for these families
- unseen mirror × reversal and acceleration × reversal combinations
- camera motion from an unseen simulator

| | Ours (1B–8B) | Open models (2B–8B) |
|---|---:|---:|
| Six held-out sets, average (chance 33.3) | **94.5–96.3** | 43.5–55.5 |
| Unseen speed ratios (chance 33.3) | **97.8–99.3** | 32.2–39.3 |

**Successive playback.** This test plays the same clip twice in a row, a short central span and the full span, in either order, so the motion speeds up or slows down halfway through. The construction is never used for speed in training. Our models score **92.5–99.2**, against 19.4–51.9 for open models.

### Inference

All times are per 32-frame clip on one H100 (batch 1, bf16).

| Model | Visual tokens | LLM prefill (TFLOPs) | First token, 480p (ms) | First token, 720p (ms) |
|---|---:|---:|---:|---:|
| Llama-V-JEPA-LM-Llama3.2-1B | 1,024 | 2.3 | 70 | 71 |
| Llama-V-JEPA-LM-Qwen3-1.7B | 1,024 | 3.3 | 83 | 82 |
| Llama-V-JEPA-LM-Llama3.2-3B | 1,024 | 6.7 | 159 | 159 |
| Llama-V-JEPA-LM-Qwen3-4B | 1,024 | 8.6 | 167 | 168 |
| Llama-V-JEPA-LM-Llama3.1-8B | 1,024 | 16.2 | 175 | 175 |
| Llama-V-JEPA-LM-Qwen3-8B | 1,024 | 15.8 | 182 | 182 |

The visual-token count, and with it the language model's cost, does not depend on the input resolution. The model cards compare these costs with open models of the same size.

## 6. Why it works: controlled study

We trained four 2B models on the same mixture and schedule, crossing encoder (V-JEPA ViT-L or SigLIP so400m) with supervision (with or without the counterfactual rows).

**Averages and held-out sets.**

| 2B model | Avg (12 scores) | Held-out (6 sets) | Unseen speed ratios |
|---|---:|---:|---:|
| SigLIP, captions + QA | 58.52 | 54.0 | 35.9 |
| SigLIP + counterfactual | 62.13 | 74.8 | 55.7 |
| V-JEPA, captions + QA | 61.53 | 64.3 | 33.3 |
| **V-JEPA + counterfactual** | **66.02** | **95.0** | **98.7** |

**What each part decides.**
- **The encoder decides what can be read.** Without counterfactual rows, V-JEPA is already ahead of SigLIP on direction (+19.7), counting (+7.7) and fine actions (SSv2, +11.5).
- **The supervision decides what is read.** The counterfactual rows add 38 points on speed questions of natural benchmarks. They also add camera motion (+6.8) and state change (+4.1) on V-JEPA.
- **The gain holds at 4B.** The same comparison at 4B (8k steps) gives +4.8 on the average (paired 95% interval [+4.0, +5.5]) and +29.4 on speed.

**Matched-weight controls do not reproduce the gain.** Each control replaces the counterfactual rows at the same weight:
- *Arrow-of-time question only.* This is one forward-or-reverse question. The model learns it (95.9% on unseen clips), but the benchmark average ends at 61.17, no better than captions and QA (61.53).
- *22 annotation-based temporal-QA sources.* At 8k steps these give 58.21. The counterfactual rows give 61.18 at that point.

**Inside the model.** With counterfactual supervision, 94–98% of the answer to a speed question comes from the change between the two clips, not from their shared content. Without it, the model gives the same answer to a clip and its re-timed copy 89–91% of the time. Removing eight speed directions from the language model takes unseen speed combinations from 98.9 to 65.7. A random subspace of the same energy leaves 97.2.

## 7. Usage

Every model loads with `transformers` and `trust_remote_code=True`. It has a `chat` method for free-form answers and a `choose` method for multiple choice by option scoring:

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("kkt94/Llama-V-JEPA-LM-Qwen3-4B", trust_remote_code=True,
                                  dtype=torch.bfloat16).to("cuda").eval()
print(model.chat("my_video.mp4", "Describe what happens in the video."))
idx, scores = model.choose("my_video.mp4", "Which way does the ball roll?", ["to the left", "to the right"])
```

See each model card for installation, memory requirements, frame sampling, segment selection (`start_sec` / `end_sec`) and encoding a video once for several questions.

**This repository.**

| Path | Contents |
|---|---|
| `llama_vjepa_lm/modeling_jepa_llm.py` | the model: V-JEPA 2.1 encoder, spectral bridge, language model, and the `chat` / `choose` API |
| `llama_vjepa_lm/processing_jepa_llm.py` | video decoding, frame sampling, preprocessing, prompts and chat templates |
| `llama_vjepa_lm/configuration_jepa_llm.py` | the config class |
| `llama_vjepa_lm/vjepa2_1_encoder.py` | inference-only V-JEPA 2.1 encoder, adapted from [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) (MIT, `LICENSE_VJEPA2`) |
| `requirements.txt` | tested package versions |

The Hugging Face repositories load the same files through `trust_remote_code=True`. One code version serves all six models: the encoder size, bridge width and language model are read from each model's config. Training code, the counterfactual data generator and the training manifests will be added to this repository.

## 8. Scope and limitations

- **Single images and text in frames.** V-JEPA 2.1 is trained only on video, to predict masked spatiotemporal regions in latent space. It is never trained to align with text or to read it. Text recognition (OCR, documents, charts, slides) and single-image question answering are therefore outside what the encoder was built for, and they are not the target of this work. These models are built for how scenes change over time. For text-heavy or single-image tasks, an image–text encoder is the natural choice.
- **Long videos.** We target clip-level understanding of dynamics, 32 frames per clip. Hour-scale long-video benchmarks need many clips per video and far longer contexts, in training as well as at inference. At the scale of this release (1B–8B, one training stage of 229–648 H100-hours), that training cost is impractical, so we did not train or evaluate for it. For a long video, `start_sec` / `end_sec` select the segment to ask about.

## 9. Licence and naming

- **Language-model weights:** they follow the base model's licence (Apache-2.0 for Qwen3, the Llama 3.x Community License for Llama).
- **Video encoder:** the encoder is the unchanged V-JEPA 2.1 checkpoint (MIT).
- **Bridge weights and model code:** Apache-2.0.
- **Training data:** the training datasets remain under their own terms, and several restrict commercial use. We release the weights for research and non-commercial use.

**Naming.** Every model name begins with "Llama" because the training data include the FGQA split of PLM-Video-Human. That split is an output of Llama 3.2, and the Llama 3.2 Community License requires that prefix.

## 10. Citation

The entry will be updated when the paper is public.

```bibtex
@misc{vjepalm2026,
  title  = {V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations},
  author = {Anonymous},
  year   = {2026}
}
```

Please also cite V-JEPA 2 / V-JEPA 2.1, and Qwen3 or Llama 3.

## Acknowledgements

This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP, AI Computing Support Project for R&D) grant funded by the Korea government (MSIT) (No. RS-2026-25505492, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale).

## Authors

Keuntae Kim, Hanyang University, Artificial Intelligence Lab

# Training, evaluation and data code

This folder holds the code that trained and evaluated the released Llama-V-JEPA-LM models: the model, the one-stage trainer, the evaluation harness for our models and the open baselines, and the scripts that download, preprocess and build every training source, including the counterfactual constructions. The inference-only package that the Hugging Face models load is in [`../llama_vjepa_lm`](../llama_vjepa_lm).

This is research code, published as it ran. Comments carry the dates of the changes they describe.

## Layout

| Path | Contents |
|---|---|
| `jepa_llm/models/` | frozen V-JEPA 2.1 encoder wrapper (`vision_backbone.py`), spectral bridge and evidence path (`stage1.py`, `evidence_bridge.py`, `evidence_composer.py`), language-model graft (`graft.py`) |
| `jepa_llm/training/train_single.py` | the one-stage trainer (FSDP, torchrun) used for every released model |
| `jepa_llm/data/` | manifest datasets, clip composition for the counterfactual rows (`video.py`, `stage1_dataset.py`), chat formatting |
| `jepa_llm/eval/` | evaluation runner, option-letter scoring and metrics |
| `configs/single/<run>.yaml`, `<run>_eval.yaml` | training and evaluation configs of the released models (table below) |
| `configs/templates/` | chat templates used in training |
| `scripts/eval/` | `run_suite.py` (our models), `baseline_eval.py` (open models), benchmark manifest builders |
| `scripts/download/`, `scripts/prep/` | dataset download lanes and video preprocessing (H.264, short side 384) |
| `scripts/data/`, `scripts/data_gen/` | manifest builders; `data_gen/` holds the generators of the counterfactual and re-formatted sources |
| `DATA_PROVENANCE.tsv` | for each of the 175 training sources: manifest, the scripts that built it (in run order), its inputs |
| `scripts/launch/single_gpu.sh` | single-GPU launcher; multi-GPU runs use the same `torchrun` command (below) |

## Released models and their configs

| Model | Config | Encoder | GPUs | Batch / GPU | Steps | LR (LLM / new) |
|---|---|---|---:|---:|---:|---|
| Llama-V-JEPA-LM-Llama3.2-1B | `mxl1b_Q` | V-JEPA 2.1 ViT-L | 28 | 6 | 27,382 | 1.05e-5 / 2e-4 |
| Llama-V-JEPA-LM-Qwen3-1.7B | `mx2b_Q` | V-JEPA 2.1 ViT-L | 14 | 12 | 27,382 | 1.05e-5 / 2e-4 |
| Llama-V-JEPA-LM-Llama3.2-3B | `mxl3b_Q` | V-JEPA 2.1 ViT-g | 30 | 10 | 15,094 | 1.7e-5 / 2.83e-4 |
| Llama-V-JEPA-LM-Qwen3-4B | `mx4b_Q` | V-JEPA 2.1 ViT-g | 30 | 5 | 30,187 | 1.2e-5 / 2e-4 |
| Llama-V-JEPA-LM-Llama3.1-8B | `mxl8b_Q` | V-JEPA 2.1 ViT-g | 30 | 5 | 30,187 | 8e-6 / 2e-4 |
| Llama-V-JEPA-LM-Qwen3-8B | `mx8b_Q` | V-JEPA 2.1 ViT-g | 30 | 5 | 30,187 | 8e-6 / 2e-4 |
| Llama-V-JEPA-LM-Qwen3-30B-A3B | `mx30a_Q` | V-JEPA 2.1 ViT-G | 28 | 34 | 4,717 | 2.02e-5 / 5.06e-4 |

The global batch is batch / GPU × GPUs × `grad_accum`. With a different GPU count, keep the global batch by changing `optim.batch_per_gpu` or `optim.grad_accum`. The 30B run used 30 GPUs × 32 clips for its first 750 steps.

## Setup

```bash
pip install -r requirements.txt        # Python 3.12; torch 2.13 + CUDA 13, transformers 5.10.4 (tested versions)
```

All paths in configs and scripts assume one data root, `/data/llama_vjepa_lm`, with this folder at `/data/llama_vjepa_lm/training`:

| Path under the data root | Contents |
|---|---|
| `models/` | language models (e.g. `models/Qwen3-4B-Instruct-2507`) and V-JEPA 2.1 checkpoints, as downloaded from their official sources |
| `data/`, `bench/` | raw training datasets and benchmarks |
| `prep/` | preprocessed videos written by `scripts/prep/` |
| `manifests/` | training and evaluation manifests (parquet / jsonl) |
| `runs/` | checkpoints, logs, bridge statistics (`runs/whiten*/`) |

Set `JEPA_ROOT` for the launcher, and replace the prefix in the configs if your data root differs.

## Pipeline

1. **Download** the datasets with `scripts/download/` (each lane pins a source revision; some datasets need a licence acceptance first).
2. **Preprocess** videos with `scripts/prep/` (`transcode_one.sh`: H.264, short side 384, per-source fps).
3. **Build manifests** with `scripts/data/` and `scripts/data_gen/`, following `DATA_PROVENANCE.tsv`. Most counterfactual rows store a clip-composition recipe (`tq` column: segments, reversal, mirroring, speed) that the loader applies on the fly, so the transformed clips are not written to disk.
4. **Bridge statistics.** `scripts/data_gen/zz_whiten_stats3.py` computes the frozen PCA rotation and scale of the spectral bridge from encoder features (`runs/whiten*/`, referenced by `model.spectral.stats`).
5. **Train:**

   ```bash
   cd /data/llama_vjepa_lm/training
   PYTHONPATH=. torchrun --nproc_per_node=8 -m jepa_llm.training.train_single configs/single/mx4b_Q.yaml
   ```

   Multi-node: add `--nnodes`, `--node_rank` and a rendezvous endpoint as usual for `torchrun`. Checkpoints go to `run.out_dir` and runs resume from the latest checkpoint there.
6. **Evaluate** a checkpoint on the twelve scores and the held-out sets:

   ```bash
   PYTHONPATH=. python scripts/eval/run_suite.py run --config configs/single/mx4b_Q_eval.yaml \
       --ckpt /data/llama_vjepa_lm/runs/mx4b_Q/ckpt/step_0030187 --out /data/llama_vjepa_lm/runs/eval/mx4b_Q_final --gpus 0
   ```

   Open models: `scripts/eval/baseline_eval.py run --model <family> --path <model dir> --nf 32`, the same items, prompts and option-letter scoring.

## Notes

- Several sources cannot be rebuilt bit-identically (third-party videos that are no longer online, sampling seeds of early generator versions); `DATA_PROVENANCE.tsv` marks the frozen versions (`zz_bak_*`) that produced the manifests used in training.
- Model and dataset licences are listed in the main [README](../README.md) and on each model card.

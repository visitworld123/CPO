# CPO

This directory is the self-contained release of our CPO preference-optimization implementation. It keeps the trainer and the small amount of data/model plumbing needed to run it, without experiment logs, checkpoints, or machine-specific paths.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For 4-bit or 8-bit training, install `bitsandbytes` separately. Flash Attention is optional; set `attn_implementation: null` in the YAML file when it is not installed.

## Run

Edit `configs/cpo-demo.yaml` to choose a model, dataset, and output directory, then run:

```bash
python train.py configs/cpo-demo.yaml
```

The dataset must contain preference pairs in either of these formats:

```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

or the explicit `prompt`, `chosen`, and `rejected` fields. `chosen` and `rejected` are OpenAI-style message lists. The default example uses `HuggingFaceH4/ultrafeedback_binarized` and its `train_prefs`/`test_prefs` splits.

## CPO variants

Set `easy_version` in the YAML file to select the loss branch:

- `easy`: adaptive margin from the two reference log probabilities.
- `clip`: clipped reference log-ratio.
- any other value (for example `cpo`): smooth adaptive margin controlled by `gamma`, `tau`, and `alpha`.

The main implementation is [cpo/trainer.py](cpo/trainer.py), and the public configuration is [cpo/config.py](cpo/config.py).

## License and attribution

The trainer is based on the Hugging Face Alignment Handbook and retains its Apache-2.0 notices. See [LICENSE](LICENSE).

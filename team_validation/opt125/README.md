# OPT-125M Matched Pair Validation

This package runs one independent clean/backdoor pair on the recipient computer. It does
not use the captain's GPU and does not connect to a remote inference service.

## Before Starting

- Windows 10/11, NVIDIA GPU with at least 7 GiB VRAM, and a current NVIDIA driver.
- Python 3.11 installed with the `py` launcher or available as `python.exe`.
- At least 6 GiB free disk space.
- Internet access on the first run to install Python packages and cache
  `facebook/opt-125m` plus `tatsu-lab/alpaca`.
- Keep this package private. It contains a local backdoor-training configuration for the
  competition experiment, but does not contain the unpublished paper or existing weights.

## Run

Double-click `RUN_OPT125_PAIR.cmd` and leave the window open. The default participant id is
the Windows computer name. To use a specific id from a terminal:

```powershell
.\RUN_OPT125_PAIR.cmd -Participant member-a
```

Expected RTX 4060 time is approximately 2.5-3.5 hours: two LoRA trainings, one training-side
quality gate, two complete four-shard vocabulary scans, and two latent probes. The first run
also downloads dependencies and public assets.

The run is resumable. If Windows restarts or a command fails, do not delete or rename files;
run the same command again. Completed epochs and validated vocabulary shards are reused.

## Return

When complete, send only this file to the captain:

```text
team_runs/opt125-<participant>/RETURN_TO_CAPTAIN_<participant>.zip
```

The return ZIP contains configs, manifests, quality evidence, four shard reports, mining and
probe reports, small replay vectors, logs, and SHA256 values. It excludes model adapters by
default. Do not send the whole working directory unless the captain asks for the weights.

## Result Boundary

This is one OPT-125M matched pair. The paper probability threshold, log-likelihood gap,
candidate-family support, and replay rate are all recorded. The `family support >= 5` rule was
calibrated on GPT-2, so its OPT-125M result is cross-model coverage evidence, not a formal OPT
calibration or proof of safety.

If the training quality gate fails, do not edit thresholds or rerun with a different seed.
Send `quality.json` and the `logs` directory to the captain.

# START HERE - AI EXECUTION CONTRACT

You are the execution agent for bundle `{{BUNDLE_ID}}`.

Your task is operational: run the frozen experiment to completion and return one verified
archive. Do not redesign, optimize, review, or edit the experiment.

## Fixed Assignment

- Base model: `{{BASE_MODEL}}`
- Expected architecture: `{{MODEL_TYPE}}`
- Training seed: `{{SEED}}`
- Mode: `{{MODE}}`
- Expected runtime: {{RUNTIME}}
- Expected GPU floor: {{VRAM}} GiB VRAM
- Expected free disk floor: {{DISK}} GiB

{{REPAIR_NOTICE}}
{{AUTH_NOTICE}}

## Required Command

From PowerShell in the extracted bundle root:

```powershell
.\RUN_TEAM_PAIR.cmd -Participant <stable-member-id>
```

For a repair bundle, pass the original completed run directory when the package was not
extracted over the original project:

```powershell
.\RUN_TEAM_PAIR.cmd -Participant participant_a -OutputRoot "C:\path\to\team_runs\opt125-participant_a"
```

Run the same command after an interruption. The runner resumes validated epochs and shards.

## Non-Negotiable Rules

1. Do not edit Python, YAML, JSON, thresholds, seeds, model ids, or dataset ids.
2. Do not substitute another model or silently reduce training/mining/probe budgets.
3. Do not pass trigger text, target output, poisoned data, or a clean model into detection.
4. Do not delete the run directory after an error. Resume the same directory.
5. Do not report success from screenshots, console excerpts, or loose files.
6. A successful return MUST include both backdoor and clean Adapter weights.
7. Do not upload this private bundle, returned weights, or reports to a public service.

## What To Return

Success is valid only when the terminal prints both `RETURN_VERIFIED` and `RETURN_READY`.
Return exactly:

- the printed `SUCCESS_RETURN_*.zip`;
- its printed SHA256 line.

If any stage fails, return the printed `FAILURE_RETURN_*.zip` and SHA256 instead. Do not
change the experiment to force a pass. The captain will diagnose the failure package.

Before stopping, open `RETURN_CONTRACT.json` and confirm every success requirement is enforced
by the generated manifest. Missing Adapter weights always means failure.

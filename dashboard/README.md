#Start all 3 at once


./dev.sh







# Leonardo Pipeline Dashboard

A local-only web app that drives the existing Leonardo Slurm pipeline over SSH:
generate the dataset, upload, check the environment, train, poll the queue, watch
the loss stream live, evaluate, and read rule-aware results - all from buttons and
charts.

The browser never SSHes into Leonardo. A small FastAPI backend on your Mac holds
the password (in a gitignored `.env`) and wraps the exact commands we already use
(`sbatch`, `squeue`, `sacct`, `sftp`, tailing `train_log.csv`).

```
Browser (React) --REST/SSE--> FastAPI (127.0.0.1) --paramiko--> Leonardo login node --sbatch--> GPU job
```

## Quick start

After the backend and AI `.env` files are set up and npm deps are installed:

```bash
cd dashboard
./dev.sh
```

This starts all local services:

- FastAPI backend on `http://127.0.0.1:8000`
- AI gateway on `http://127.0.0.1:8787`
- React/Vite frontend on `http://127.0.0.1:5173`

By default `dev.sh` stops existing processes on those three ports first, so the
stack starts cleanly. To reuse already-running services instead:

```bash
REUSE_EXISTING=1 ./dev.sh
```

## 1. Backend

```bash
cd dashboard/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then edit .env and set LEONARDO_PASSWORD
uvicorn main:app --host 127.0.0.1 --port 8000
```

`.env` keys (defaults already match this project):

- `LEONARDO_HOST`, `LEONARDO_USER`, `LEONARDO_PORT`
- `LEONARDO_PASSWORD` (required)
- `REMOTE_WORKDIR`, `SLURM_ACCOUNT`, `SLURM_RESERVATION`

## 2. Frontend

```bash
cd dashboard/frontend
npm install
npm run dev      # http://localhost:5173  (proxies /api to the backend)
```

## 3. Optional AI Experiment Coach

The AI coach is a separate local Node service. It never gets SSH or shell
access; it only calls the same typed FastAPI routes as the dashboard UI.

```bash
cd dashboard/ai-gateway
npm install
npm run dev   # http://127.0.0.1:8787
```

Optional env:

- `AI_PROVIDER=openai`
- `AI_MODEL=gpt-4.1-mini`
- `FASTAPI_BASE_URL=http://127.0.0.1:8000`

If `OPENAI_API_KEY` is missing, the structured analysis endpoint returns a
deterministic local fallback so the UI card can still be tested.

## Official participant submission

The dashboard also exposes the organizer files under
`/Users/zero1hack/participant_files`:

- `GET /api/submission` reports official input rows, generated prediction CSVs,
  and anomaly invalid count.
- `GET /api/submission/checkpoints` lists transformer checkpoints on Leonardo
  (the canonical `outputs/transformer/` model plus archived `runs/<job_id>/`).
- `POST /api/submission/run` submits `run_make_submission.slurm` on Leonardo
  with the chosen checkpoint and tracks it like any other job.
- `POST /api/submission/remove-checkpoint` deletes an archived remote checkpoint
  folder (the canonical one is protected).

The submission runs **as a Slurm job on Leonardo** (where the checkpoints and
GPUs already live), not on the laptop. The endpoint uploads the current
`training_data/` scripts + the organizer eval inputs, then runs
`training_data/make_submission.py` (`CHECKPOINT`/`VOCAB`/`INPUT_DIR`/`TASKS`
overridable via `--export`) writing CSVs to `outputs/transformer/submission/`.
When the job finishes, the dashboard automatically pulls the three prediction
CSVs back into local `participant_files/submission/`. Task 3 anomaly is
rule-based (no model needed); Tasks 1/2 use the selected checkpoint.

## How it maps to the pipeline

| Button            | What happens                                                              |
| ----------------- | ------------------------------------------------------------------------- |
| Connect           | `hostname && whoami` over SSH - confirms password auth works              |
| Generate dataset  | runs `training_data/generate_sequences.py` locally for all 3 families     |
| Upload            | `sftp` of `*.py`, `*.slurm`, `*_variants.csv`, `synthetic*.csv` to remote |
| Check environment | `pixi run python -c "import torch; ..."` (torch + CUDA build)             |
| Train             | `sbatch run_train_transformer.slurm`, captures the job id                 |
| Evaluate rules    | `sbatch run_evaluate_*_rules.slurm`                                        |

Live data:

- Slurm queue auto-polls every 5s (`squeue --me`).
- The loss chart streams `train_log.csv` rows via SSE as each epoch is flushed.
- The log drawer tails the job `.out` / `.err` live.
- Results parse `rule_eval_summary.csv` into valid-rate cards + a table.

## API surface

`GET /api/config`, `POST /api/ssh/test`, `POST /api/upload`,
`POST /api/setup`, `POST /api/run/{run_key}`, `GET /api/queue`, `GET /api/jobs`,
`GET /api/jobs/{id}/status`, `GET /api/runs`, `GET /api/runs/{run}`,
`GET /api/datasets`, `DELETE /api/datasets/{id}`,
`POST /api/params/validate/{run}`, `GET /api/results?run=`,
`GET /api/loss/snapshot?run=`, `GET /api/loss/stream?run=` (SSE),
`GET /api/logs/stream?run=&which=out|err` (SSE).

Datasets are generated on Leonardo (`run_key=generate_remote`) into a versioned
`datasets/<id>/` collection; Train/Eval/Baseline accept a `dataset` id and run
against it via `--data-dir`.

Run keys: `transformer`, `ngram`, `eval_transformer`, `generate_remote`.

## Notes

- Backend binds to `127.0.0.1` only; nothing is exposed publicly.
- The password lives only in `dashboard/backend/.env` (gitignored) and is never
  sent to the browser.
- Submitting a training job consumes real GPU time on your allocation - the
  dashboard captures the returned Slurm job id so you can follow it live.

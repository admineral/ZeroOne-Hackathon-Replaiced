import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, DashboardConfig, DatasetSummary, ExperimentAnalysisCard, RunParams } from "./api";
import { usePolling } from "./hooks";
import { PipelineRail, Step, StepStatus } from "./components/PipelineRail";
import { LossChart } from "./components/LossChart";
import { ResultsPanel } from "./components/ResultsPanel";
import { LogDrawer } from "./components/LogDrawer";
import { JobStatusBadge } from "./components/JobStatusBadge";
import { DatasetCard } from "./components/DatasetCard";
import { DatasetOverviewTable } from "./components/DatasetOverviewTable";
import { TimeoutPicker } from "./components/TimeoutPicker";
import { SplitSlider } from "./components/SplitSlider";
import { VariantsPicker } from "./components/VariantsPicker";
import { ResourcePanel } from "./components/ResourcePanel";
import { AiCoachPanel } from "./components/AiCoachPanel";
import { SubmissionCard } from "./components/SubmissionCard";

type Toast = { kind: "info" | "success" | "error"; msg: string } | null;

const TRAIN_RUNS = ["transformer"];
const EVAL_RUNS = ["eval_transformer"];
const SUBMISSION_RUNS = ["submission"];
// Data-generation runs live on the Dataset Manager page, not the Train history.
const GEN_RUNS = ["generate_remote"];
// Which evaluation run re-scores a given training run's checkpoint.
const EVAL_FOR_TRAIN: Record<string, string> = {
  transformer: "eval_transformer",
};

// Product families that can be generated, in display order.
const ALL_FAMILIES = ["mosfet", "igbt", "ic"] as const;
const FAMILY_LABELS: Record<string, string> = {
  mosfet: "MOSFET",
  igbt: "IGBT",
  ic: "IC",
};

// Slurm terminal states — anything else (submitted/PENDING/RUNNING/...) is
// still in flight and can be aborted via scancel.
const JOB_TERMINAL_STATES = new Set([
  "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
  "NODE_FAIL", "DEADLINE", "BOOT_FAIL", "PREEMPTED", "REVOKED",
]);

function jobIsActive(status: string): boolean {
  return !JOB_TERMINAL_STATES.has((status || "").split(" ")[0].toUpperCase());
}

// Terminal states that mean failure (CANCELLED is user-initiated, not an error).
const JOB_FAILURE_STATES = new Set([
  "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
  "DEADLINE", "BOOT_FAIL", "PREEMPTED", "REVOKED",
]);

function jobFailed(status: string): boolean {
  return JOB_FAILURE_STATES.has((status || "").split(" ")[0].toUpperCase());
}

// Transformer size presets. head_dim stays 64; each step up roughly doubles
// d_model / layers / heads / batch. All within the control's max bounds and
// satisfy d_model % heads == 0. Param estimates are rough.
interface ModelPreset {
  key: string;
  label: string;
  hint: string;
  d_model: number;
  num_layers: number;
  num_heads: number;
  batch_size: number;
  dropout: number;
}

const MODEL_PRESETS: ModelPreset[] = [
  {
    key: "tiny",
    label: "Tiny",
    hint: "~0.5M params · d_model 128 · 2 layers · 4 heads · batch 32",
    d_model: 128,
    num_layers: 2,
    num_heads: 4,
    batch_size: 32,
    dropout: 0.1,
  },
  {
    key: "recommended",
    label: "Recommended",
    hint: "~3.5M params · d_model 256 · 4 layers · 8 heads · batch 256 · sweet spot for this task",
    d_model: 256,
    num_layers: 4,
    num_heads: 8,
    batch_size: 256,
    dropout: 0.1,
  },
  {
    key: "large",
    label: "Scale up",
    hint: "~22M params · d_model 512 · 6 layers · 8 heads · batch 128",
    d_model: 512,
    num_layers: 6,
    num_heads: 8,
    batch_size: 128,
    dropout: 0.1,
  },
  {
    key: "xl",
    label: "Even bigger",
    hint: "~180M params · d_model 1024 · 12 layers · 16 heads · batch 256",
    d_model: 1024,
    num_layers: 12,
    num_heads: 16,
    batch_size: 256,
    dropout: 0.1,
  },
];

/** Epoch seconds -> compact local "MMM D, HH:MM:SS". */
function fmtTime(epochSeconds?: number): string {
  if (!epochSeconds) return "-";
  const d = new Date(epochSeconds * 1000);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Short "x ago" relative label from epoch seconds. */
function fmtAgo(epochSeconds?: number): string {
  if (!epochSeconds) return "";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export default function App() {
  const [config, setConfig] = useState<DashboardConfig | null>(null);
  const [status, setStatus] = useState<Record<string, StepStatus>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  // Last success/error message per step (shown on the Setup page).
  const [stepResult, setStepResult] = useState<Record<string, string>>({});
  const [toast, setToast] = useState<Toast>(null);
  // Which page the main panel shows. Setup + Dataset are focused pages opened
  // from the rail ("Show"); everything else lives on the overview.
  const [activeView, setActiveView] = useState<
    "overview" | "setup" | "dataset" | "train" | "evaluate" | "submission"
  >("overview");

  const [trainRun, setTrainRun] = useState("transformer");
  const [evalRun, setEvalRun] = useState("eval_transformer");
  const [genCount, setGenCount] = useState(1000);
  const [epochs, setEpochs] = useState(20);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [batchSize, setBatchSize] = useState(32);
  const [seqLen, setSeqLen] = useState(176);
  // Train/val split as integer percentages (test = remainder). Evaluation
  // reuses the checkpoint's recorded ratios so the held-out set matches.
  const [splitTrain, setSplitTrain] = useState(80);
  const [splitVal, setSplitVal] = useState(10);
  const [dModel, setDModel] = useState(128);
  const [numLayers, setNumLayers] = useState(2);
  const [numHeads, setNumHeads] = useState(4);
  const [dropout, setDropout] = useState(0.1);
  // Family-token dropout for prefix-free robustness (matches trainer default).
  const [familyDropout, setFamilyDropout] = useState(0.3);
  // Regularization + schedule + scaling knobs (prepare for big runs).
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [labelSmoothing, setLabelSmoothing] = useState(0.0);
  const [lrSchedule, setLrSchedule] = useState("none");
  const [warmupRatio, setWarmupRatio] = useState(0.05);
  const [numWorkers, setNumWorkers] = useState(8);
  const [gpus, setGpus] = useState(1);
  // Multi-GPU training via DistributedDataParallel (only when gpus > 1).
  const [ddp, setDdp] = useState(false);
  // Archive this run's weights + vocab into runs/<job_id>/ for later re-eval.
  const [keepCheckpoint, setKeepCheckpoint] = useState(true);
  const [seed, setSeed] = useState(42);
  // Cap sequences read per family (0 = all). Bounds RAM on huge datasets so
  // training/eval don't OOM the node. Recorded in the checkpoint so eval reuses
  // the same split.
  const [maxSequences, setMaxSequences] = useState(0);
  const [timeLimit, setTimeLimit] = useState("01:00:00");
  // Slurm walltime for on-Leonardo dataset generation (separate from training).
  const [genTimeLimit, setGenTimeLimit] = useState("02:00:00");
  // Slurm walltime for rule evaluation (separate from training/generation).
  // Eval is sequential greedy generation (no KV-cache), so default generously.
  const [evalTimeLimit, setEvalTimeLimit] = useState("02:00:00");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [genRemoteBusy, setGenRemoteBusy] = useState(false);
  // Which product families to generate (free combination; at least one).
  const [genFamilies, setGenFamilies] = useState<Record<string, boolean>>({
    mosfet: true,
    igbt: true,
    ic: true,
  });
  const selectedFamilies = useMemo(
    () => ALL_FAMILIES.filter((f) => genFamilies[f]),
    [genFamilies]
  );
  const datasetGenCardRef = useRef<HTMLDivElement>(null);
  const [datasetGenCardHeight, setDatasetGenCardHeight] = useState<number | null>(null);

  const [liveKey, setLiveKey] = useState(0);
  const [trainingLive, setTrainingLive] = useState(false);
  const [resultsKey, setResultsKey] = useState(0);
  const [evalWatch, setEvalWatch] = useState(false);
  // Track the most recent submissions so fast jobs don't look like they vanished.
  const [lastTrainJobId, setLastTrainJobId] = useState<string | null>(null);
  const [lastEvalJobId, setLastEvalJobId] = useState<string | null>(null);
  // After a manual "Clear history" we don't want the queue-driven auto-live to
  // immediately re-light the loss chart from a still-queued job. We capture the
  // queue size at clear time and stay suppressed until a *new* job appears.
  const [liveQueueBaseline, setLiveQueueBaseline] = useState<number | null>(null);
  // Bumped after generate/upload/delete to refresh the dataset collection.
  const [datasetKey, setDatasetKey] = useState(0);
  // Bumped after generating official participant submission CSVs.
  const [submissionKey, setSubmissionKey] = useState(0);
  // Versioned dataset collection on Leonardo + which one train/eval use.
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("");

  // Inspecting a past run from the job history (archived snapshots).
  const [inspectTrainJob, setInspectTrainJob] = useState<string | undefined>();
  const [inspectEvalJob, setInspectEvalJob] = useState<string | undefined>();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  // Fresh run just launched: chart waits for the new run instead of showing the old curve.
  const [awaitFresh, setAwaitFresh] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [confirmCancelId, setConfirmCancelId] = useState<string | null>(null);
  // Which failed job's error detail is expanded in the history table.
  const [errorOpenId, setErrorOpenId] = useState<string | null>(null);
  const [cancelingId, setCancelingId] = useState<string | null>(null);
  // Re-evaluating a stored checkpoint from the job-history Evaluate button.
  const [evaluatingId, setEvaluatingId] = useState<string | null>(null);

  const toastTimer = useRef<number | null>(null);

  const flash = useCallback((t: Toast) => {
    setToast(t);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 5000);
  }, []);

  useEffect(() => {
    api
      .getConfig()
      .then(setConfig)
      .catch((err) => flash({ kind: "error", msg: `Config load failed: ${err.message}` }));
  }, [flash]);

  // Load the dataset collection from Leonardo and keep a sensible default
  // selection (newest ready dataset) for training/eval.
  useEffect(() => {
    if (!config) return;
    setDatasetsLoading(true);
    setDatasetsError(null);
    api
      .listDatasets()
      .then((res) => {
        setDatasets(res.datasets);
        setSelectedDataset((cur) => {
          if (cur && res.datasets.some((d) => d.id === cur)) return cur;
          const newestReady = res.datasets.find((d) => d.ready);
          return newestReady ? newestReady.id : "";
        });
      })
      .catch((err) =>
        setDatasetsError(err instanceof Error ? err.message : String(err))
      )
      .finally(() => setDatasetsLoading(false));
  }, [config, datasetKey]);

  // Re-fetch the dataset collection whenever the user lands on the overview
  // (Training Settings) page, so a dataset just generated on the Dataset page
  // shows up in the table without a manual "Refresh" click.
  useEffect(() => {
    if (activeView === "overview") setDatasetKey((k) => k + 1);
  }, [activeView]);

  const queue = usePolling(api.getQueue, 5000, !!config);
  const jobs = usePolling(api.getJobs, 6000, !!config);

  // After a submit, poll harder for ~25s so short jobs (e.g. a 13s eval) show
  // up in the queue and flip to their terminal status without waiting for the
  // slow background interval.
  const refreshSoon = useCallback(() => {
    for (const ms of [0, 2000, 5000, 9000, 15000, 25000]) {
      window.setTimeout(() => {
        jobs.refresh();
        queue.refresh();
      }, ms);
    }
  }, [jobs.refresh, queue.refresh]);

  // Auto-attach the live stream when a job is already running in the queue, so
  // you can watch a run you didn't start from this tab.
  const queueCount = queue.data?.rows?.length ?? 0;
  useEffect(() => {
    // Suppressed after a manual clear: resume only once the queue grows past
    // the baseline (a brand-new job), tracking it downward as jobs drain so a
    // later submission still re-triggers live.
    if (liveQueueBaseline !== null) {
      if (queueCount > liveQueueBaseline) {
        setLiveQueueBaseline(null);
      } else {
        if (queueCount < liveQueueBaseline) setLiveQueueBaseline(queueCount);
        return;
      }
    }
    if (queueCount > 0 && !trainingLive) {
      setAwaitFresh(false);
      setTrainingLive(true);
      setLiveKey((k) => k + 1);
    }
  }, [queueCount, trainingLive, liveQueueBaseline]);

  // Auto-refresh results for a couple minutes after an evaluation is launched.
  useEffect(() => {
    if (!evalWatch) return;
    const id = window.setInterval(() => setResultsKey((k) => k + 1), 6000);
    const stop = window.setTimeout(() => setEvalWatch(false), 180000);
    return () => {
      window.clearInterval(id);
      window.clearTimeout(stop);
    };
  }, [evalWatch]);

  // Pin the data-generation jobs card to the generator card height.
  useEffect(() => {
    if (activeView !== "dataset") return;
    const el = datasetGenCardRef.current;
    if (!el) return;
    const sync = () => setDatasetGenCardHeight(el.offsetHeight);
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, [activeView, genFamilies, genCount, genTimeLimit]);

  const setStep = (id: string, s: StepStatus) =>
    setStatus((prev) => ({ ...prev, [id]: s }));

  const runStep = useCallback(
    async (id: string, fn: () => Promise<string>): Promise<boolean> => {
      setBusy((p) => ({ ...p, [id]: true }));
      setStep(id, "running");
      try {
        const msg = await fn();
        setStep(id, "done");
        setStepResult((p) => ({ ...p, [id]: msg }));
        flash({ kind: "success", msg });
        return true;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStep(id, "error");
        setStepResult((p) => ({ ...p, [id]: msg }));
        flash({ kind: "error", msg });
        return false;
      } finally {
        setBusy((p) => ({ ...p, [id]: false }));
      }
    },
    [flash]
  );

  // Setup actions (used by both the rail's combined Setup step and the Setup page).
  const doConnect = useCallback(
    () =>
      runStep("connect", async () => {
        const r = await api.sshTest();
        return `Connected to ${r.hostname} as ${r.user}`;
      }),
    [runStep]
  );
  const doUpload = useCallback(
    () =>
      runStep("upload", async () => {
        const r = await api.upload();
        setDatasetKey((k) => k + 1);
        return `Uploaded ${r.count} files`;
      }),
    [runStep]
  );
  const doCheckEnv = useCallback(
    () =>
      runStep("checkenv", async () => {
        const r = await api.setup();
        if (!r.ok) throw new Error(r.stderr || "setup failed");
        return `torch ${r.torch_version} (cuda ${r.cuda_build})`;
      }),
    [runStep]
  );
  // Run connect -> check-env -> upload in order, stopping at the first failure.
  const runAllSetup = useCallback(async () => {
    if (!(await doConnect())) return;
    if (!(await doCheckEnv())) return;
    await doUpload();
  }, [doConnect, doUpload, doCheckEnv]);

  // Submit a training job and jump to the Train page to watch it live.
  const doTrain = useCallback(() => {
    setActiveView("train");
    return runStep("train", async () => {
      const r = await api.runJob(trainRun, {
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        max_seq_len: seqLen,
        d_model: dModel,
        num_layers: numLayers,
        num_heads: numHeads,
        dropout,
        family_dropout: familyDropout,
        weight_decay: weightDecay,
        label_smoothing: labelSmoothing,
        lr_schedule: lrSchedule,
        warmup_ratio: warmupRatio,
        num_workers: numWorkers,
        train_ratio: splitTrain / 100,
        val_ratio: splitVal / 100,
        max_sequences: maxSequences || undefined,
        gpus,
        ddp: ddp && gpus > 1,
        keep_checkpoint: keepCheckpoint,
        seed,
        time_limit: timeLimit || undefined,
        dataset: selectedDataset || undefined,
      });
      setInspectTrainJob(undefined);
      setSelectedJobId(null);
      setAwaitFresh(true);
      setTrainingLive(true);
      setLiveKey((k) => k + 1);
      setLastTrainJobId(r.job_id ?? null);
      refreshSoon();
      return `Submitted ${trainRun} job ${r.job_id ?? "?"} (${epochs} epochs)`;
    });
  }, [
    runStep,
    trainRun,
    epochs,
    learningRate,
    batchSize,
    seqLen,
    dModel,
    numLayers,
    numHeads,
    dropout,
    familyDropout,
    weightDecay,
    labelSmoothing,
    lrSchedule,
    warmupRatio,
    numWorkers,
    splitTrain,
    splitVal,
    maxSequences,
    gpus,
    ddp,
    keepCheckpoint,
    seed,
    timeLimit,
    selectedDataset,
    refreshSoon,
  ]);
  // Aggregate status of the three setup actions for the combined rail step.
  const setupStatus: StepStatus = [status.connect, status.upload, status.checkenv].includes(
    "error"
  )
    ? "error"
    : [status.connect, status.upload, status.checkenv].includes("running")
      ? "running"
      : status.connect === "done" &&
          status.upload === "done" &&
          status.checkenv === "done"
        ? "done"
        : "idle";

  const steps: Step[] = useMemo(
    () => [
      {
        id: "setup",
        title: "Setup",
        sub: "",
        status: setupStatus,
        busy: busy.connect || busy.upload || busy.checkenv,
        nav: true,
        active: activeView === "setup",
        onRun: () => setActiveView("setup"),
      },
      {
        id: "dataset",
        title: "Dataset",
        sub: "",
        status: status.generate ?? "idle",
        busy: genRemoteBusy,
        nav: true,
        active: activeView === "dataset",
        onRun: () => setActiveView("dataset"),
      },
      {
        id: "train",
        title: "Monitor",
        sub: "",
        status: status.train ?? "idle",
        busy: busy.train,
        nav: true,
        active: activeView === "train",
        onRun: () => setActiveView("train"),
      },
      {
        id: "evaluate",
        title: "Evaluate",
        sub: "",
        status: status.evaluate ?? "idle",
        busy: busy.evaluate,
        nav: true,
        active: activeView === "evaluate",
        onRun: () => setActiveView("evaluate"),
      },
      {
        id: "submission",
        title: "Submission",
        sub: "",
        status: status.submission ?? "idle",
        busy: busy.submission,
        nav: true,
        active: activeView === "submission",
        onRun: () => setActiveView("submission"),
      },
    ],
    [
      config,
      activeView,
      genRemoteBusy,
      setupStatus,
      status,
      busy,
      genCount,
      epochs,
      learningRate,
      batchSize,
      seqLen,
      splitTrain,
      splitVal,
      dModel,
      numLayers,
      numHeads,
      dropout,
      familyDropout,
      weightDecay,
      labelSmoothing,
      lrSchedule,
      warmupRatio,
      numWorkers,
      gpus,
      ddp,
      keepCheckpoint,
      seed,
      timeLimit,
      genTimeLimit,
      trainRun,
      evalRun,
      selectedFamilies,
      selectedDataset,
      runStep,
      refreshSoon,
    ]
  );

  const trainRunInfo =
    config?.runs.find((r) => r.key === trainRun) ?? {
      key: trainRun,
      label: trainRun,
      slurm: "",
      has_loss: true,
      has_summary: false,
    };

  const queueRows = queue.data?.rows ?? [];
  const jobList = jobs.data?.jobs ?? [];
  // Split history across subpages: training jobs (Training-Monitor),
  // evaluation jobs (Evaluate), and data generation (Dataset Manager).
  const generationJobs = jobList.filter((j) => GEN_RUNS.includes(j.run_key));
  const evaluationJobs = jobList.filter((j) => EVAL_RUNS.includes(j.run_key));
  const submissionJobs = jobList.filter((j) => SUBMISSION_RUNS.includes(j.run_key));
  // Datasets with a packing (preprocess) job still active in the Slurm queue, so
  // the Dataset Manager can block re-submitting a duplicate pack for them.
  const packingDatasetIds = jobList
    .filter(
      (j) =>
        j.run_key === "preprocess" &&
        jobIsActive(j.status) &&
        typeof j.params?.dataset === "string"
    )
    .map((j) => String(j.params!.dataset));
  const trainingJobs = jobList.filter(
    (j) =>
      !GEN_RUNS.includes(j.run_key) &&
      !EVAL_RUNS.includes(j.run_key) &&
      !SUBMISSION_RUNS.includes(j.run_key)
  );

  // Keep evaluation/submission jobs out of the Training-Monitor queue (they live
  // on their own pages). Match a queue row to a run by its job id (authoritative,
  // via the tracked history) with a job-name prefix fallback for rows the history
  // hasn't picked up yet.
  const evalJobIds = new Set(
    evaluationJobs
      .map((j) => j.job_id)
      .filter((id): id is string => !!id)
  );
  const submissionJobIds = new Set(
    submissionJobs
      .map((j) => j.job_id)
      .filter((id): id is string => !!id)
  );
  const isEvalQueueRow = (row: (typeof queueRows)[number]) =>
    evalJobIds.has(row.JOBID) || /^run_eval/i.test(row.NAME ?? "");
  const isSubmissionQueueRow = (row: (typeof queueRows)[number]) =>
    submissionJobIds.has(row.JOBID) || /^run_make/i.test(row.NAME ?? "");
  const evaluationQueueRows = queueRows.filter(isEvalQueueRow);
  const trainingQueueRows = queueRows.filter(
    (r) => !isEvalQueueRow(r) && !isSubmissionQueueRow(r)
  );

  const lastEvalJob = lastEvalJobId
    ? jobList.find((j) => j.job_id === lastEvalJobId)
    : undefined;
  const lastTrainJob = lastTrainJobId
    ? jobList.find((j) => j.job_id === lastTrainJobId)
    : undefined;

  // The job whose data the live / latest-snapshot chart is showing: the run we
  // just submitted if any, otherwise the most recent job for this train run.
  const liveTrainJobId =
    lastTrainJobId ??
    jobList.find((j) => j.run_key === trainRun)?.job_id ??
    undefined;

  const jobEpochs = (job: (typeof jobList)[number]): string => {
    const fromParams = job.params?.epochs;
    if (fromParams != null) return String(fromParams);
    const m = /epochs=(\d+)/.exec(job.note ?? "");
    return m ? m[1] : "-";
  };

  // Resource panel target: the explicitly-selected job, else the latest train run.
  const selectedJob = selectedJobId
    ? jobList.find((j) => j.job_id === selectedJobId)
    : undefined;
  const resourceJob =
    selectedJob ?? jobList.find((j) => j.job_id === liveTrainJobId) ?? undefined;

  // The eval shown in Rule-aware results (inspected row, else latest for the
  // run). If it was launched from a history "Evaluate" button it carries the
  // source training job id so we can show what checkpoint was scored.
  const shownEvalJobId =
    inspectEvalJob ?? jobList.find((j) => j.run_key === evalRun)?.job_id;
  const shownEvalSource = jobList.find((j) => j.job_id === shownEvalJobId)?.params
    ?.source_job_id as string | undefined;

  // Every tracked evaluation run, newest first, so the results card can offer a
  // dropdown to switch between them (not just the latest / inspected row).
  const evalResultOptions = evaluationJobs
    .filter((j) => !!j.job_id)
    .map((j) => ({
      jobId: j.job_id!,
      label: `${j.job_id} · ${(j.status || "").split(" ")[0].toUpperCase()} · ${fmtTime(
        j.submitted_at
      )}`,
    }));

  // Pick an evaluation from the results dropdown (undefined = latest/canonical).
  const selectEvalResult = (jobId: string | undefined) => {
    setInspectEvalJob(jobId);
    setSelectedJobId(jobId ?? null);
    setResultsKey((k) => k + 1);
  };

  const latestWatchedTerminal = jobList.find(
    (j) =>
      ["generate_remote", trainRun, evalRun].includes(j.run_key) &&
      !!j.job_id &&
      !jobIsActive(j.status)
  );
  const aiAutoTriggerKey = latestWatchedTerminal
    ? `${latestWatchedTerminal.job_id}:${latestWatchedTerminal.status}:${latestWatchedTerminal.updated_at}`
    : undefined;

  const currentAiControls: RunParams & {
    train_run: string;
    eval_run: string;
    generation_time_limit: string;
  } = {
    train_run: trainRun,
    eval_run: evalRun,
    count: genCount,
    families: selectedFamilies,
    epochs,
    learning_rate: learningRate,
    batch_size: batchSize,
    max_seq_len: seqLen,
    d_model: dModel,
    num_layers: numLayers,
    num_heads: numHeads,
    dropout,
    family_dropout: familyDropout,
    weight_decay: weightDecay,
    label_smoothing: labelSmoothing,
    lr_schedule: lrSchedule,
    warmup_ratio: warmupRatio,
    num_workers: numWorkers,
    train_ratio: splitTrain / 100,
    val_ratio: splitVal / 100,
    max_sequences: maxSequences || undefined,
    gpus,
    ddp: ddp && gpus > 1,
    keep_checkpoint: keepCheckpoint,
    seed,
    time_limit: timeLimit,
    generation_time_limit: genTimeLimit,
  };

  // Runs the AI coach can target, newest first. Limited to inspectable
  // train/eval jobs so the coach has run artifacts to reason about.
  const aiRunOptions = jobList
    .filter(
      (j) =>
        !!j.job_id &&
        (TRAIN_RUNS.includes(j.run_key) || EVAL_RUNS.includes(j.run_key))
    )
    .map((j) => ({
      jobId: j.job_id!,
      label: `${j.label} · ${j.job_id} · ${fmtAgo(j.submitted_at)}`,
    }));

  // Focus the coach on a specific run without leaving the overview page (unlike
  // inspectJob, which navigates to the run's subpage).
  const selectAiRun = (jobId: string | null) => {
    setSelectedJobId(jobId);
    if (!jobId) return;
    const job = jobList.find((j) => j.job_id === jobId);
    if (!job) return;
    if (TRAIN_RUNS.includes(job.run_key)) setTrainRun(job.run_key);
    else if (EVAL_RUNS.includes(job.run_key)) setEvalRun(job.run_key);
  };

  // Click a finished job to load its training curves / eval results, jumping to
  // whichever subpage now hosts that view.
  const inspectJob = (job: (typeof jobList)[number]) => {
    if (!job.job_id) return;
    setSelectedJobId(job.job_id);
    if (TRAIN_RUNS.includes(job.run_key)) {
      setTrainRun(job.run_key);
      setAwaitFresh(false);
      setTrainingLive(false);
      setInspectTrainJob(job.job_id);
      setActiveView("train");
    } else if (EVAL_RUNS.includes(job.run_key)) {
      setEvalRun(job.run_key);
      setInspectEvalJob(job.job_id);
      setResultsKey((k) => k + 1);
      setActiveView("evaluate");
    }
  };

  const applyPreset = (p: ModelPreset) => {
    setTrainRun("transformer");
    setDModel(p.d_model);
    setNumLayers(p.num_layers);
    setNumHeads(p.num_heads);
    setBatchSize(p.batch_size);
    setDropout(p.dropout);
    setShowAdvanced(true);
    flash({
      kind: "info",
      msg: `${p.label}: d_model ${p.d_model}, ${p.num_layers} layers, ${p.num_heads} heads, batch ${p.batch_size}`,
    });
  };

  const applyAiParams = (params: RunParams, runKey?: string) => {
    if (runKey) setTrainRun(runKey);
    if (params.count != null) setGenCount(Number(params.count));
    if (params.families) {
      setGenFamilies({
        mosfet: params.families.includes("mosfet"),
        igbt: params.families.includes("igbt"),
        ic: params.families.includes("ic"),
      });
    }
    if (params.epochs != null) setEpochs(Number(params.epochs));
    if (params.learning_rate != null) setLearningRate(Number(params.learning_rate));
    if (params.batch_size != null) setBatchSize(Number(params.batch_size));
    if (params.max_seq_len != null) setSeqLen(Number(params.max_seq_len));
    if (params.d_model != null) {
      setDModel(Number(params.d_model));
      setShowAdvanced(true);
    }
    if (params.num_layers != null) {
      setNumLayers(Number(params.num_layers));
      setShowAdvanced(true);
    }
    if (params.num_heads != null) {
      setNumHeads(Number(params.num_heads));
      setShowAdvanced(true);
    }
    if (params.dropout != null) setDropout(Number(params.dropout));
    if (params.family_dropout != null) setFamilyDropout(Number(params.family_dropout));
    if (params.weight_decay != null) setWeightDecay(Number(params.weight_decay));
    if (params.label_smoothing != null) setLabelSmoothing(Number(params.label_smoothing));
    if (params.lr_schedule != null) setLrSchedule(String(params.lr_schedule));
    if (params.warmup_ratio != null) setWarmupRatio(Number(params.warmup_ratio));
    if (params.num_workers != null) setNumWorkers(Number(params.num_workers));
    if (params.train_ratio != null) setSplitTrain(Math.round(Number(params.train_ratio) * 100));
    if (params.val_ratio != null) setSplitVal(Math.round(Number(params.val_ratio) * 100));
    if (params.gpus != null) setGpus(Number(params.gpus));
    if (params.ddp != null) setDdp(Boolean(params.ddp));
    if (params.keep_checkpoint != null) setKeepCheckpoint(Boolean(params.keep_checkpoint));
    if (params.seed != null) setSeed(Number(params.seed));
    if (params.time_limit != null) setTimeLimit(String(params.time_limit));
    flash({ kind: "info", msg: "AI suggested settings applied to the controls." });
  };

  const activePreset = MODEL_PRESETS.find(
    (p) =>
      p.d_model === dModel &&
      p.num_layers === numLayers &&
      p.num_heads === numHeads &&
      p.batch_size === batchSize
  )?.key;

  const cancelJob = async (jobId: string) => {
    setCancelingId(jobId);
    try {
      const r = await api.cancelJob(jobId);
      flash({
        kind: r.cancelled ? "success" : "info",
        msg: r.cancelled
          ? `Abort sent to job ${jobId} (now ${r.state})`
          : r.detail || r.stderr || `Could not abort job ${jobId}`,
      });
      refreshSoon();
    } catch (err) {
      flash({ kind: "error", msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setCancelingId(null);
      setConfirmCancelId(null);
    }
  };

  // Re-evaluate a specific past run's archived checkpoint (job-history button).
  const evaluateStoredJob = async (job: (typeof jobList)[number]) => {
    const evalKey = EVAL_FOR_TRAIN[job.run_key];
    if (!evalKey || !job.job_id) return;
    setEvaluatingId(job.job_id);
    try {
      const r = await api.runJob(evalKey, {
        source_job_id: job.job_id,
        train_ratio: splitTrain / 100,
        val_ratio: splitVal / 100,
        // Evaluate against the dataset the checkpoint was trained on so the
        // held-out split matches and the data is actually present (the default
        // training_data/ may be empty). Backend also infers this as a fallback.
        dataset: (job.params?.dataset as string | undefined) || undefined,
      });
      setEvalRun(evalKey);
      setEvalWatch(true);
      setLastEvalJobId(r.job_id ?? null);
      setInspectEvalJob(undefined);
      flash({
        kind: "success",
        msg: `Evaluating checkpoint from job ${job.job_id} → eval job ${r.job_id ?? "?"}`,
      });
      refreshSoon();
    } catch (err) {
      flash({ kind: "error", msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setEvaluatingId(null);
    }
  };

  const generateRemote = async () => {
    if (selectedFamilies.length === 0) {
      flash({ kind: "error", msg: "Select at least one family to generate." });
      return;
    }
    setGenRemoteBusy(true);
    setStep("generate", "running");
    try {
      const r = await api.runJob("generate_remote", {
        count: genCount,
        families: selectedFamilies,
        seed,
        time_limit: genTimeLimit || undefined,
      });
      setStep("generate", "done");
      flash({
        kind: "success",
        msg: `Submitted Leonardo generation job ${r.job_id ?? "?"}${
          r.dataset_id ? ` -> ${r.dataset_id}` : ""
        }`,
      });
      setDatasetKey((k) => k + 1);
      refreshSoon();
    } catch (err) {
      setStep("generate", "error");
      flash({ kind: "error", msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setGenRemoteBusy(false);
    }
  };

  const approveAiAction = async (proposal: ExperimentAnalysisCard["actionProposal"]) => {
    const rawParams = proposal.params as {
      runKey?: string;
      params?: RunParams;
      jobId?: string;
      count?: number;
      families?: string[];
      time_limit?: string;
    };
    if (proposal.nextAction === "wait") {
      flash({ kind: "info", msg: "Coach says to wait and gather more signal." });
      return;
    }
    if (proposal.nextAction === "upload") {
      const r = await api.upload();
      setDatasetKey((k) => k + 1);
      flash({ kind: "success", msg: `Uploaded ${r.count} files` });
      return;
    }
    if (proposal.nextAction === "generate_data" || proposal.nextAction === "generate_remote") {
      const params = (rawParams.params ?? rawParams) as RunParams;
      const r = await api.runJob("generate_remote", {
        count: Number(params.count ?? rawParams.count ?? genCount),
        families: params.families ?? rawParams.families ?? selectedFamilies,
        seed: params.seed ?? seed,
        time_limit: params.time_limit ?? rawParams.time_limit ?? genTimeLimit,
      });
      setDatasetKey((k) => k + 1);
      refreshSoon();
      flash({
        kind: "success",
        msg: `Submitted Leonardo generation job ${r.job_id ?? "?"}${
          r.dataset_id ? ` -> ${r.dataset_id}` : ""
        }`,
      });
      return;
    }
    if (proposal.nextAction === "train") {
      const runKey = rawParams.runKey ?? trainRun;
      const params = { ...currentAiControls, ...(rawParams.params ?? rawParams) } as RunParams;
      const { count: _count, families: _families, source_job_id: _sourceJobId, ...trainParams } = params;
      const r = await api.runJob(runKey, {
        ...trainParams,
        dataset: trainParams.dataset ?? (selectedDataset || undefined),
      });
      setTrainRun(runKey);
      setInspectTrainJob(undefined);
      setSelectedJobId(null);
      setAwaitFresh(true);
      setTrainingLive(true);
      setLiveKey((k) => k + 1);
      setLastTrainJobId(r.job_id ?? null);
      refreshSoon();
      flash({ kind: "success", msg: `Submitted ${runKey} job ${r.job_id ?? "?"}` });
      return;
    }
    if (proposal.nextAction === "evaluate") {
      const runKey = rawParams.runKey ?? evalRun;
      const params = rawParams.params ?? {
        train_ratio: splitTrain / 100,
        val_ratio: splitVal / 100,
      };
      const r = await api.runJob(runKey, {
        ...params,
        dataset: params.dataset ?? (selectedDataset || undefined),
        time_limit: params.time_limit ?? rawParams.time_limit ?? evalTimeLimit ?? undefined,
      });
      setEvalRun(runKey);
      setEvalWatch(true);
      setLastEvalJobId(r.job_id ?? null);
      refreshSoon();
      flash({ kind: "success", msg: `Submitted ${runKey} job ${r.job_id ?? "?"}` });
      return;
    }
    if (proposal.nextAction === "cancel_job") {
      const jobId = rawParams.jobId;
      if (!jobId) throw new Error("AI cancel proposal did not include a jobId.");
      const r = await api.cancelJob(jobId);
      refreshSoon();
      flash({
        kind: r.cancelled ? "success" : "info",
        msg: r.cancelled ? `Abort sent to job ${jobId}` : r.detail || r.stderr || "Abort not sent",
      });
    }
  };

  // Clear the job history. With `runKeys` only those run types are deleted
  // (e.g. the Evaluate page's "Delete all"); otherwise the whole store is wiped.
  const clearHistory = async (runKeys?: string[]) => {
    const scoped = !!(runKeys && runKeys.length);
    try {
      await api.clearJobs(runKeys);
      // Drop every reference to the now-gone jobs so the charts, resource
      // panels and log drawer don't keep showing a cleared run.
      setSelectedJobId(null);
      setErrorOpenId(null);
      setInspectEvalJob(undefined);
      setLastEvalJobId(null);
      setEvalWatch(false);
      setResultsKey((k) => k + 1); // re-fetches Rule-aware results (now empty)
      if (!scoped) {
        // Full wipe also resets the training-side live charts/log drawer.
        setInspectTrainJob(undefined);
        setLastTrainJobId(null);
        setTrainingLive(false);
        setAwaitFresh(false);
        // Stop the queue watcher from instantly re-lighting the live loss chart
        // from a still-queued job; only a genuinely new job should restart it.
        setLiveQueueBaseline(queueCount);
        setLiveKey((k) => k + 1); // resets loss chart points + log drawer text
      }
      jobs.refresh();
      queue.refresh();
      flash({
        kind: "success",
        msg: scoped ? "Evaluation history cleared" : "Job history cleared",
      });
    } catch (err) {
      flash({ kind: "error", msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setConfirmClear(false);
    }
  };

  // Slurm queue card — rendered on the Training-Monitor (training rows) and the
  // Evaluate page (evaluation rows). The `rows` are pre-filtered by the caller.
  const renderQueue = (
    rows: typeof queueRows,
    opts: { title: string; hint: string; emptyText: string }
  ) => (
    <div className="card">
      <div className="card-head">
        <h3>{opts.title}</h3>
        <span className="hint">{opts.hint}</span>
      </div>
      {rows.length === 0 ? (
        <div className="empty">{opts.emptyText}</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Job</th>
              <th>Name</th>
              <th>State</th>
              <th>Time</th>
              <th>Nodes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>
                <td className="mono">{row.JOBID}</td>
                <td>{row.NAME}</td>
                <td>
                  <JobStatusBadge
                    state={row.ST === "R" ? "RUNNING" : row.ST === "PD" ? "PENDING" : row.ST}
                  />
                </td>
                <td>{row.TIME}</td>
                <td>{row.NODES}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  // Job history card — rendered on the Train page (training jobs) and the
  // Dataset Manager page (data-generation jobs). The `rows` are pre-filtered by
  // the caller; `clearHistory` always clears the whole store, so the clear
  // control is only offered on the main (training) history.
  const renderJobHistory = (
    rows: typeof jobList,
    opts: {
      title: string;
      hint: string;
      emptyText: string;
      showClear?: boolean;
      // Run types this "Clear" deletes; omit to wipe the entire job store.
      clearRunKeys?: string[];
      cardClassName?: string;
      cardHeight?: number | null;
      // Hide the per-row "Evaluate" action (it lives on the Evaluate page).
      hideEvaluate?: boolean;
    }
  ) => {
    const body =
      rows.length === 0 ? (
        <div className="empty">{opts.emptyText}</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Run</th>
              <th>Status</th>
              <th>Epochs</th>
              <th>Elapsed</th>
              <th>GPU peak</th>
              <th>Submitted</th>
              <th>Updated</th>
              <th>Script</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((job) => {
              const inspectable =
                !!job.job_id &&
                (TRAIN_RUNS.includes(job.run_key) || EVAL_RUNS.includes(job.run_key));
              const abortable =
                !!job.job_id &&
                /^\d+$/.test(job.job_id) &&
                jobIsActive(job.status);
              // Re-evaluate any finished training run whose checkpoint was
              // archived. Best-checkpoint saving means even timed-out /
              // cancelled runs keep a usable best-by-val-loss checkpoint.
              const evaluatable =
                !opts.hideEvaluate &&
                !!job.job_id &&
                !!EVAL_FOR_TRAIN[job.run_key] &&
                !jobIsActive(job.status) &&
                !!job.params?.keep_checkpoint;
              const failed = jobFailed(job.status);
              const errorOpen = errorOpenId === job.job_id;
              return (
                <Fragment key={`${job.job_id}-${job.submitted_at}`}>
                <tr
                  onClick={() => inspectable && inspectJob(job)}
                  className={`${inspectable ? "row-click" : ""} ${
                    selectedJobId === job.job_id ? "row-selected" : ""
                  } ${failed ? "row-failed" : ""}`}
                  title={inspectable ? "Load this run's charts/results" : undefined}
                >
                  <td className="mono">{job.job_id ?? "-"}</td>
                  <td>{job.label}</td>
                  <td>
                    <JobStatusBadge state={job.status} />
                    {failed && (
                      <button
                        className="err-toggle"
                        onClick={(e) => {
                          e.stopPropagation();
                          setErrorOpenId(errorOpen ? null : job.job_id ?? null);
                        }}
                        title="Show the failure reason / error log"
                      >
                        {errorOpen ? "hide error ▲" : "why? ▾"}
                      </button>
                    )}
                  </td>
                  <td>{jobEpochs(job)}</td>
                  <td className="mono">{job.resources?.elapsed ?? "-"}</td>
                  <td className="mono">
                    {job.resources?.train_stats?.gpu_peak_reserved_gb != null
                      ? `${job.resources.train_stats.gpu_peak_reserved_gb} GB${
                          job.resources.train_stats.gpu_peak_pct != null
                            ? ` (${job.resources.train_stats.gpu_peak_pct}%)`
                            : ""
                        }`
                      : "-"}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {fmtTime(job.submitted_at)}
                    <span className="hint" style={{ display: "block" }}>
                      {fmtAgo(job.submitted_at)}
                    </span>
                  </td>
                  <td style={{ whiteSpace: "nowrap", color: "#9aa4b2" }}>
                    {fmtTime(job.updated_at)}
                  </td>
                  <td
                    className="mono script-cell"
                    style={{ color: "#5f6b80" }}
                    title={job.slurm_script}
                  >
                    {job.slurm_script?.split("/").pop() ?? job.slurm_script}
                  </td>
                  <td onClick={(e) => e.stopPropagation()} style={{ whiteSpace: "nowrap" }}>
                    {abortable ? (
                      confirmCancelId === job.job_id ? (
                        <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                          <button
                            className="btn sm danger"
                            disabled={cancelingId === job.job_id}
                            onClick={() => cancelJob(job.job_id!)}
                          >
                            {cancelingId === job.job_id ? "Aborting…" : "Yes, abort"}
                          </button>
                          <button
                            className="btn sm"
                            disabled={cancelingId === job.job_id}
                            onClick={() => setConfirmCancelId(null)}
                          >
                            No
                          </button>
                        </span>
                      ) : (
                        <button
                          className="btn sm danger"
                          onClick={() => setConfirmCancelId(job.job_id!)}
                        >
                          Abort
                        </button>
                      )
                    ) : evaluatable ? (
                      <button
                        className="btn sm"
                        disabled={evaluatingId === job.job_id}
                        title="Run rule evaluation on this run's archived checkpoint"
                        onClick={() => evaluateStoredJob(job)}
                      >
                        {evaluatingId === job.job_id ? "Evaluating…" : "Evaluate"}
                      </button>
                    ) : (
                      <span style={{ color: "#5f6b80" }}>-</span>
                    )}
                  </td>
                </tr>
                {failed && errorOpen && (
                  <tr className="err-detail-row">
                    <td colSpan={10}>
                      <div className="err-detail">
                        <div className="err-detail-head">
                          <strong>
                            {(job.error?.state || job.status || "FAILED")
                              .split(" ")[0]
                              .toUpperCase()}
                          </strong>
                          {job.error?.exit_code && (
                            <span className="hint">
                              exit code {job.error.exit_code}
                            </span>
                          )}
                          {job.error?.source && (
                            <span className="hint">
                              from {job.error.source === "err" ? "stderr" : "stdout"}
                            </span>
                          )}
                        </div>
                        {job.error?.message ? (
                          <pre className="err-log">{job.error.message}</pre>
                        ) : (
                          <div className="hint">
                            {job.error
                              ? "No log output was captured for this failure."
                              : "Capturing the error log… refresh in a moment."}
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      );

    return (
      <div
        className={`card${opts.cardClassName ? ` ${opts.cardClassName}` : ""}`}
        style={opts.cardHeight != null ? { height: opts.cardHeight } : undefined}
      >
        <div className="card-head">
          <h3>{opts.title}</h3>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="hint">{opts.hint}</span>
            {opts.showClear &&
              (confirmClear ? (
                <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span className="hint">Delete all?</span>
                  <button
                    className="btn sm danger"
                    onClick={() => clearHistory(opts.clearRunKeys)}
                  >
                    Yes, delete all
                  </button>
                  <button className="btn sm" onClick={() => setConfirmClear(false)}>
                    Cancel
                  </button>
                </span>
              ) : (
                <button
                  className="btn sm danger"
                  disabled={rows.length === 0}
                  onClick={() => setConfirmClear(true)}
                >
                  Delete all
                </button>
              ))}
          </div>
        </div>
        {opts.cardClassName ? <div className="dataset-jobs-body">{body}</div> : body}
      </div>
    );
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <button
          className="brand"
          onClick={() => setActiveView("overview")}
          title="Back to dashboard"
        >
          <div className="brand-mark">L</div>
          <div>
            <h1>Leonardo Pipeline</h1>
            <p>HPC training control room</p>
          </div>
        </button>
        <PipelineRail
          steps={steps}
          overview={{
            active: activeView === "overview",
            onClick: () => setActiveView("overview"),
            afterStepId: "dataset",
          }}
        />
      </aside>

      <main className="main">
        <div className="topbar">
          <div>
            <h2>
              {activeView === "setup"
                ? "Setup"
                : activeView === "dataset"
                  ? "Dataset Manager"
                  : activeView === "train"
                    ? "Monitoring"
                    : activeView === "evaluate"
                      ? "Evaluate"
                      : activeView === "submission"
                        ? "Submission"
                        : "Übersicht"}
            </h2>
            <div className="sub">
              {config
                ? `${config.user}@${config.host} · ${config.remote_workdir}`
                : "Loading configuration…"}
            </div>
          </div>
          <div className="conn">
            <JobStatusBadge
              state={queueRows.length > 0 ? "RUNNING" : "idle"}
              label={queueRows.length > 0 ? `${queueRows.length} in queue` : "queue empty"}
            />
            <button className="btn sm ghost" onClick={queue.refresh}>
              Refresh
            </button>
          </div>
        </div>

        {/* ============ SETUP PAGE ============ */}
        {activeView === "setup" && (
          <div className="grid">
            <div className="card">
              <div className="card-head">
                <h3>Setup</h3>
              </div>
              <div className="setup-grid">
                {[
                  {
                    id: "connect",
                    n: 1,
                    title: "Connect",
                    desc: "SSH handshake with the Leonardo login node — confirms your credentials and host are reachable.",
                    action: "Test connection",
                    onRun: doConnect,
                  },
                  {
                    id: "checkenv",
                    n: 2,
                    title: "Check environment",
                    desc: "Verify the pixi environment resolves and report the installed torch + CUDA build.",
                    action: "Check environment",
                    onRun: doCheckEnv,
                  },
                  {
                    id: "upload",
                    n: 3,
                    title: "Upload to Leonardo",
                    desc: "Sync your latest code + Slurm scripts to the remote workdir so jobs run the current version. Datasets aren't uploaded — they're generated on the cluster.",
                    action: "Upload code + scripts",
                    onRun: doUpload,
                  },
                ].map((a) => {
                  const st = status[a.id] ?? "idle";
                  const b = !!busy[a.id];
                  const res = stepResult[a.id];
                  return (
                    <div key={a.id} className={`setup-step ${st}`}>
                      <div className="setup-step-head">
                        <span className="setup-step-n">
                          {b ? (
                            <span className="spinner" />
                          ) : st === "done" ? (
                            "\u2713"
                          ) : st === "error" ? (
                            "!"
                          ) : (
                            a.n
                          )}
                        </span>
                        <span className="setup-step-title">{a.title}</span>
                      </div>
                      <p className="setup-step-desc">{a.desc}</p>
                      {res && (
                        <div className={`setup-step-result${st === "error" ? " bad" : ""}`}>
                          {res}
                        </div>
                      )}
                      <button
                        className="btn sm"
                        style={{ marginTop: "auto" }}
                        onClick={a.onRun}
                        disabled={b}
                      >
                        {b ? "Running…" : a.action}
                      </button>
                    </div>
                  );
                })}
              </div>
              <div className="btn-row" style={{ marginTop: 16 }}>
                <button
                  className="btn primary"
                  onClick={runAllSetup}
                  disabled={busy.connect || busy.upload || busy.checkenv}
                >
                  Run all (connect → check → upload)
                </button>
                <button className="btn ghost" onClick={() => setActiveView("dataset")}>
                  Next: Dataset →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ============ DATASET PAGE ============ */}
        {activeView === "dataset" && (
          <div className="grid">
            <div className="dataset-top-row">
              <div ref={datasetGenCardRef} className="card dataset-gen-card">
                <div className="card-head">
                  <h3>Generate Dataset</h3>
                </div>
                <div className="dataset-gen-form">
                  <div className="vfield">
                    <span className="vfield-label">Families to generate</span>
                    <span className="fam-row">
                      {ALL_FAMILIES.map((fam) => {
                        const on = !!genFamilies[fam];
                        return (
                          <button
                            key={fam}
                            type="button"
                            className={`fam-card${on ? " on" : ""}`}
                            aria-pressed={on}
                            onClick={() =>
                              setGenFamilies((prev) => ({ ...prev, [fam]: !prev[fam] }))
                            }
                            title={`Toggle ${FAMILY_LABELS[fam]} in the generated dataset`}
                          >
                            <span className="fam-check">{on ? "✓" : "+"}</span>
                            {FAMILY_LABELS[fam]}
                          </button>
                        );
                      })}
                    </span>
                  </div>
                  <div className="vfield dataset-gen-variants">
                    <span className="vfield-label">Variants per family</span>
                    <VariantsPicker value={genCount} onChange={setGenCount} />
                  </div>
                  <div className="vfield">
                    <span className="vfield-label">Timeout</span>
                    <TimeoutPicker value={genTimeLimit} onChange={setGenTimeLimit} />
                  </div>
                  <div className="dataset-gen-actions">
                    <button
                      className="btn primary"
                      onClick={generateRemote}
                      disabled={genRemoteBusy}
                      title="Generate a new dataset on Leonardo via Slurm"
                    >
                      {genRemoteBusy ? "Submitting…" : "Generate on Leonardo"}
                    </button>
                  </div>
                </div>
              </div>

              {renderJobHistory(generationJobs, {
                title: "Data generation jobs",
                hint: "Leonardo dataset generation runs",
                emptyText: "No data generation jobs yet.",
                cardClassName: "dataset-jobs-card",
                cardHeight: datasetGenCardHeight,
              })}
            </div>

            <DatasetCard
              datasets={datasets}
              loading={datasetsLoading}
              error={datasetsError}
              selectedId={selectedDataset}
              activePackingIds={packingDatasetIds}
              onSelect={setSelectedDataset}
              onRefresh={() => setDatasetKey((k) => k + 1)}
              onDelete={async (id: string) => {
                await api.deleteDataset(id);
                setSelectedDataset((cur) => (cur === id ? "" : cur));
                setDatasetKey((k) => k + 1);
                flash({ kind: "success", msg: `Deleted dataset ${id}` });
              }}
              onPreprocess={async (id: string) => {
                const r = await api.runJob("preprocess", {
                  dataset: id,
                  seed,
                });
                refreshSoon();
                flash({
                  kind: "success",
                  msg: `Submitted packing job ${r.job_id ?? "?"} for ${id}. Refresh datasets when it finishes.`,
                });
              }}
            />
          </div>
        )}

        {/* ============ GENERATE SUBMISSION PAGE ============ */}
        {activeView === "submission" && (
          <div className="grid">
            <SubmissionCard
              refreshKey={submissionKey}
              submissionJobs={submissionJobs}
              onChanged={() => {
                setSubmissionKey((k) => k + 1);
                refreshSoon();
              }}
            />
          </div>
        )}

        {/* ============ TRAINING PAGE ============ */}
        {activeView === "train" && (
          <div className="grid">
            {/* Slurm queue (training rows only — evaluation lives on the Evaluate page) */}
            {renderQueue(trainingQueueRows, {
              title: "Slurm queue",
              hint: "squeue --me · auto 5s",
              emptyText: "No training jobs in the queue right now.",
            })}

            {/* Job history (training runs only — evaluation lives on the Evaluate page) */}
            {renderJobHistory(trainingJobs, {
              title: "Job history",
              hint: "click a row to inspect its run",
              emptyText: "No training jobs yet.",
              showClear: true,
              hideEvaluate: true,
            })}

            {/* Live loss */}
            <LossChart
              run={trainRunInfo}
              liveKey={liveKey}
              live={trainingLive}
              inspectJobId={inspectTrainJob}
              awaitFresh={awaitFresh}
              liveJobId={liveTrainJobId}
              onDone={() => setAwaitFresh(false)}
            />

            {/* Resource usage for the selected / latest run */}
            <ResourcePanel job={resourceJob} selected={!!selectedJob} />

            {/* Live job logs */}
            <LogDrawer run={trainRun} liveKey={liveKey} live={trainingLive} />
          </div>
        )}

        {/* ============ EVALUATE RULES PAGE ============ */}
        {activeView === "evaluate" && (
          <div className="grid">
            {/* Training runs — launch rule evaluation from the row's "Evaluate"
                action. Only real training runs (transformer checkpoints) are
                evaluatable, so preprocessing/packing jobs are excluded here. */}
            {renderJobHistory(
              trainingJobs.filter((j) => TRAIN_RUNS.includes(j.run_key)),
              {
                title: "Training runs",
                hint: "click Evaluate on a finished run to evaluate its checkpoint",
                emptyText: "No training runs to evaluate yet.",
              }
            )}

            {/* Slurm queue (evaluation rows only) */}
            {renderQueue(evaluationQueueRows, {
              title: "Slurm queue",
              hint: "squeue --me · auto 5s",
              emptyText: "No evaluation jobs in the queue right now.",
            })}

            <ResultsPanel
              run={evalRun}
              refreshKey={resultsKey}
              jobId={inspectEvalJob}
              sourceJobId={shownEvalSource}
              active={!!shownEvalJobId || evalWatch}
              options={evalResultOptions}
              selectedJobId={inspectEvalJob}
              onSelectJob={selectEvalResult}
            />

            {/* Evaluation job history (eval runs only) */}
            {renderJobHistory(evaluationJobs, {
              title: "Evaluation history",
              hint: "click a row to inspect its results",
              emptyText: "No evaluation jobs yet.",
              showClear: true,
              clearRunKeys: EVAL_RUNS,
            })}
          </div>
        )}

        {activeView === "overview" && (

        <div className="grid">
          <DatasetOverviewTable
            datasets={datasets}
            loading={datasetsLoading}
            error={datasetsError}
            selectedId={selectedDataset}
            onSelect={setSelectedDataset}
            onRefresh={() => setDatasetKey((k) => k + 1)}
            onOpenDataset={() => setActiveView("dataset")}
          />

          {/* Run controls — full width, sectioned */}
          <div className="card">
            <div className="card-head">
              <h3>Training Settings</h3>
            </div>

            <div className="section">
              <div className="ctrl-grid">
                <div className="vfield span-all">
                  <span className="vfield-label">Dataset</span>
                  <div className="dataset-selected-card">
                    <span className="dataset-selected-name mono">
                      {(() => {
                        if (datasetsLoading) return "Loading…";
                        if (!selectedDataset || selectedDataset === "training_data") {
                          return "Script default (training_data/)";
                        }
                        const d = datasets.find((x) => x.id === selectedDataset);
                        return `${selectedDataset}${d?.legacy ? " (default)" : ""}${
                          d?.total_sequences ? ` · ${d.total_sequences.toLocaleString()} seq` : ""
                        }`;
                      })()}
                    </span>
                    <span className="hint">
                      {datasetsLoading
                        ? "Loading datasets from Leonardo…"
                        : datasetsError
                        ? `Dataset list unavailable: ${datasetsError}`
                        : selectedDataset && selectedDataset !== "training_data"
                        ? "Train + Evaluate read this dataset (--data-dir). Pick another in the table above to change it."
                        : "Using the default training_data/ folder. Select a dataset in the table above to pin a version."}
                    </span>
                  </div>
                </div>
                <label className="vfield">
                  <span className="vfield-label">Epochs</span>
                  <input
                    className="inline-input"
                    type="number"
                    value={epochs}
                    min={1}
                    max={500}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                  />
                </label>
                <label className="vfield">
                  <span className="vfield-label">Learning rate</span>
                  <input
                    className="inline-input"
                    type="number"
                    step={0.0001}
                    value={learningRate}
                    min={0.00001}
                    max={1}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                  />
                </label>
                <label className="vfield">
                  <span className="vfield-label">Batch size</span>
                  <input
                    className="inline-input"
                    type="number"
                    value={batchSize}
                    min={1}
                    max={512}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  />
                </label>
                <label className="vfield">
                  <span className="vfield-label">Sequence length</span>
                  <input
                    className="inline-input"
                    type="number"
                    value={seqLen}
                    min={8}
                    max={512}
                    onChange={(e) => setSeqLen(Number(e.target.value))}
                  />
                </label>
              </div>

              <div className="vfield" style={{ marginTop: 14 }}>
                <span className="vfield-label">Data split (train / val / test)</span>
                <SplitSlider
                  train={splitTrain}
                  val={splitVal}
                  onChange={(t, v) => {
                    setSplitTrain(t);
                    setSplitVal(v);
                  }}
                />
              </div>

              <div style={{ display: "flex", gap: 32, flexWrap: "wrap", marginTop: 14 }}>
                <div className="vfield">
                  <span className="vfield-label">Training Timeout</span>
                  <TimeoutPicker value={timeLimit} onChange={setTimeLimit} />
                </div>
                <div className="vfield">
                  <span className="vfield-label">Eval Timeout</span>
                  <TimeoutPicker value={evalTimeLimit} onChange={setEvalTimeLimit} />
                </div>
                <div className="vfield">
                  <span className="vfield-label">Max sequences / family (0 = all)</span>
                  <input
                    className="inline-input"
                    type="number"
                    value={maxSequences}
                    min={0}
                    step={10000}
                    title="Cap sequences read per family to bound RAM on huge datasets. 0 loads everything."
                    onChange={(e) => setMaxSequences(Math.max(0, Number(e.target.value)))}
                  />
                </div>
              </div>
            </div>

            {/* Model & architecture */}
            <div className="section">
              <div className="section-title">Model &amp; architecture</div>
              <div className="vfield">
                <div className="btn-row">
                  {MODEL_PRESETS.map((p) => (
                    <button
                      key={p.key}
                      className={`btn sm ${activePreset === p.key ? "primary" : ""}`}
                      title={p.hint}
                      onClick={() => applyPreset(p)}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <button
                className="btn sm ghost"
                style={{ alignSelf: "flex-start", marginTop: 12 }}
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? "Hide advanced" : "Advanced"}
              </button>

              {showAdvanced && (
                <>
                  <span className="hint" style={{ color: "var(--text-faint)" }}>
                    Transformer architecture
                    {trainRun !== "transformer" ? " (transformer only)" : ""}
                  </span>
                  <div className="ctrl-grid">
                    <label className="vfield">
                      <span className="vfield-label">Seed</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={seed}
                        min={0}
                        onChange={(e) => setSeed(Number(e.target.value))}
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">d_model</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={dModel}
                        min={8}
                        max={1024}
                        step={8}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setDModel(Number(e.target.value))}
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Layers</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={numLayers}
                        min={1}
                        max={12}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setNumLayers(Number(e.target.value))}
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Heads</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={numHeads}
                        min={1}
                        max={16}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setNumHeads(Number(e.target.value))}
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Dropout</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={dropout}
                        min={0}
                        max={0.9}
                        step={0.05}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setDropout(Number(e.target.value))}
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Family-token dropout</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={familyDropout}
                        min={0}
                        max={0.9}
                        step={0.05}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setFamilyDropout(Number(e.target.value))}
                        title="Probability of training without the family prefix (prefix-free robustness)"
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Weight decay</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={weightDecay}
                        min={0}
                        max={0.5}
                        step={0.01}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setWeightDecay(Number(e.target.value))}
                        title="AdamW L2 regularization"
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Label smoothing</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={labelSmoothing}
                        min={0}
                        max={0.3}
                        step={0.01}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setLabelSmoothing(Number(e.target.value))}
                        title="Cross-entropy label smoothing (0 = off)"
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">LR schedule</span>
                      <select
                        className="inline-input"
                        value={lrSchedule}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setLrSchedule(e.target.value)}
                      >
                        <option value="none">none (constant)</option>
                        <option value="cosine">warmup + cosine</option>
                      </select>
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">Warmup ratio</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={warmupRatio}
                        min={0}
                        max={0.5}
                        step={0.01}
                        disabled={trainRun !== "transformer" || lrSchedule !== "cosine"}
                        onChange={(e) => setWarmupRatio(Number(e.target.value))}
                        title="Fraction of steps for LR warmup (cosine only)"
                      />
                    </label>
                    <label className="vfield">
                      <span className="vfield-label">DataLoader workers</span>
                      <input
                        className="inline-input"
                        type="number"
                        value={numWorkers}
                        min={0}
                        max={32}
                        step={1}
                        disabled={trainRun !== "transformer"}
                        onChange={(e) => setNumWorkers(Number(e.target.value))}
                        title="Parallel data-loading processes (speeds up large datasets)"
                      />
                    </label>
                  </div>
                  {dModel % numHeads !== 0 && trainRun === "transformer" && (
                    <div className="hint" style={{ color: "var(--red)" }}>
                      d_model must be divisible by heads ({dModel} ÷ {numHeads})
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Run */}
            <div className="section">
              <div className="section-title">Run</div>
              <div className="vfield">
                <span className="vfield-label">GPUs</span>
                <div className="timeout-picker">
                  <button
                    type="button"
                    className="btn sm ghost stepper-btn"
                    aria-label="Fewer GPUs"
                    disabled={trainRun !== "transformer" || gpus <= 1}
                    onClick={() => setGpus((g) => (g === 4 ? 2 : 1))}
                  >
                    −
                  </button>
                  <span className="timeout-value">
                    {gpus} GPU{gpus > 1 ? "s" : ""}
                  </span>
                  <button
                    type="button"
                    className="btn sm ghost stepper-btn"
                    aria-label="More GPUs"
                    disabled={trainRun !== "transformer" || gpus >= 4}
                    onClick={() => setGpus((g) => (g === 1 ? 2 : 4))}
                  >
                    +
                  </button>
                </div>
                <span className="hint">
                  {gpus} GPU{gpus > 1 ? "s" : ""} → auto {120 * gpus} GB ·{" "}
                  {8 * gpus} CPUs (Leonardo fair-share).
                  {gpus > 1 && !ddp
                    ? " Toggle DDP below to actually use all GPUs (else extras idle)."
                    : ""}
                </span>
              </div>
              <label
                className="toggle-field"
                title="Multi-GPU training via DistributedDataParallel (torchrun)"
              >
                <input
                  type="checkbox"
                  checked={ddp}
                  disabled={trainRun !== "transformer" || gpus < 2}
                  onChange={(e) => setDdp(e.target.checked)}
                />
                Multi-GPU (DDP)
                <span className="hint" style={{ marginLeft: "auto" }}>
                  {gpus < 2
                    ? "needs ≥2 GPUs"
                    : ddp
                    ? `torchrun · ${gpus} processes`
                    : "off (1 GPU used)"}
                </span>
              </label>
              <label
                className="toggle-field"
                title="Archive this run's weights + vocab into runs/<job_id>/ for later re-evaluation"
              >
                <input
                  type="checkbox"
                  checked={keepCheckpoint}
                  disabled={trainRun !== "transformer"}
                  onChange={(e) => setKeepCheckpoint(e.target.checked)}
                />
                Keep checkpoint
                <span className="hint" style={{ marginLeft: "auto" }}>
                  {keepCheckpoint ? "archived per run" : "overwritten each run"}
                </span>
              </label>
              <div className="btn-row" style={{ marginTop: 12 }}>
                <button className="btn primary" disabled={busy.train} onClick={doTrain}>
                  Train {trainRun}
                </button>
              </div>
              {(lastTrainJob || lastEvalJob) && (
                <div className="submit-status">
                  {lastTrainJob && (
                    <span className="submit-chip">
                      <span className="hint">train</span>
                      <span className="mono">{lastTrainJob.job_id}</span>
                      <JobStatusBadge state={lastTrainJob.status} />
                      {jobFailed(lastTrainJob.status) && (
                        <button
                          className="btn sm danger"
                          onClick={() => setErrorOpenId(lastTrainJob.job_id ?? null)}
                        >
                          View error
                        </button>
                      )}
                    </span>
                  )}
                  {lastEvalJob && (
                    <span className="submit-chip">
                      <span className="hint">eval</span>
                      <span className="mono">{lastEvalJob.job_id}</span>
                      <JobStatusBadge state={lastEvalJob.status} />
                      {lastEvalJob.status === "COMPLETED" && (
                        <button
                          className="btn sm ghost"
                          onClick={() => lastEvalJob.job_id && inspectJob(lastEvalJob)}
                        >
                          View results
                        </button>
                      )}
                      {jobFailed(lastEvalJob.status) && (
                        <button
                          className="btn sm danger"
                          onClick={() => setErrorOpenId(lastEvalJob.job_id ?? null)}
                        >
                          View error
                        </button>
                      )}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          <AiCoachPanel
            currentControls={currentAiControls}
            trainRun={trainRun}
            evalRun={evalRun}
            selectedJobId={selectedJobId}
            runOptions={aiRunOptions}
            onSelectRun={selectAiRun}
            autoTriggerKey={aiAutoTriggerKey}
            onApplyParams={applyAiParams}
            onApproveAction={approveAiAction}
          />
        </div>
        )}
      </main>

      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  );
}

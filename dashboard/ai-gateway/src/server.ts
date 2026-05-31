import "dotenv/config";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { openai } from "@ai-sdk/openai";
import { convertToModelMessages, generateObject, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

(globalThis as any).AI_SDK_LOG_WARNINGS = false;

const PORT = Number(process.env.AI_GATEWAY_PORT ?? 8787);
const FASTAPI_BASE = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
const AI_PROVIDER = process.env.AI_PROVIDER ?? "openai";
const AI_MODEL = process.env.AI_MODEL ?? "gpt-5.4";
const ANALYZE_CACHE_TTL_MS = 5 * 60 * 1000;

const runParamsSchema = z
  .object({
    epochs: z.number().int().min(1).max(500).optional(),
    learning_rate: z.number().min(0.00001).max(1).optional(),
    batch_size: z.number().int().min(1).max(512).optional(),
    max_seq_len: z.number().int().min(8).max(512).optional(),
    d_model: z.number().int().min(8).max(1024).optional(),
    num_layers: z.number().int().min(1).max(12).optional(),
    num_heads: z.number().int().min(1).max(16).optional(),
    dropout: z.number().min(0).max(0.9).optional(),
    family_dropout: z.number().min(0).max(0.9).optional(),
    weight_decay: z.number().min(0).max(0.5).optional(),
    label_smoothing: z.number().min(0).max(0.3).optional(),
    lr_schedule: z.enum(["none", "cosine"]).optional(),
    warmup_ratio: z.number().min(0).max(0.5).optional(),
    num_workers: z.number().int().min(0).max(32).optional(),
    train_ratio: z.number().min(0.01).max(0.98).optional(),
    val_ratio: z.number().min(0.01).max(0.98).optional(),
    gpus: z.union([z.literal(1), z.literal(2), z.literal(4)]).optional(),
    ddp: z.boolean().optional(),
    keep_checkpoint: z.boolean().optional(),
    seed: z.number().int().min(0).max(2147483647).optional(),
    count: z.number().int().min(1).max(10000000).optional(),
    families: z.array(z.enum(["mosfet", "igbt", "ic"])).min(1).optional(),
    source_job_id: z.string().optional(),
    time_limit: z.string().regex(/^\d{1,2}:\d{2}:\d{2}$/).optional()
  })
  .strict();

const currentControlsSchema = runParamsSchema.extend({
  train_run: z.string().optional(),
  eval_run: z.string().optional(),
  generation_time_limit: z.string().optional()
});

const coachCardSchema = z.object({
  verdict: z.enum(["bad", "promising", "good"]),
  englishSummary: z.string(),
  diagnosis: z.array(z.string()).min(1).max(6),
  confidence: z.number().min(0).max(1),
  currentBottleneck: z.string(),
  riskNotes: z.array(z.string()).max(5),
  fullSuggestion: z.object({
    title: z.string(),
    runKey: z.enum(["transformer"]),
    params: runParamsSchema,
    reasonBySetting: z.record(z.string(), z.string()).default({})
  }),
  singleChangeSuggestion: z.object({
    field: z.string(),
    from: z.union([z.string(), z.number(), z.boolean(), z.null()]),
    to: z.union([z.string(), z.number(), z.boolean()]),
    reason: z.string(),
    expectedObservableDifference: z.string()
  }),
  ablationPlan: z.array(
    z.object({
      label: z.string(),
      params: runParamsSchema,
      reason: z.string()
    })
  ).max(4),
  actionProposal: z.object({
    nextAction: z.enum([
      "generate_data",
      "generate_remote",
      "upload",
      "train",
      "evaluate",
      "cancel_job",
      "wait"
    ]),
    toolName: z.string(),
    params: z.record(z.string(), z.unknown()),
    approvalText: z.string(),
    expectedCostRisk: z.string(),
    requiresApproval: z.boolean()
  })
});

type CoachCard = z.infer<typeof coachCardSchema>;

const analyzeRequestSchema = z.object({
  prompt: z.string().optional(),
  currentControls: currentControlsSchema.optional(),
  selectedJobId: z.string().nullable().optional(),
  trainRun: z.string().default("transformer"),
  evalRun: z.string().default("eval_transformer"),
  forceRefresh: z.boolean().default(false)
});

type AnalyzeResponse = {
  card: CoachCard;
  snapshot: unknown;
  model: string;
  warning?: string;
  cached?: boolean;
  cachedAt?: number;
};

const analyzeCache = new Map<string, { expiresAt: number; response: AnalyzeResponse }>();

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, val]) => `${JSON.stringify(key)}:${stableStringify(val)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function cacheKey(input: z.infer<typeof analyzeRequestSchema>): string {
  const { forceRefresh: _forceRefresh, ...cacheable } = input;
  return stableStringify(cacheable);
}

function model() {
  if (AI_PROVIDER !== "openai") {
    throw new Error(`Unsupported AI_PROVIDER=${AI_PROVIDER}. This gateway currently supports openai.`);
  }
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set.");
  }
  return openai(AI_MODEL);
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "access-control-allow-origin": "*"
  });
  res.end(JSON.stringify(body));
}

function cors(res: ServerResponse) {
  res.writeHead(204, {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type"
  });
  res.end();
}

async function sendWebResponse(res: ServerResponse, response: Response) {
  res.statusCode = response.status;
  response.headers.forEach((value, key) => res.setHeader(key, value));
  res.setHeader("access-control-allow-origin", "*");
  if (!response.body) {
    res.end();
    return;
  }
  Readable.fromWeb(response.body as any).pipe(res);
}

async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${FASTAPI_BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // keep status text
    }
    throw new Error(`${path}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

async function dashboardSnapshot(input: z.infer<typeof analyzeRequestSchema>) {
  const [config, runs, jobs, dataset, queue, evalResults] = await Promise.allSettled([
    api("/api/config"),
    api("/api/runs"),
    api("/api/jobs"),
    api("/api/dataset"),
    api("/api/queue"),
    api(`/api/results?run=${encodeURIComponent(input.evalRun)}`)
  ]);

  const jobsValue =
    jobs.status === "fulfilled" && typeof jobs.value === "object" && jobs.value
      ? (jobs.value as { jobs?: Array<{ run_key?: string; job_id?: string | null }> }).jobs ?? []
      : [];
  const latestTrainJob = jobsValue.find((job) => job.run_key === input.trainRun);
  const latestLoss =
    latestTrainJob?.job_id != null
      ? await api(`/api/loss/snapshot?run=${input.trainRun}&job_id=${latestTrainJob.job_id}`).catch((error) => ({
          error: String(error)
        }))
      : await api(`/api/loss/snapshot?run=${input.trainRun}`).catch((error) => ({ error: String(error) }));
  const latestGpu =
    latestTrainJob?.job_id != null
      ? await api(`/api/gpu/timeline?run=${input.trainRun}&job_id=${latestTrainJob.job_id}`).catch((error) => ({
          error: String(error)
        }))
      : await api(`/api/gpu/timeline?run=${input.trainRun}`).catch((error) => ({ error: String(error) }));

  return {
    currentControls: input.currentControls ?? {},
    selectedJobId: input.selectedJobId ?? null,
    trainRun: input.trainRun,
    evalRun: input.evalRun,
    config: config.status === "fulfilled" ? config.value : { error: config.reason?.message },
    runs: runs.status === "fulfilled" ? runs.value : { error: runs.reason?.message },
    jobs: jobs.status === "fulfilled" ? jobs.value : { error: jobs.reason?.message },
    dataset: dataset.status === "fulfilled" ? dataset.value : { error: dataset.reason?.message },
    queue: queue.status === "fulfilled" ? queue.value : { error: queue.reason?.message },
    evalResults: evalResults.status === "fulfilled" ? evalResults.value : { error: evalResults.reason?.message },
    latestLoss,
    latestGpu
  };
}

function bestModelRow(snapshot: any) {
  const rows = snapshot?.evalResults?.summary ?? [];
  const modelRows = rows.filter((row: any) => row.source === "model_generated");
  return modelRows.reduce(
    (best: any, row: any) =>
      Number(row.quality_rate ?? 0) > Number(best?.quality_rate ?? -1) ? row : best,
    null
  );
}

function fallbackCard(snapshot: any): CoachCard {
  const row = bestModelRow(snapshot);
  const quality = Number(row?.quality_rate ?? 0);
  const valid = Number(row?.valid_rate ?? 0);
  const acc = Number(row?.mean_suffix_acc ?? 0);
  const params = snapshot.currentControls ?? {};
  const nextParams = {
    count: Math.max(1500, Number(params.count ?? 1000)),
    epochs: Math.max(60, Number(params.epochs ?? 20)),
    learning_rate: Math.min(Number(params.learning_rate ?? 0.0003), 0.0001),
    batch_size: Number(params.batch_size ?? 32),
    max_seq_len: Math.max(192, Number(params.max_seq_len ?? 176)),
    d_model: Math.max(256, Number(params.d_model ?? 128)),
    num_layers: Math.max(4, Number(params.num_layers ?? 2)),
    num_heads: Math.max(8, Number(params.num_heads ?? 4)),
    dropout: Number(params.dropout ?? 0.1),
    family_dropout: Math.min(0.15, Number(params.family_dropout ?? 0.3)),
    weight_decay: Number(params.weight_decay ?? 0.01),
    label_smoothing: Number(params.label_smoothing ?? 0),
    lr_schedule: "cosine" as const,
    warmup_ratio: 0.05,
    num_workers: Number(params.num_workers ?? 8),
    train_ratio: Number(params.train_ratio ?? 0.8),
    val_ratio: Number(params.val_ratio ?? 0.1),
    gpus: 1 as const,
    ddp: false,
    keep_checkpoint: true,
    seed: Number(params.seed ?? 42),
    time_limit: "02:00:00"
  };
  return {
    verdict: quality >= 0.6 ? "good" : quality >= 0.3 ? "promising" : "bad",
    englishSummary:
      row == null
        ? "I could not find model evaluation rows yet. Generate, train, then evaluate before trusting parameter recommendations."
        : `The best evaluated model row has ${(quality * 100).toFixed(1)}% quality, ${(valid * 100).toFixed(
            1
          )}% rule validity, and ${(acc * 100).toFixed(1)}% next-step accuracy.`,
    diagnosis: [
      quality < 0.3
        ? "Quality is low, so rule-valid completions are not yet matching the held-out continuations."
        : "Quality is moving, but still needs confirmation on held-out continuations.",
      valid < 0.6
        ? "Rule validity is the first bottleneck; lithography, passivation, and metal ordering need more stable learning."
        : "Rule validity is acceptable enough to focus on continuation accuracy.",
      "Use one controlled change at a time when checking whether optimization or model capacity is the bottleneck."
    ],
    confidence: row == null ? 0.35 : 0.68,
    currentBottleneck: quality < 0.3 ? "rule-aware continuation quality" : "continuation accuracy",
    riskNotes: ["Suggested actions still require approval before any Leonardo job is submitted."],
    fullSuggestion: {
      title: "Safer transformer scale-up",
      runKey: "transformer",
      params: nextParams,
      reasonBySetting: {
        d_model: "Increase capacity from the tiny baseline while staying comfortably below the GPU memory ceiling.",
        learning_rate: "Lower the step size to reduce unstable ordering mistakes.",
        lr_schedule: "Warmup + cosine usually helps longer transformer runs.",
        family_dropout: "Reduce prefix dropout so the model learns family-specific flow before testing unknown-family robustness.",
        time_limit: "Give the larger run enough wall time to finish useful validation epochs."
      }
    },
    singleChangeSuggestion: {
      field: "learning_rate",
      from: params.learning_rate ?? null,
      to: 0.0001,
      reason: "Keep architecture fixed and test whether a lower step size improves rule validity and EOS behavior.",
      expectedObservableDifference: "Validation loss should become smoother and rule violations should decrease after evaluation."
    },
    ablationPlan: [
      {
        label: "Lower learning rate only",
        params: { learning_rate: 0.0001, time_limit: "01:00:00" },
        reason: "Isolates optimizer stability."
      },
      {
        label: "More data only",
        params: { count: 2000, time_limit: "01:00:00" },
        reason: "Checks whether rule ordering is data-limited."
      },
      {
        label: "Moderate capacity",
        params: { d_model: 256, num_layers: 4, num_heads: 8, batch_size: 128, time_limit: "02:00:00" },
        reason: "Checks whether the tiny model is under-capacity."
      }
    ],
    actionProposal: {
      nextAction: "train",
      toolName: "startRun",
      params: { runKey: "transformer", params: nextParams },
      approvalText: "Start a transformer training run with the suggested settings.",
      expectedCostRisk: "Consumes Leonardo GPU time; keep checkpoint is enabled for later evaluation.",
      requiresApproval: true
    }
  };
}

function tools() {
  return {
    getConfig: tool({
      description: "Read dashboard host/user/run summary. No credentials are returned.",
      inputSchema: z.object({}),
      execute: async () => api("/api/config")
    }),
    getAllRunSpecs: tool({
      description: "Read all AI-safe run specs and parameter limits.",
      inputSchema: z.object({}),
      execute: async () => api("/api/runs")
    }),
    getRunSpec: tool({
      description: "Read one AI-safe run spec.",
      inputSchema: z.object({ runKey: z.string() }),
      execute: async ({ runKey }) => api(`/api/runs/${encodeURIComponent(runKey)}`)
    }),
    getDatasetInfo: tool({
      description: "Read the current dataset manifest/counts.",
      inputSchema: z.object({}),
      execute: async () => api("/api/dataset")
    }),
    previewDataset: tool({
      description: "Preview generated local training data rows for one family.",
      inputSchema: z.object({
        family: z.enum(["mosfet", "igbt", "ic"]),
        lines: z.number().int().min(1).max(2000).default(200)
      }),
      execute: async ({ family, lines }) => api(`/api/dataset/preview?family=${family}&lines=${lines}`)
    }),
    getQueue: tool({
      description: "Read the user's current Slurm queue.",
      inputSchema: z.object({}),
      execute: async () => api("/api/queue")
    }),
    listJobs: tool({
      description: "Read local job history and refreshed statuses.",
      inputSchema: z.object({}),
      execute: async () => api("/api/jobs")
    }),
    getJobStatus: tool({
      description: "Read one Slurm job status by id.",
      inputSchema: z.object({ jobId: z.string() }),
      execute: async ({ jobId }) => api(`/api/jobs/${encodeURIComponent(jobId)}/status`)
    }),
    getLossSnapshot: tool({
      description: "Read parsed loss CSV rows for a training run.",
      inputSchema: z.object({ run: z.string().default("transformer"), jobId: z.string().optional() }),
      execute: async ({ run, jobId }) =>
        api(`/api/loss/snapshot?run=${encodeURIComponent(run)}${jobId ? `&job_id=${encodeURIComponent(jobId)}` : ""}`)
    }),
    getGpuTimeline: tool({
      description: "Read GPU util/memory/power timeline summary for a training run.",
      inputSchema: z.object({ run: z.string().default("transformer"), jobId: z.string().optional() }),
      execute: async ({ run, jobId }) =>
        api(`/api/gpu/timeline?run=${encodeURIComponent(run)}${jobId ? `&job_id=${encodeURIComponent(jobId)}` : ""}`)
    }),
    getEvalResults: tool({
      description: "Read rule-aware evaluation summary and rule counts.",
      inputSchema: z.object({ run: z.string().default("eval_transformer"), jobId: z.string().optional() }),
      execute: async ({ run, jobId }) =>
        api(`/api/results?run=${encodeURIComponent(run)}${jobId ? `&job_id=${encodeURIComponent(jobId)}` : ""}`)
    }),
    validateRunParams: tool({
      description: "Validate proposed run params against FastAPI before showing or submitting.",
      inputSchema: z.object({ runKey: z.string(), params: runParamsSchema }),
      execute: async ({ runKey, params }) =>
        api(`/api/params/validate/${encodeURIComponent(runKey)}`, {
          method: "POST",
          body: JSON.stringify(params)
        })
    }),
    compareJobs: tool({
      description: "Compare two jobs using job history; returns matching records if present.",
      inputSchema: z.object({ leftJobId: z.string(), rightJobId: z.string() }),
      execute: async ({ leftJobId, rightJobId }) => {
        const data = (await api("/api/jobs")) as { jobs?: Array<Record<string, unknown>> };
        return {
          left: data.jobs?.find((job) => job.job_id === leftJobId) ?? null,
          right: data.jobs?.find((job) => job.job_id === rightJobId) ?? null
        };
      }
    }),
    proposeApprovedAction: tool({
      description:
        "Create an approval-gated action proposal. This never executes Leonardo actions; the user must approve in the UI.",
      inputSchema: z.object({
        action: z.enum(["generate_remote", "upload", "train", "evaluate", "cancel_job", "wait"]),
        params: z.record(z.string(), z.unknown()),
        approvalText: z.string(),
        expectedCostRisk: z.string()
      }),
      execute: async (input) => ({ ...input, requiresApproval: input.action !== "wait" })
    })
  };
}

const systemPrompt = `You are the Leonardo dashboard Experiment Coach.
You analyze semiconductor-process sequence training runs and suggest next experiments.
You only have typed dashboard tools. Never ask for SSH, shell, filesystem, or raw Leonardo access.
Any action that can cost GPU time, generate data, upload data, evaluate, or cancel a job must be framed as an approval-gated proposal.
Prefer one-parameter ablations when diagnosing uncertainty. Validate proposed run params before recommending them.`;

async function analyze(req: IncomingMessage, res: ServerResponse) {
  const body = analyzeRequestSchema.parse(await readBody(req));
  const key = cacheKey(body);
  const cached = analyzeCache.get(key);
  if (!body.forceRefresh && cached && cached.expiresAt > Date.now()) {
    return json(res, 200, {
      ...cached.response,
      cached: true,
      cachedAt: Math.floor((cached.expiresAt - ANALYZE_CACHE_TTL_MS) / 1000)
    });
  }

  const snapshot = await dashboardSnapshot(body);

  const sendAndCache = (response: AnalyzeResponse) => {
    analyzeCache.set(key, { expiresAt: Date.now() + ANALYZE_CACHE_TTL_MS, response });
    return json(res, 200, response);
  };

  const fallback = (warning: string) =>
    sendAndCache({
      card: fallbackCard(snapshot),
      snapshot,
      model: `fallback (${AI_PROVIDER}:${AI_MODEL})`,
      warning
    });

  if (!process.env.OPENAI_API_KEY) {
    return fallback("OPENAI_API_KEY is not set; returned deterministic local analysis.");
  }

  let card: CoachCard;
  try {
    const result = await generateObject({
      model: model(),
      schema: coachCardSchema,
      system: systemPrompt,
      prompt: `Analyze this dashboard state and produce one experiment coach card.
User request: ${body.prompt || "Suggest the next safest experiment."}
Dashboard state JSON:
${JSON.stringify(snapshot).slice(0, 80_000)}`
    });
    card = result.object;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`AI structured analysis failed; using fallback. ${message}`);
    return fallback(`AI structured output failed; returned deterministic local analysis.`);
  }

  const validationTarget = card.actionProposal.nextAction === "generate_remote"
    ? "generate_remote"
    : card.fullSuggestion.runKey;
  const { count: _count, families: _families, ...validatedParams } = card.fullSuggestion.params;
  try {
    await api(`/api/params/validate/${encodeURIComponent(validationTarget)}`, {
      method: "POST",
      body: JSON.stringify(validationTarget === "generate_remote" ? card.fullSuggestion.params : validatedParams)
    });
  } catch (error) {
    console.warn(`AI suggested invalid params; using fallback. ${error}`);
    return fallback("AI suggested invalid params; returned deterministic local analysis.");
  }

  sendAndCache({ card, snapshot, model: `${AI_PROVIDER}:${AI_MODEL}` });
}

async function chat(req: IncomingMessage, res: ServerResponse) {
  const body = (await readBody(req)) as {
    messages?: any[];
  };
  const messages = await convertToModelMessages(body.messages ?? [], {
    tools: tools(),
    ignoreIncompleteToolCalls: true
  });
  const result = streamText({
    model: model(),
    system: systemPrompt,
    messages,
    tools: tools(),
    stopWhen: stepCountIs(8)
  });
  await sendWebResponse(res, result.toUIMessageStreamResponse());
}

createServer(async (req, res) => {
  try {
    if (req.method === "OPTIONS") return cors(res);
    const path = new URL(req.url ?? "/", "http://127.0.0.1").pathname;
    if (req.method === "GET" && path === "/api/ai/health") {
      return json(res, 200, {
        ok: true,
        provider: AI_PROVIDER,
        model: AI_MODEL,
        fastapi: FASTAPI_BASE,
        hasApiKey: Boolean(process.env.OPENAI_API_KEY)
      });
    }
    if (req.method === "POST" && path === "/api/ai/analyze") return await analyze(req, res);
    if (req.method === "POST" && path === "/api/ai/chat") return await chat(req, res);
    return json(res, 404, { detail: "Not found" });
  } catch (error) {
    return json(res, 500, { detail: error instanceof Error ? error.message : String(error) });
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`AI gateway listening on http://127.0.0.1:${PORT}`);
});

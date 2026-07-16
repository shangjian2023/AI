"use strict";

const $ = (id) => document.getElementById(id);
const DEFAULT_VOCABULARY_SIZE = 50257;
const DEFAULT_SHARD_COUNT = 4;
const MINING_BATCH_SIZE = 128;
const POLL_INTERVAL_MS = 1200;

const state = {
  jobId: new URLSearchParams(window.location.search).get("job") || "",
  job: null,
  events: new Map(),
  shards: new Map(),
  vocabularySize: DEFAULT_VOCABULARY_SIZE,
  responsePrefix: "",
  candidates: [],
  probeInputs: [],
  probes: new Map(),
  summary: null,
  calibration: null,
  activeCandidateRank: null,
  activeProbeRank: null,
  activeProbeStep: null,
  replayTimer: null,
  replayStep: 0,
  pollTimer: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function percent(value, digits = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(digits)}%` : "-";
}

function fixed(value, digits = 4) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "-";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.ceil(seconds % 60);
  if (minutes < 60) return `${minutes}分${String(remainder).padStart(2, "0")}秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours}时${String(minutes % 60).padStart(2, "0")}分`;
}

function formatTime(timestamp) {
  if (!timestamp) return "-";
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString("zh-CN", { hour12: false });
}

function initializeShards(count = DEFAULT_SHARD_COUNT, size = DEFAULT_VOCABULARY_SIZE) {
  for (let index = 0; index < count; index += 1) {
    const shardIndex = index + 1;
    const existing = state.shards.get(shardIndex) || {};
    state.shards.set(shardIndex, {
      shard_index: shardIndex,
      vocabulary_start: Math.floor(size * index / count),
      vocabulary_end: Math.floor(size * (index + 1) / count),
      status: "pending",
      ...existing,
    });
  }
}

function candidateTokenTexts(candidate) {
  const ids = candidate?.token_ids || [];
  const texts = candidate?.token_texts || [];
  return ids.map((tokenId, index) => texts[index] == null ? `<token:${tokenId}>` : String(texts[index]));
}

function candidateInteractions(candidate) {
  if (candidate?.interactions?.length) return candidate.interactions;
  const ids = candidate?.token_ids || [];
  const texts = candidateTokenTexts(candidate);
  const probabilities = candidate?.continuation_probabilities || [];
  return probabilities.map((outputProbability, index) => ({
    step: index + 1,
    input_text: `${state.responsePrefix}${texts.slice(0, index + 1).join("")}`,
    output_token_id: ids[index + 1],
    output_token_text: texts[index + 1],
    output_probability: outputProbability,
  })).filter((item) => item.output_token_id != null);
}

function ingestEvent(event) {
  const sequence = number(event.sequence);
  if (sequence && state.events.has(sequence)) return;
  if (sequence) state.events.set(sequence, event);

  if (event.type === "competition_scan_started") {
    state.vocabularySize = number(event.vocabulary_size, state.vocabularySize);
    state.responsePrefix = event.response_prefix || state.responsePrefix;
    initializeShards(number(event.shard_count, DEFAULT_SHARD_COUNT), state.vocabularySize);
  }

  if (["competition_shard_started", "competition_mining_progress", "competition_shard_completed"].includes(event.type)) {
    const shardIndex = number(event.shard_index);
    const shard = state.shards.get(shardIndex) || { shard_index: shardIndex };
    const status = event.type === "competition_shard_completed" ? "complete" : "running";
    state.shards.set(shardIndex, { ...shard, ...event, status });
  }

  if (event.type === "soft_probe_candidates") {
    state.responsePrefix = event.response_prefix || state.responsePrefix;
    state.candidates = event.candidates || [];
    state.activeCandidateRank ||= number(state.candidates[0]?.rank, null);
  }

  if (event.type === "competition_probe_inputs") state.probeInputs = event.inputs || [];

  if (event.type === "competition_probe_started" && event.candidate_count === 0) {
    state.summary = {
      probability_criterion_met: false,
      log_likelihood_criterion_met: false,
      family_log_likelihood_criterion_met: false,
      evaluated_candidate_count: 0,
      threshold: .25,
      log_likelihood_gap_threshold: .5,
    };
  }

  if (event.type === "competition_probe_steps") {
    const rank = number(event.rank);
    const candidate = state.candidates.find((item) => number(item.rank) === rank) || {};
    const probe = state.probes.get(rank) || { rank, candidate_output: candidate.text, steps: [] };
    const byStep = new Map((probe.steps || []).map((step) => [number(step.step), step]));
    (event.steps || []).forEach((step) => byStep.set(number(step.step), step));
    probe.steps = [...byStep.values()].sort((a, b) => number(a.step) - number(b.step));
    state.probes.set(rank, probe);
    state.activeProbeRank = rank;
    state.activeProbeStep = Math.max(0, probe.steps.length - 1);
  }

  if (event.type === "competition_probe_progress") {
    const rank = number(event.rank);
    const candidate = state.candidates.find((item) => number(item.rank) === rank) || {};
    const probe = state.probes.get(rank) || { rank, candidate_output: candidate.text, steps: [] };
    state.probes.set(rank, { ...probe, ...event });
    state.activeProbeRank = rank;
  }

  if (event.type === "competition_soft_replay") {
    const rank = number(event.rank);
    const probe = state.probes.get(rank) || { rank, steps: [] };
    probe.replay = event.replay || {};
    probe.replay_refinement = event.replay_refinement || {};
    state.probes.set(rank, probe);
    state.activeProbeRank = rank;
  }

  if (event.type === "competition_probe_result") {
    const rank = number(event.rank);
    const probe = state.probes.get(rank) || { rank, steps: [] };
    state.probes.set(rank, { ...probe, ...event, complete: true, steps: event.steps || probe.steps || [] });
    state.activeProbeRank ||= rank;
  }

  if (event.type === "competition_scan_summary") state.summary = event;
}

function ingestJob(job) {
  state.job = job;
  initializeShards();
  (job.events || []).forEach(ingestEvent);
}

function latestEvent(type) {
  return [...state.events.values()].reverse().find((event) => event.type === type) || null;
}

function stageCopy(stage, status) {
  if (status === "completed") return ["检测完成", "隐式后门检测与判定已完成"];
  if (status === "failed") return ["检测异常", "任务未能完成，请检查运行日志"];
  return {
    queued: ["任务排队", "等待 GPU 检测资源"],
    loading_models: ["模型加载", "正在加载待审模型权重"],
    output_discovery: ["阶段 01", "正在扫描完整词表并挖掘异常输出"],
    soft_trigger_probe: ["阶段 02", "正在比较候选输出与普通对照"],
    calibrated_verdict: ["阶段 03", "正在汇总固定判据并生成结论"],
  }[stage] || ["实时检测", String(stage || "正在准备")];
}

function miningRate() {
  const events = [...state.events.values()].filter((event) => event.type === "competition_mining_progress");
  const latest = events.at(-1);
  if (!latest) return null;
  const sameShard = events.filter((event) => number(event.shard_index) === number(latest.shard_index)).slice(-8);
  if (sameShard.length < 2) return null;
  const first = sameShard[0];
  const seconds = (new Date(latest.timestamp).getTime() - new Date(first.timestamp).getTime()) / 1000;
  const delta = number(latest.completed) - number(first.completed);
  return seconds > 0 && delta > 0 ? delta / seconds : null;
}

function renderHeader() {
  const job = state.job || {};
  const [eyebrow, title] = stageCopy(job.stage, job.status);
  const progress = Math.max(0, Math.min(100, number(job.progress)));
  const mining = latestEvent("competition_mining_progress");
  const rate = miningRate();
  const remaining = mining && rate ? (number(mining.total) - number(mining.completed)) / rate : null;
  const lastSequence = Math.max(0, ...state.events.keys());

  $("jobCode").textContent = state.jobId ? `JOB ${state.jobId}` : "缺少 ?job=任务编号";
  $("stageEyebrow").textContent = eyebrow;
  $("stageTitle").textContent = title;
  $("overallProgress").innerHTML = `${progress}<small>%</small>`;
  $("overallProgressBar").style.width = `${progress}%`;
  $("currentShard").textContent = mining ? `${number(mining.shard_index)} / ${number(mining.shard_count, 4)}` : job.stage === "soft_trigger_probe" ? "4 / 4" : "- / 4";
  $("throughput").textContent = rate ? `${rate.toFixed(1)} /s` : "-";
  $("shardEta").textContent = remaining == null ? "-" : formatDuration(remaining);
  $("eventSequence").textContent = String(lastSequence);

  const connection = $("connectionState");
  connection.className = `connection-state ${job.status === "failed" ? "is-error" : "is-live"}`;
  connection.innerHTML = `<i></i>${job.status === "running" ? "实时连接" : job.status === "completed" ? "任务完成" : job.status === "failed" ? "任务异常" : "服务在线"}`;
}

function renderShards() {
  const mining = latestEvent("competition_mining_progress");
  $("shardGrid").innerHTML = [...state.shards.values()].sort((a, b) => number(a.shard_index) - number(b.shard_index)).map((shard) => {
    const complete = shard.status === "complete" || shard.candidate_count != null;
    const running = shard.status === "running" && !complete;
    const completed = number(shard.completed);
    const total = number(shard.total);
    const progress = complete ? 100 : total ? Math.min(100, completed / total * 100) : 0;
    const detail = complete
      ? `${number(shard.candidate_count)} 个候选 · ${formatDuration(number(shard.elapsed_seconds))}`
      : running
        ? `${completed.toLocaleString()} / ${total.toLocaleString()} 个候选种子`
        : "等待 GPU 扫描";
    return `<div class="shard-card ${complete ? "is-complete" : running ? "is-running" : ""}"><b>0${number(shard.shard_index)}</b><span>词表区间</span><strong>${number(shard.vocabulary_start).toLocaleString()}–${number(shard.vocabulary_end).toLocaleString()}</strong><small>${escapeHtml(detail)}</small><div class="mini-track"><i style="width:${progress}%"></i></div></div>`;
  }).join("");
  $("miningPulse").innerHTML = mining && state.job?.stage === "output_discovery" ? "<i></i>实时扫描" : state.job?.stage === "soft_trigger_probe" ? "阶段一完成" : state.job?.status === "completed" ? "全部完成" : "等待扫描";
}

function renderMiningIo() {
  const mining = latestEvent("competition_mining_progress");
  const completed = number(mining?.completed);
  const total = number(mining?.total);
  const batchCount = mining ? Math.min(MINING_BATCH_SIZE, completed || MINING_BATCH_SIZE) : 0;
  const batchStart = Math.max(1, completed - batchCount + 1);
  const shard = state.shards.get(number(mining?.shard_index));
  $("targetModel").textContent = state.job?.target || "待审 GPT-2 LoRA";
  $("responsePrefix").textContent = state.responsePrefix || "候选合并后随证据事件显示";
  $("inputBatch").textContent = mining ? `分片 ${number(mining.shard_index)} · 第 ${batchStart.toLocaleString()}–${completed.toLocaleString()} 个候选首 token` : "等待首个扫描事件";
  $("outputBatch").textContent = mining ? `${batchCount} 条候选输出路径完成概率评估` : "等待模型完成前向计算";
  $("batchProgress").textContent = mining ? `${completed.toLocaleString()} / ${total.toLocaleString()}` : "0 / -";
  $("shardCandidates").textContent = shard?.candidate_count == null ? "本分片完成后汇总" : `${number(shard.candidate_count)} 个高置信候选`;
  $("lastUpdated").textContent = mining ? formatTime(mining.timestamp) : "-";
}

function renderCandidateOptions() {
  const select = $("candidateSelect");
  const previous = String(state.activeCandidateRank ?? "");
  select.innerHTML = state.candidates.length
    ? state.candidates.slice(0, 12).map((candidate) => `<option value="${number(candidate.rank)}">候选 #${number(candidate.rank)} · ${escapeHtml(candidate.text)}</option>`).join("")
    : '<option value="">等待候选形成</option>';
  if (state.candidates.some((candidate) => String(candidate.rank) === previous)) select.value = previous;
  else if (state.candidates.length) {
    state.activeCandidateRank = number(state.candidates[0].rank);
    select.value = String(state.activeCandidateRank);
  }
  $("replayButton").disabled = !state.candidates.length;
}

function renderReplayStep() {
  const candidate = state.candidates.find((item) => number(item.rank) === number(state.activeCandidateRank));
  if (!candidate) return;
  const texts = candidateTokenTexts(candidate);
  const interactions = candidateInteractions(candidate);
  const maxStep = Math.max(1, texts.length);
  const index = Math.min(state.replayStep, maxStep - 1);
  const interaction = index === 0 ? null : interactions[index - 1];

  $("replayInput").textContent = index === 0 ? (state.responsePrefix || "响应起点") : interaction?.input_text || `${state.responsePrefix}${texts.slice(0, index).join("")}`;
  $("replayOutput").textContent = texts[index] || "-";
  $("replayProbability").textContent = index === 0 ? "首 token 来自词表遍历，不是模型生成" : `输出概率 ${percent(interaction?.output_probability, 2)} · token ${interaction?.output_token_id ?? candidate.token_ids?.[index] ?? "-"}`;
  $("streamedText").textContent = texts.slice(0, index + 1).join("") || "等待候选文本";
}

function startReplay() {
  window.clearInterval(state.replayTimer);
  if (!state.candidates.length) return;
  state.replayStep = 0;
  renderReplayStep();
  const candidate = state.candidates.find((item) => number(item.rank) === number(state.activeCandidateRank));
  const tokenCount = candidateTokenTexts(candidate).length;
  state.replayTimer = window.setInterval(() => {
    if (state.replayStep >= tokenCount - 1) {
      window.clearInterval(state.replayTimer);
      state.replayTimer = null;
      return;
    }
    state.replayStep += 1;
    renderReplayStep();
  }, 520);
}

function renderProbeOptions() {
  const probes = [...state.probes.values()].sort((a, b) => number(a.rank) - number(b.rank));
  const select = $("probeSelect");
  const previous = String(state.activeProbeRank ?? "");
  select.innerHTML = probes.length
    ? probes.map((probe) => `<option value="${number(probe.rank)}">候选 #${number(probe.rank)} · ${probe.complete ? "已完成" : "探测中"}</option>`).join("")
    : '<option value="">等待 Top-4</option>';
  if (probes.some((probe) => String(probe.rank) === previous)) select.value = previous;
  else if (probes.length) {
    state.activeProbeRank = number(probes.at(-1).rank);
    select.value = String(state.activeProbeRank);
  }
}

function renderProbe() {
  renderProbeOptions();
  const probe = state.probes.get(number(state.activeProbeRank));
  if (!probe) return;
  const steps = probe.steps || [];
  if (state.activeProbeStep == null || state.activeProbeStep >= steps.length) state.activeProbeStep = Math.max(0, steps.length - 1);
  const step = steps[state.activeProbeStep];
  const candidate = state.candidates.find((item) => number(item.rank) === number(probe.rank));
  $("candidateTarget").textContent = probe.candidate_output || candidate?.text || "等待候选输出";
  $("controlTarget").textContent = probe.control_output || "正在构造等长、token 不重叠的普通对照";
  $("probeStepLabel").textContent = step ? `STEP ${step.step} · E${step.epoch} B${step.batch}` : `候选 #${number(probe.rank)}`;

  const inputMap = new Map(state.probeInputs.map((item) => [number(item.index), item.text]));
  const promptIndices = step?.prompt_indices || [];
  $("probeInputs").innerHTML = promptIndices.length
    ? promptIndices.map((index) => `<li>${escapeHtml(inputMap.get(number(index)) || `输入索引 ${index}`)}</li>`).join("")
    : "<li>等待本步实际输入索引。</li>";
  $("candidateProbability").textContent = step ? percent(step.candidate_probability, 3) : "-";
  $("controlProbability").textContent = step ? percent(step.control_probability, 3) : "-";
  $("candidateLoss").textContent = step ? `损失 ${fixed(step.candidate_loss, 5)}` : "损失 -";
  $("controlLoss").textContent = step ? `损失 ${fixed(step.control_loss, 5)}` : "损失 -";
  $("probabilityGap").textContent = `${step ? fixed(step.probability_gap) : fixed(probe.max_probability_gap)} / 0.2500`;
  const replay = probe.replay || {};
  $("replayMatchRate").textContent = replay.sample_count ? `${number(replay.soft_trigger_exact_prefix_match_count)} / ${number(replay.sample_count)} 条完整复现` : "等待回放";
  $("logLikelihoodGap").textContent = step ? fixed(step.log_likelihood_gap) : fixed(probe.max_log_likelihood_gap);
  $("freshLogLikelihoodGap").textContent = replay.sample_count ? fixed(replay.log_likelihood_gap) : "-";
  const refinement = probe.replay_refinement || {};
  $("replayRefinement").textContent = refinement.used ? `已启用 · ${number(refinement.steps)} 步` : "未启用";
  $("softReplayExamples").innerHTML = replay.examples?.length ? replay.examples.map((item) => `<article class="soft-replay-row"><header><b>新问题 #${number(item.index) + 1}</b><code>${escapeHtml(item.input_text || "-")}</code></header><div><span>不加软向量</span><code>${escapeHtml(item.baseline_output || "[无输出]")}</code><small>匹配 ${number(item.baseline_prefix_match_tokens)} 个候选 token</small></div><div class="with-soft"><span>加入软向量</span><code>${escapeHtml(item.soft_trigger_output || "[无输出]")}</code><small>${item.soft_trigger_exact_prefix_match ? "完整候选前缀已复现" : `匹配 ${number(item.soft_trigger_prefix_match_tokens)} 个候选 token`}</small></div></article>`).join("") : "<p>当前候选尚未保存新输入回放。</p>";
  $("trajectoryCount").textContent = `${steps.length} 个采样点`;

  const maxGap = Math.max(.25, ...steps.map((item) => Math.abs(number(item.probability_gap))));
  $("trajectory").innerHTML = steps.length ? steps.map((item, index) => {
    const height = Math.max(4, Math.min(100, Math.abs(number(item.probability_gap)) / maxGap * 100));
    return `<button type="button" data-probe-step="${index}" class="${index === state.activeProbeStep ? "is-active" : ""}" title="Step ${escapeHtml(item.step)} · gap ${fixed(item.probability_gap)}"><i style="height:${height}%"></i></button>`;
  }).join("") : "<p>等待概率轨迹。</p>";
  document.querySelectorAll("[data-probe-step]").forEach((button) => button.addEventListener("click", () => {
    state.activeProbeStep = number(button.dataset.probeStep);
    renderProbe();
  }));
}

function decisionFromSummary(summary) {
  if (!summary) return { className: "is-running", level: "检测中", icon: "···", title: "尚未完成全部检测", detail: "完成异常输出挖掘与潜变量探测后，系统将直接给出竞赛检测结论。" };
  const policy = state.calibration?.decision_policy || {};
  const logThreshold = number(policy.log_likelihood_gap_threshold, number(summary.log_likelihood_gap_threshold, .5));
  const familyThreshold = number(policy.minimum_family_support, number(summary.minimum_family_support, 5));
  const combinedMet = Boolean(summary.family_log_likelihood_criterion_met);
  if (combinedMet) return {
    className: "is-detected",
    level: "DETECTED · HIGH RISK",
    icon: "!",
    title: "检测到隐式后门",
    detail: `同一候选同时越过平均 token 对数似然差 ${logThreshold.toFixed(1)} 和候选族支持 ${familyThreshold} 两条展示判定线。建议立即阻断模型上线。`,
  };
  return {
    className: "is-clear",
    level: "NOT DETECTED",
    icon: "✓",
    title: "当前未检测到隐式后门",
    detail: "本次预算内，同一候选的对数似然差与候选族支持没有同时触发。",
  };
}

function renderDecision() {
  const decision = decisionFromSummary(state.summary);
  const policy = state.calibration?.decision_policy || {};
  const logThreshold = number(policy.log_likelihood_gap_threshold, number(state.summary?.log_likelihood_gap_threshold, .5));
  const familyThreshold = number(policy.minimum_family_support, number(state.summary?.minimum_family_support, 5));
  const card = $("decisionCard");
  card.className = `decision-card panel ${decision.className}`;
  $("decisionLevel").textContent = decision.level;
  $("decisionIcon").textContent = decision.icon;
  $("decisionTitle").textContent = decision.title;
  $("decisionDetail").textContent = decision.detail;
  const logMet = state.summary && Boolean(state.summary.log_likelihood_criterion_met);
  const combinedMet = state.summary && Boolean(state.summary.family_log_likelihood_criterion_met);
  $("probabilitySignal").textContent = state.summary ? (logMet ? `越线 · ${fixed(state.summary.maximum_log_likelihood_gap)} ≥ ${fixed(logThreshold, 1)}` : `未越线 · ${fixed(state.summary.maximum_log_likelihood_gap)} < ${fixed(logThreshold, 1)}`) : "等待";
  $("familySignal").textContent = state.summary ? (combinedMet ? `成立 · 族支持 ${number(state.summary.maximum_family_support)} / ${familyThreshold} · 双条件通过` : `未成立 · 族支持 ${number(state.summary.maximum_family_support)} / ${familyThreshold} · 不检出`) : "等待";
}

function eventCopy(event) {
  const shard = number(event.shard_index);
  return {
    competition_scan_started: ["检测启动", `完整词表 ${number(event.vocabulary_size).toLocaleString()} · ${number(event.shard_count)} 分片`],
    competition_shard_started: [`分片 ${shard} 开始`, `词表 ${number(event.vocabulary_start).toLocaleString()}–${number(event.vocabulary_end).toLocaleString()}`],
    competition_mining_progress: [`分片 ${shard} 前向完成`, `${number(event.completed).toLocaleString()} / ${number(event.total).toLocaleString()} 候选种子`],
    competition_shard_completed: [`分片 ${shard} 完成`, `${number(event.candidate_count)} 个候选 · ${formatDuration(number(event.elapsed_seconds))}`],
    competition_merge_started: ["候选合并", "四个分片去重并按置信度排序"],
    soft_probe_candidates: ["候选形成", `${(event.candidates || []).length} 个高置信输出进入候选池`],
    competition_probe_started: ["潜变量探测开始", `将检查 Top-${number(event.candidate_count)} 候选`],
    competition_probe_inputs: ["Holdout 输入就绪", `${(event.inputs || []).length} 条问题 · batch ${number(event.batch_size)}`],
    competition_probe_steps: [`候选 #${number(event.rank)} 轨迹更新`, `新增 ${(event.steps || []).length} 个真实优化步`],
    competition_probe_progress: [`候选 #${number(event.rank)} 完成`, `最大概率差 ${fixed(event.max_probability_gap)}`],
    competition_soft_replay: [`候选 #${number(event.rank)} 白盒回放`, `新输入完整复现率 ${percent(event.replay?.soft_trigger_exact_prefix_match_rate)}`],
    competition_probe_result: [`候选 #${number(event.rank)} 证据归档`, `固定判据 ${event.criterion_met ? "满足" : "未满足"}`],
    competition_scan_summary: ["正式检测结论生成", `最大概率差 ${fixed(event.score)} · 阈值 ${fixed(event.threshold)}`],
  }[event.type] || [event.type || "检测事件", `进度 ${number(event.progress)}%`];
}

function renderEvents() {
  const events = [...state.events.values()].sort((a, b) => number(b.sequence) - number(a.sequence)).slice(0, 18);
  $("eventStream").innerHTML = events.length ? events.map((event) => {
    const [title, detail] = eventCopy(event);
    const major = ["competition_shard_completed", "soft_probe_candidates", "competition_soft_replay", "competition_probe_result", "competition_scan_summary"].includes(event.type);
    return `<div class="event-row ${major ? "is-major" : ""}"><time>${escapeHtml(formatTime(event.timestamp))}</time><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></div></div>`;
  }).join("") : "<p>等待结构化事件。</p>";
}

function renderAll({ restartReplay = false } = {}) {
  renderHeader();
  renderShards();
  renderMiningIo();
  const candidateCountBefore = $("candidateSelect").options.length;
  renderCandidateOptions();
  renderProbe();
  renderDecision();
  renderEvents();
  if (state.candidates.length && (restartReplay || candidateCountBefore <= 1) && !state.replayTimer) startReplay();
}

async function loadCalibration() {
  try {
    const response = await fetch("/static/competition-calibration.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.calibration = await response.json();
    const clean = number(state.calibration.clean_calibration?.model_count);
    const backdoor = number(state.calibration.backdoor_development_validation?.model_count);
    $("calibrationSource").textContent = `校准档案 ${state.calibration.profile_id} · ${clean} clean 冻结阈值 · ${backdoor} backdoor 开发验证。`;
    renderDecision();
  } catch (error) {
    $("calibrationSource").textContent = `校准档案载入失败：${error.message}`;
  }
}

async function poll() {
  if (!/^[a-zA-Z0-9_-]+$/.test(state.jobId)) {
    $("connectionState").className = "connection-state is-error";
    $("connectionState").innerHTML = "<i></i>缺少有效任务编号";
    $("stageTitle").textContent = "请在地址后添加 ?job=任务编号";
    return;
  }
  try {
    const response = await fetch(`/api/scans/${encodeURIComponent(state.jobId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const before = state.candidates.length;
    ingestJob(await response.json());
    renderAll({ restartReplay: before === 0 && state.candidates.length > 0 });
    if (["completed", "failed", "cancelled"].includes(state.job.status)) window.clearInterval(state.pollTimer);
  } catch (error) {
    $("connectionState").className = "connection-state is-error";
    $("connectionState").innerHTML = `<i></i>连接失败 ${escapeHtml(error.message)}`;
  }
}

$("candidateSelect").addEventListener("change", (event) => {
  state.activeCandidateRank = number(event.target.value);
  startReplay();
});
$("replayButton").addEventListener("click", startReplay);
$("probeSelect").addEventListener("change", (event) => {
  state.activeProbeRank = number(event.target.value);
  state.activeProbeStep = null;
  renderProbe();
});

initializeShards();
renderAll();
window.setInterval(() => { $("wallClock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }); }, 1000);
loadCalibration().finally(() => {
  poll();
  state.pollTimer = window.setInterval(poll, POLL_INTERVAL_MS);
});

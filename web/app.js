"use strict";

const state = {
  activeId: null,
  activeReport: null,
  catalog: [],
  catalogMethod: "all",
  models: [],
  calibrations: [],
  scenarios: [],
  jobId: null,
  pollTimer: null,
  lastEventSequence: 0,
  modelRoots: [],
  competitionReport: { report: null, candidateRank: 1, probeRank: 1, probeStep: 0 },
  reportView: "overview",
  processPlayer: { stages: [], stageIndex: 0, frameIndex: 0, speed: 1, timer: null, playing: false },
  experience: { controller: null, candidateRank: null, candidateTokenCount: 0 },
  live: {
    discovery: new Map(), validation: new Map(), targetStates: new Map(),
    refinements: new Map(), candidates: [], events: [], activeStage: "output_discovery", currentTarget: null,
    searchProgress: null, searchIterations: [],
  },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) { return `${Math.round(Number(value || 0) * 100)}%`; }
function points(value) {
  const number = Math.round(Number(value || 0) * 100);
  return `${number >= 0 ? "+" : ""}${number} pp`;
}
function probability(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "-";
}
function fixed(value, digits = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}
function softReplayExamplesHtml(replay) {
  const examples = replay?.examples || [];
  if (!examples.length) return '<p class="empty-copy">当前候选尚未保存新输入回放。</p>';
  return examples.map((item) => `<article class="soft-replay-row"><header><b>新问题 #${Number(item.index || 0) + 1}</b><code>${escapeHtml(item.input_text || "-")}</code></header><div><span>不加软向量</span><code>${escapeHtml(item.baseline_output || "[无输出]")}</code><small>匹配 ${Number(item.baseline_prefix_match_tokens || 0)} 个候选 token</small></div><div class="with-soft"><span>加入软向量</span><code>${escapeHtml(item.soft_trigger_output || "[无输出]")}</code><small>${item.soft_trigger_exact_prefix_match ? "完整候选前缀已复现" : `匹配 ${Number(item.soft_trigger_prefix_match_tokens || 0)} 个候选 token`}</small></div></article>`).join("");
}
function selectionModeText(mode) {
  return {
    greedy: "Greedy · 取当前最高概率",
    beam_search: "Beam Search · 保留多条路线",
    beam_assisted_route: "Beam 辅助路线 · 旧记录未保存切换点",
  }[mode] || "未记录选择方式";
}
function referenceSeparation(value) {
  return Number(value?.reference_separation ?? value?.lift ?? 0);
}
function isCompetitionReport(report) { return Boolean(report?.evidence?.competition_core); }
function riskClass(risk) {
  const value = String(risk || "INCONCLUSIVE").toLowerCase();
  return ["high", "medium", "control", "oracle"].includes(value) ? value : "inconclusive";
}
function riskText(risk) {
  return { HIGH: "HIGH 高风险", MEDIUM: "MEDIUM 可疑", CONTROL: "CONTROL 对照", ORACLE: "ORACLE 取证", INCONCLUSIVE: "INCONCLUSIVE 无结论" }[risk] || "INCONCLUSIVE 无结论";
}
function stageText(status) {
  return { complete: "完成", passed: "通过", suspicious: "可疑", control: "对照", inconclusive: "证据不足" }[status] || status || "等待";
}
function scanRoleText(role) {
  return { formal_blind: "正式盲检", blind_detection: "正式盲检", coverage_audit: "隐式后门开发检测", oracle_diagnostic: "Oracle 取证", development_calibration: "开发校准", negative_control: "负对照" }[role] || "独立扫描";
}
function selectedScanMode() {
  return document.querySelector('input[name="scanMode"]:checked')?.value || "coverage_audit";
}
function selectedDetectorMode() {
  return document.querySelector('input[name="detectorMode"]:checked')?.value || "competition_sequence_probe";
}
function isCompetitionMode(mode) { return mode === "competition_sequence_probe"; }
function isSingleModelProbe(mode) { return mode === "reference_free_soft_probe" || isCompetitionMode(mode); }
function isCompetitionCompatibleModel(model) {
  const base = String(model?.base_model || "").replaceAll("\\", "/").toLowerCase();
  return base === "gpt2" || base.endsWith("/gpt2");
}
function isCompetitionFinalModel(model) {
  const path = String(model?.path || "").replaceAll("\\", "/").toLowerCase();
  const inCompetitionRuns = path.startsWith("competition_runs/") || path.includes("/competition_runs/");
  return isCompetitionCompatibleModel(model)
    && inCompetitionRuns
    && path.endsWith("/adapter")
    && !path.includes("competition_runs/smoke_");
}
function competitionModelOptions(models) {
  const preferredOrder = new Map([
    ["gpt2_register", 0],
    ["gpt2_register_seed2", 1],
    ["gpt2_clean", 2],
    ["gpt2_clean_seed2", 3],
    ["gpt2_clean_seed3", 4],
  ]);
  const displayNames = new Map([
    ["gpt2_register", "隐式后门开发样本 A"],
    ["gpt2_register_seed2", "隐式后门开发样本 B"],
    ["gpt2_clean", "干净开发对照 A"],
    ["gpt2_clean_seed2", "干净开发对照 B"],
    ["gpt2_clean_seed3", "干净开发对照 C"],
  ]);
  const runName = (model) => model.path.replaceAll("\\", "/").split("/").at(-2) || model.path;
  return models
    .filter(isCompetitionFinalModel)
    .sort((first, second) => (preferredOrder.get(runName(first)) ?? 99) - (preferredOrder.get(runName(second)) ?? 99) || first.path.localeCompare(second.path))
    .map((model, index) => {
      const name = runName(model);
      return {
        ...model,
        source: "Competition Core · 最终 Adapter",
        label: `模型 ${String(index + 1).padStart(2, "0")} · ${displayNames.get(name) || name} · GPT-2 LoRA`,
      };
    });
}
function catalogMethod(item) {
  return item?.role === "coverage_audit" ? "implicit" : "hotflip";
}
function displayCatalogItems(items) {
  return (items || []).filter((item) => item.available !== false);
}
function methodLabel(method) {
  return method === "implicit" ? "隐式检测" : "Beam HotFlip";
}

function renderCalibrationOptions() {
  const select = $("calibrationInput");
  const previous = select.value;
  select.innerHTML = '<option value="">未加载校准档案</option>' + state.calibrations.map((profile) => {
    const tier = profile.tier === "formal" ? "正式" : "MVP";
    return `<option value="${escapeHtml(profile.path)}">${escapeHtml(`${tier} · ${profile.id} · ${profile.clean_model_count} clean`)}</option>`;
  }).join("");
  if (state.calibrations.some((profile) => profile.path === previous)) select.value = previous;
  else if (state.calibrations.length) select.value = (state.calibrations.find((profile) => profile.formal_ready) || state.calibrations[0]).path;
  renderCalibrationInfo();
}

function renderCalibrationInfo() {
  const profile = state.calibrations.find((item) => item.path === $("calibrationInput").value);
  $("calibrationInfo").textContent = !profile
    ? "未选择时只展示未校准过程，不形成风险裁决。"
    : profile.formal_ready
      ? `正式校准：${profile.clean_model_count} 个独立 clean 模型，允许输出正式结论。`
      : `MVP 校准：${profile.clean_model_count} 个 clean 模型，仅观察分数，固定为 INCONCLUSIVE。`;
}

function selectedCalibration() {
  return state.calibrations.find((item) => item.path === $("calibrationInput").value) || null;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail || detail; } catch (_) { /* response is plain text */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.status === 204 ? null : response.json();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("is-visible");
  window.setTimeout(() => el.classList.remove("is-visible"), 2600);
}

const {
  calibratedCompetitionDecision,
  evidenceSummaryHtml,
  normalizedShards,
  renderCompetitionExperience,
  runCompetitionExperience,
  shardGridHtml,
} = window.BdShieldCompetitionUI.create({ $, state, escapeHtml, fixed, toast });

const {
  candidateInteractions,
  candidateTokenTexts,
  renderCompetitionCandidate,
  renderCompetitionProbeStep,
  renderCompetitionReport,
} = window.BdShieldCompetitionReport.create({
  $, state, escapeHtml, fixed, probability, selectionModeText,
  softReplayExamplesHtml, normalizedShards, shardGridHtml,
  calibratedCompetitionDecision, evidenceSummaryHtml,
  renderCompetitionExperience,
});

const {
  renderCompetitionProbe,
  renderCompetitionVerdict,
  renderCompetitionWorkbench,
} = window.BdShieldCompetitionLive.create({
  $, state, escapeHtml, fixed, probability, selectionModeText,
  softReplayExamplesHtml, candidateInteractions, candidateTokenTexts,
  currentRfProbe, calibratedCompetitionDecision, evidenceSummaryHtml,
  shardGridHtml,
});

function renderCatalog() {
  const available = state.catalog.filter((item) => item.available);
  const visible = available.filter((item) => state.catalogMethod === "all" || catalogMethod(item) === state.catalogMethod);
  $("recordCount").textContent = `${visible.length} / ${available.length} 份报告`;
  document.querySelectorAll("[data-catalog-method]").forEach((button) => {
    const active = button.dataset.catalogMethod === state.catalogMethod;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $("recordList").innerHTML = visible.map((item) => {
    const active = item.id === state.activeId ? " is-active" : "";
    const time = item.modified_at ? new Date(item.modified_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }) : "";
    const badge = item.role === "coverage_audit" ? "已校准" : item.risk || "N/A";
    const method = catalogMethod(item);
    return `<button class="record-item${active}" type="button" data-report-id="${escapeHtml(item.id)}" ${item.available ? "" : "disabled"}>
      <span class="record-title"><b>${escapeHtml(item.title)}</b><i class="mini-risk ${item.role === "coverage_audit" ? "control" : riskClass(item.risk)}">${escapeHtml(badge)}</i></span>
      <span class="record-method ${method}">${escapeHtml(methodLabel(method))}</span>
      <span class="record-meta">${escapeHtml(item.model || "-")}<em>${escapeHtml(time)}</em></span>
    </button>`;
  }).join("") || '<p class="empty-copy">当前筛选下没有检测报告</p>';
  document.querySelectorAll("[data-report-id]").forEach((button) => {
    button.addEventListener("click", () => {
      setSidebarOpen(false);
      void loadReport(button.dataset.reportId);
    });
  });
}

function responseRow(row, index) {
  const input = row.input || row.question || "-";
  const stageBadge = row.perturbation ? `轮 ${row.round} · ${row.perturbation || "基线"}` : `留出问题 ${row.round || index + 1}`;
  const output = (value, hit) => value == null
    ? '<p class="stream-pending">等待该模型输出</p>'
    : `<p class="model-output${hit ? " is-hit" : ""}">${escapeHtml(value || "（空响应）")}</p>`;
  const arriving = row.updatedAt && Date.now() - row.updatedAt < 900 ? " is-arriving" : "";
  return `<article class="response-row${arriving}">
    <div class="response-input"><span>${escapeHtml(stageBadge)}</span><strong>${escapeHtml(row.question || "-")}</strong><code>${escapeHtml(input)}</code></div>
    <div class="model-response target"><span>待审模型</span>${output(row.target_response, row.target_hit)}</div>
    <div class="model-response reference"><span>干净参考</span>${output(row.reference_response, row.reference_hit)}</div>
  </article>`;
}

function renderResponseStream(id, rows, emptyText, limit = 20) {
  const el = $(id);
  const visibleRows = rows.slice(0, limit);
  el.innerHTML = visibleRows.length
    ? visibleRows.map(responseRow).join("")
    : `<p class="empty-copy">${escapeHtml(emptyText)}</p>`;
}

function setRail(name, text, stateName) {
  const rail = document.querySelector(`[data-rail="${name}"]`);
  rail.className = `rail-step ${stateName || ""}`;
  rail.querySelector("small").textContent = text;
}

function targetStatus(status) {
  return {
    pending: "待执行",
    running: "搜索中",
    completed: "已验证",
    screened_out: "快速筛除",
    inconclusive: "无结论",
    not_run_after_success: "阈值后停止",
    not_recorded: "历史未记录",
    candidate_found: "已验证",
  }[status] || "待执行";
}

function renderCandidates(candidates, execution) {
  $("candidateCount").textContent = `${candidates.length} 个候选`;
  if (!candidates.length) {
    $("candidateList").innerHTML = '<p class="empty-copy">未记录候选输出</p>';
    return;
  }
  const highest = Math.max(...candidates.map((candidate) => Number(candidate.score || 0)), 1);
  const executionByTarget = new Map((execution?.candidates || []).map((item) => [item.target_text, item]));
  $("candidateList").innerHTML = candidates.map((candidate) => `
    <div class="candidate-row"><b>${candidate.rank}</b><code>${escapeHtml(candidate.text)}</code>
      <span><i style="width:${Math.max(4, Number(candidate.score || 0) / highest * 100)}%"></i></span>
      <strong title="${candidate.family_support == null ? "候选分数" : `族支持 ${candidate.family_support} · 概率差 ${Number(candidate.probability_gap || 0).toFixed(3)}`}">${Number(candidate.score || 0).toFixed(2)}</strong>${(() => {
        const entry = executionByTarget.get(candidate.text);
        const status = entry?.status || "not_recorded";
        return `<em class="candidate-status ${escapeHtml(status)}" title="${escapeHtml(entry?.reason || "阶段二执行状态")}">${escapeHtml(targetStatus(status))}</em>`;
      })()}
    </div>`).join("");
}

function renderTrace(trace) {
  $("traceCount").textContent = `${trace.length} 步`;
  $("searchTrace").innerHTML = trace.length ? trace.map((item) => `
    <div class="trace-row ${item.accepted ? "accepted" : ""}"><span>#${escapeHtml(item.iteration ?? "-")}</span><code>${escapeHtml(item.trigger || "∅")}</code><b>${Number(item.loss || 0).toFixed(3)}</b><i>${item.accepted ? "保留" : "淘汰"}</i></div>
  `).join("") : '<p class="empty-copy">未记录梯度轨迹</p>';
}

function renderRefinement(refinement) {
  const panel = $("refinementPanel");
  if (!refinement?.enabled) {
    panel.innerHTML = '<p class="empty-copy">本次未启用局部字母精修。</p>';
    return;
  }
  if (refinement.legacy_missing) {
    panel.innerHTML = `<div class="refinement-heading"><span>局部字母精修</span><b>历史数据不完整</b></div>
      <p class="refinement-note">该报告启用了精修，最终选择 <code>${escapeHtml(refinement.selected_trigger || "-")}</code>；种子、候选排名和评分没有保存。</p>`;
    return;
  }
  const metricName = refinement.selection_metric === "reference_separation" ? "参考分离度" : "待审模型 ASR";
  const candidates = refinement.top_candidates || [];
  panel.innerHTML = `<div class="refinement-heading"><span>局部字母精修</span><b>${escapeHtml(metricName)}</b></div>
    <div class="refinement-path"><code>${escapeHtml(refinement.seed_trigger || "-")}</code><i>→</i><code>${escapeHtml(refinement.selected_trigger || "-")}</code></div>
    <p class="refinement-note">在 ${Number(refinement.questions_scored || 0)} 个搜索问题上比较 ${Number(refinement.candidates_scored || 0)} 个${refinement.preserve_length ? "同长度" : "局部"}变体。</p>
    <div class="refinement-rankings">${candidates.map((candidate, index) => `<div><b>#${index + 1}</b><code>${escapeHtml(candidate.trigger)}</code><span>${percent(candidate.target_asr)} / ${percent(candidate.reference_asr)}</span><strong>${points(candidate.primary_score)}</strong></div>`).join("") || '<p class="empty-copy">未保存精修候选排名。</p>'}</div>`;
}

function renderActivationLandscape(refinement) {
  const panel = $("activationLandscape");
  const candidates = refinement?.top_candidates || [];
  if (!refinement?.enabled || !candidates.length) {
    panel.innerHTML = '<div><p class="eyebrow">触发器邻域</p><h3>未保存可视化候选</h3></div><p class="empty-copy">局部精修形成候选后，此处显示触发器邻域的参考分离度。</p>';
    return;
  }
  const maximum = Math.max(...candidates.map((item) => Number(item.primary_score || 0)), 0.01);
  panel.innerHTML = `<div><p class="eyebrow">触发器邻域</p><h3>局部变体的激活强度</h3></div>
    <div class="activation-bars">${candidates.map((candidate, index) => {
      const score = Number(candidate.primary_score || 0);
      const width = Math.max(3, Math.min(100, score / maximum * 100));
      return `<div class="activation-row ${index === 0 ? "is-peak" : ""}"><code>${escapeHtml(candidate.trigger)}</code><span><i style="width:${width}%"></i></span><strong>${points(score)}</strong></div>`;
    }).join("")}</div>`;
}

function renderCoverageReceipt(receipt, scope) {
  const scenario = scope?.scenario || {};
  const legacy = receipt?.legacy_missing;
  $("reportScope").textContent = scanRoleText(scope?.scan_role || scope?.experiment_role);
  $("scenarioBadge").textContent = scenario.label || "历史场景未保存";
  if (legacy) {
    $("coverageClaim").textContent = receipt?.claim || "历史报告未保存覆盖凭证。";
    $("coverageGrid").innerHTML = '<div><span>场景</span><strong>未记录</strong></div><div><span>探针策略</span><strong>未记录</strong></div><div><span>搜索 / 验证</span><strong>未记录</strong></div><div><span>输入位置</span><strong>未记录</strong></div>';
    return;
  }
  const promptSets = receipt?.prompt_sets || {};
  $("coverageClaim").textContent = receipt?.claim || "本报告未提供覆盖声明。";
  $("coverageGrid").innerHTML = `<div><span>场景</span><strong>${escapeHtml(receipt?.scenario_label || scenario.label || "-")}</strong></div>
    <div><span>探针策略</span><strong>${escapeHtml(receipt?.stage1_policy || "-")}</strong></div>
    <div><span>搜索 / 验证</span><strong>${Number(promptSets.search || 0)} / ${Number(promptSets.validation || 0)}</strong></div>
    <div><span>输入位置</span><strong>${escapeHtml((receipt?.input_placement || []).join("、") || "-")}</strong></div>`;
}

function stopProcessPlayer() {
  if (state.processPlayer.timer) window.clearInterval(state.processPlayer.timer);
  state.processPlayer.timer = null;
  state.processPlayer.playing = false;
  $("processPlayBtn").textContent = "▶";
  $("processPlayBtn").title = "播放过程";
  $("processPlayBtn").setAttribute("aria-label", "播放过程");
  $("processPlayBtn").setAttribute("aria-pressed", "false");
}

function buildProcessStages(report) {
  if (isCompetitionReport(report)) {
    const core = report.evidence?.competition_core || {};
    const mining = core.mining || {};
    const miningFrames = [];
    (mining.candidates || []).slice(0, 12).forEach((candidate) => {
      const interactions = candidateInteractions(candidate, mining.response_prefix);
      if (!interactions.length) {
        miningFrames.push({ rank: Number(candidate.rank), rowIndex: 0, detail: `候选 #${candidate.rank} · ${candidate.text || "未保存文本"}` });
      }
      interactions.forEach((item, index) => miningFrames.push({
        rank: Number(candidate.rank),
        rowIndex: index + 1,
        detail: `候选 #${candidate.rank} · 输入“${item.input_text || "-"}” → 输出“${item.output_token_text || "-"}” · 概率 ${probability(item.output_probability)}`,
      }));
    });
    const probeFrames = [];
    (core.probe_evidence || []).forEach((item) => {
      (item.probe?.steps || []).forEach((step, index) => probeFrames.push({
        rank: Number(item.rank),
        stepIndex: index,
        detail: `候选 #${item.rank} · Step ${step.step ?? index + 1} · 候选 ${probability(step.candidate_probability)} / 对照 ${probability(step.control_probability)} · 对数似然差 ${fixed(step.log_likelihood_gap)}`,
      }));
    });
    const decision = calibratedCompetitionDecision(core.summary || {});
    return [
      { key: "mining", selector: "#competitionMiningStage", number: "01", title: "全词表候选发现", frames: miningFrames.length ? miningFrames : [{ detail: "报告没有保存逐 token 候选交互。" }] },
      { key: "probe", selector: "#competitionProbeStage", number: "02", title: "潜变量前缀探测", frames: probeFrames.length ? probeFrames : [{ detail: "报告没有保存逐步潜变量探测轨迹。" }] },
      { key: "decision", selector: "#competitionDecisionStage", number: "03", title: "双条件校准判定", frames: [{ detail: `${decision.code} · ${decision.text}。${decision.detail}` }] },
    ];
  }
  const observations = report.evidence?.stage1_observations || [];
  const trace = report.stages?.trigger_inversion?.trace || [];
  const validation = report.evidence?.validation_examples || [];
  return [
    { key: "discovery", selector: ".report-stage-discovery", number: "01", title: "双模型异常输出发现", frames: (observations.length ? observations : [{}]).map((item, index) => ({ rowIndex: index, detail: item.question ? `探测问题 ${index + 1} · ${item.question}` : "历史报告未保存阶段一逐题响应。" })) },
    { key: "inversion", selector: ".report-stage-inversion", number: "02", title: "多起点 Beam HotFlip 反演", frames: (trace.length ? trace : [{}]).map((item, index) => ({ rowIndex: index, detail: item.trigger ? `迭代 ${item.iteration ?? index + 1} · 触发器“${item.trigger}” · loss ${fixed(item.loss, 3)} · ${item.accepted ? "保留" : "淘汰"}` : "历史报告未保存梯度反演轨迹。" })) },
    { key: "validation", selector: ".report-stage-validation", number: "03", title: "留出问题正向验证", frames: (validation.length ? validation : [{}]).map((item, index) => ({ rowIndex: index, detail: item.question ? `留出问题 ${index + 1} · ${item.question}` : "历史报告仅保存汇总验证指标。" })) },
  ];
}

function highlightPlayerRow(selector, index) {
  document.querySelectorAll(".process-highlight").forEach((element) => element.classList.remove("process-highlight"));
  const row = document.querySelectorAll(selector)[index];
  if (!row) return;
  row.classList.add("process-highlight");
  let scroller = row.parentElement;
  while (scroller && scroller !== document.body) {
    const overflow = globalThis.getComputedStyle(scroller).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && scroller.scrollHeight > scroller.clientHeight) {
      scroller.scrollTop = Math.max(0, row.offsetTop - scroller.offsetTop - 12);
      break;
    }
    scroller = scroller.parentElement;
  }
}

function renderProcessFrame() {
  const player = state.processPlayer;
  const stage = player.stages[player.stageIndex];
  if (!stage) return;
  player.frameIndex = Math.max(0, Math.min(player.frameIndex, stage.frames.length - 1));
  const frame = stage.frames[player.frameIndex] || {};
  document.querySelectorAll(".competition-stage, .report-stage").forEach((element) => element.classList.remove("is-player-active"));
  document.querySelector(stage.selector)?.classList.add("is-player-active");
  $("processStageNumber").textContent = stage.number;
  $("processStageTitle").textContent = stage.title;
  $("processFramePosition").textContent = `${player.frameIndex + 1} / ${stage.frames.length}`;
  $("processFrameDetail").textContent = frame.detail || "该事件没有保存进一步说明。";
  $("processScrubber").max = String(Math.max(0, stage.frames.length - 1));
  $("processScrubber").value = String(player.frameIndex);
  $("processPrevBtn").disabled = player.stageIndex === 0 && player.frameIndex === 0;
  $("processNextBtn").disabled = player.stageIndex === player.stages.length - 1 && player.frameIndex === stage.frames.length - 1;
  document.querySelectorAll("[data-process-stage]").forEach((button) => {
    const active = Number(button.dataset.processStage) === player.stageIndex;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "step" : "false");
  });
  if (isCompetitionReport(state.activeReport) && stage.key === "mining" && frame.rank != null) {
    const core = state.activeReport.evidence?.competition_core || {};
    state.competitionReport.candidateRank = frame.rank;
    renderCompetitionCandidate(core);
    highlightPlayerRow("#competitionTokenTrace .token-interaction", frame.rowIndex || 0);
  } else if (isCompetitionReport(state.activeReport) && stage.key === "probe" && frame.rank != null) {
    const core = state.activeReport.evidence?.competition_core || {};
    state.competitionReport.probeRank = frame.rank;
    state.competitionReport.probeStep = frame.stepIndex || 0;
    renderCompetitionProbeStep(core);
    highlightPlayerRow("#competitionTrajectory [data-competition-step]", frame.stepIndex || 0);
  } else if (stage.key === "discovery") {
    highlightPlayerRow("#stage1ResponseStream .response-row", frame.rowIndex || 0);
  } else if (stage.key === "inversion") {
    highlightPlayerRow("#searchTrace .trace-row", frame.rowIndex || 0);
  } else if (stage.key === "validation") {
    highlightPlayerRow("#validationResponseStream .response-row", frame.rowIndex || 0);
  }
}

function setProcessStage(index) {
  state.processPlayer.stageIndex = Math.max(0, Math.min(Number(index), state.processPlayer.stages.length - 1));
  state.processPlayer.frameIndex = 0;
  renderProcessFrame();
}

function stepProcessPlayer(delta) {
  const player = state.processPlayer;
  const stage = player.stages[player.stageIndex];
  if (!stage) return false;
  const nextFrame = player.frameIndex + delta;
  if (nextFrame >= 0 && nextFrame < stage.frames.length) {
    player.frameIndex = nextFrame;
    renderProcessFrame();
    return true;
  }
  const nextStage = player.stageIndex + (delta > 0 ? 1 : -1);
  if (nextStage < 0 || nextStage >= player.stages.length) return false;
  player.stageIndex = nextStage;
  player.frameIndex = delta > 0 ? 0 : player.stages[nextStage].frames.length - 1;
  renderProcessFrame();
  return true;
}

function startProcessPlayer() {
  stopProcessPlayer();
  state.processPlayer.playing = true;
  $("processPlayBtn").textContent = "Ⅱ";
  $("processPlayBtn").title = "暂停过程";
  $("processPlayBtn").setAttribute("aria-label", "暂停过程");
  $("processPlayBtn").setAttribute("aria-pressed", "true");
  state.processPlayer.timer = window.setInterval(() => {
    if (!stepProcessPlayer(1)) stopProcessPlayer();
  }, 1200 / state.processPlayer.speed);
}

function setReportView(view) {
  const allowed = ["overview", "process", "evidence"];
  state.reportView = allowed.includes(view) ? view : "overview";
  stopProcessPlayer();
  $("reportView").dataset.reportView = state.reportView;
  document.querySelectorAll("[data-report-view]").forEach((button) => {
    const active = button.dataset.reportView === state.reportView;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const competition = isCompetitionReport(state.activeReport);
  const overview = state.reportView === "overview";
  $("verdictBand").hidden = !overview;
  document.querySelector(".review-receipt").hidden = !overview;
  $("processPlayer").hidden = state.reportView !== "process";
  $("competitionReportPanel").hidden = !competition || overview;
  $("genericStageRail").hidden = competition || overview;
  document.querySelectorAll(".report-stage").forEach((stage) => { stage.hidden = competition || overview; });
  document.querySelector(".limitations-section").hidden = state.reportView === "process";
  if (state.reportView === "process") renderProcessFrame();
}

function initializeReportNavigation(report) {
  $("competitionExperienceStage").classList.remove("is-open");
  document.body.classList.remove("experience-open");
  state.activeReport = report;
  state.processPlayer.stages = buildProcessStages(report);
  state.processPlayer.stageIndex = 0;
  state.processPlayer.frameIndex = 0;
  $("processStageJumps").innerHTML = state.processPlayer.stages.map((stage, index) => `<button type="button" data-process-stage="${index}"><b>${stage.number}</b><span>${escapeHtml(stage.title)}</span><small>${stage.frames.length} 个真实事件</small></button>`).join("");
  document.querySelectorAll("[data-process-stage]").forEach((button) => button.addEventListener("click", () => setProcessStage(button.dataset.processStage)));
  $("reportMethod").textContent = isCompetitionReport(report) ? "隐式条件后门检测" : "参考辅助 · Beam HotFlip";
  if (!isCompetitionReport(report)) $("openExperienceBtn").hidden = true;
  setReportView("overview");
}

function renderReport(report) {
  state.activeId = report.id;
  state.activeReport = report;
  renderCatalog();
  $("loadingState").hidden = true;
  $("reportView").hidden = false;

  const date = new Date(report.modified_at).toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" });
  $("reportRole").textContent = scanRoleText(report.scope.scan_role || report.scope.experiment_role);
  $("reportTime").textContent = date;
  $("reportTitle").textContent = report.title;
  $("modelLine").textContent = `${report.model.name} · ${report.model.adapter_path || report.model.base_model}`;
  const competitionReport = isCompetitionReport(report);
  const competitionDecision = competitionReport
    ? calibratedCompetitionDecision(report.evidence?.competition_core?.summary || {})
    : null;
  const risk = competitionDecision ? (competitionDecision.detected ? "high" : "control") : riskClass(report.verdict.risk);
  $("riskBadge").textContent = competitionDecision
    ? `${competitionDecision.code} ${competitionDecision.detected ? "高风险" : "未检出"}`
    : riskText(report.verdict.risk);
  $("riskBadge").className = `risk-badge ${risk}`;
  $("verdictBand").className = `verdict-band ${risk}`;
  $("verdictTitle").textContent = competitionDecision?.text || report.verdict.title;
  $("verdictDetail").textContent = competitionDecision?.detail || report.verdict.detail;
  $("metricLift").textContent = competitionReport ? "不适用" : points(referenceSeparation(report.metrics));

  const stages = report.stages;
  const candidates = stages.output_discovery.candidates || [];
  const trace = stages.trigger_inversion.trace || [];
  const reproduction = stages.forward_reproduction;
  const evidence = report.evidence || {};
  renderCoverageReceipt(evidence.coverage_receipt, report.scope);
  $("competitionReportPanel").hidden = !competitionReport;
  $("genericStageRail").hidden = competitionReport;
  document.querySelectorAll(".report-stage").forEach((stage) => { stage.hidden = competitionReport; });
  if (competitionReport) {
    renderCompetitionReport(report);
    $("limitations").innerHTML = [
      "当前竞赛校准基于 5 个独立 clean 开发模型，双条件观察误报为 0/5。",
      "开发验证中的两个隐式后门模型均被双条件规则检出。",
      "该校准只适用于当前 GPT-2、同版本配置与候选族定义。",
      "本次扫描未读取运行时干净参考模型、训练条件、目标输出或中毒数据。",
    ].map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    initializeReportNavigation(report);
    return;
  }
  setRail("discovery", `${candidates.length} 个 target_text 候选`, stages.output_discovery.status);
  setRail("inversion", report.recovered.trigger ? `触发器 ${report.recovered.trigger}` : "未形成有效触发器", stages.trigger_inversion.status);
  setRail("validation", `${percent(reproduction.asr)} / ${percent(reproduction.reference_asr)}`, reproduction.status);
  renderCandidates(candidates, evidence.target_execution);
  renderTrace(trace);
  renderRefinement(evidence.alpha_refinement);
  renderActivationLandscape(evidence.alpha_refinement);

  $("triggerValue").textContent = competitionReport ? "连续潜变量证据" : report.recovered.trigger || "未找回";
  $("validationInput").textContent = competitionReport
    ? "本模式不执行离散触发器 ASR 复现"
    : report.recovered.trigger
      ? `${report.recovered.trigger} + 留出问题（${reproduction.prompt_count || 0} 条）`
      : "未形成可验证输入";
  $("targetValue").textContent = competitionReport ? "由模型输出自动发现" : report.recovered.target_text || "未确定";
  $("metricAsr").textContent = competitionReport ? "不适用" : percent(reproduction.asr);
  $("metricRefAsr").textContent = competitionReport ? "不适用" : percent(reproduction.reference_asr);
  $("reproSeparation").textContent = competitionReport ? "不适用" : points(referenceSeparation(reproduction));
  $("reproStatus").textContent = competitionReport ? "不执行" : stageText(reproduction.status);
  $("reproStatus").className = `tag ${reproduction.status}`;
  $("limitations").innerHTML = (report.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  renderResponseStream(
    "stage1ResponseStream",
    evidence.stage1_observations || [],
    "该历史报告没有保存阶段一逐题观测。",
  );
  renderResponseStream(
    "validationResponseStream",
    evidence.validation_examples || [],
    "该历史报告没有保存正向验证逐题输出，仅保留汇总指标。",
  );
  initializeReportNavigation(report);
}

async function loadReport(id) {
  state.activeId = id;
  renderCatalog();
  $("reportView").hidden = true;
  $("loadingState").hidden = false;
  try { renderReport(await api(`/api/catalog/${encodeURIComponent(id)}`)); }
  catch (error) { $("loadingState").innerHTML = `<p>报告载入失败：${escapeHtml(error.message)}</p>`; }
}

function renderModelOptions() {
  const selectedTarget = $("targetInput").value || "competition_runs/gpt2_register/adapter";
  const effectiveDetectorMode = selectedScanMode() === "oracle_diagnostic"
    ? "reference_assisted"
    : selectedDetectorMode();
  const targetModels = isCompetitionMode(effectiveDetectorMode)
    ? competitionModelOptions(state.models)
    : state.models;
  const competitionDefault = targetModels.find((model) => model.path === "competition_runs/gpt2_register/adapter")
    || targetModels.find((model) => model.kind === "LoRA adapter" && /\/adapter$/i.test(model.path))
    || targetModels[0];
  const targetFallback = isCompetitionMode(effectiveDetectorMode)
    ? competitionDefault?.path || ""
    : "runs/opt125m_autopois_strong_v2/lora";
  const renderSelect = (id, fallback, models) => {
    const select = $(id);
    const current = select.value || fallback;
    const previous = id === "targetInput"
      && !isCompetitionMode(effectiveDetectorMode)
      && String(current).replaceAll("\\", "/").includes("competition_runs/")
      ? fallback
      : current;
    const groups = new Map();
    models.forEach((model) => {
      const group = groups.get(model.source) || [];
      group.push(model);
      groups.set(model.source, group);
    });
    select.innerHTML = [...groups].map(([source, models]) => `<optgroup label="${escapeHtml(source)}">${models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.label)}</option>`).join("")}</optgroup>`).join("");
    const available = models.some((model) => model.path === previous);
    if (available) select.value = previous;
    else if (models.length) select.value = models.some((model) => model.path === fallback) ? fallback : models[0].path;
  };
  renderSelect("targetInput", targetFallback, targetModels);
  const target = state.models.find((model) => model.path === ($("targetInput").value || selectedTarget));
  const compatibleReferences = (target?.base_model
    ? state.models.filter((model) => model.kind === "LoRA adapter" && model.base_model === target.base_model)
    : state.models.filter((model) => model.kind === "LoRA adapter"))
    .filter((model) => model.path !== target?.path);
  const referenceSelect = $("referenceInput");
  const previousReference = referenceSelect.value || "runs/opt125m_clean_ref/lora";
  const groups = new Map();
  compatibleReferences.forEach((model) => {
    const group = groups.get(model.source) || [];
    group.push(model);
    groups.set(model.source, group);
  });
  referenceSelect.innerHTML = [...groups].map(([source, models]) => `<optgroup label="${escapeHtml(source)}">${models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.label)}</option>`).join("")}</optgroup>`).join("");
  const cleanReference = compatibleReferences.find((model) => /(?:clean|reference|ref)/i.test(model.path));
  if (compatibleReferences.some((model) => model.path === previousReference)) referenceSelect.value = previousReference;
  else if (cleanReference) referenceSelect.value = cleanReference.path;
  else if (compatibleReferences.length) referenceSelect.value = compatibleReferences[0].path;
  referenceSelect.disabled = compatibleReferences.length === 0;
  const sources = [...new Set(state.modelRoots.map((root) => root.source))];
  $("modelScanScope").textContent = state.models.length
    ? isCompetitionMode(effectiveDetectorMode)
      ? `已收敛到 ${targetModels.length} 个 Competition Core 最终 Adapter；已隐藏 epoch checkpoint、smoke 和旧实验目录。`
      : `已发现 ${state.models.length} 个可选模型，扫描来源：${sources.join("、") || "工作区"}。`
    : "未发现可选模型；已扫描工作区和本机 Hugging Face 缓存。";
  syncScanSetup();
}

function renderModelSelectionInfo() {
  const describe = (path) => {
    const model = state.models.find((item) => item.path === path);
    if (!model) return path ? "手动输入的路径将在启动前校验。" : "未选择";
    const base = model.base_model ? `，基座 ${model.base_model}` : "";
    return `${model.path} · ${model.kind}${base}`;
  };
  const detectorMode = selectedScanMode() === "oracle_diagnostic" ? "reference_assisted" : selectedDetectorMode();
  if (isSingleModelProbe(detectorMode)) {
    const method = isCompetitionMode(detectorMode) ? "隐式条件后门检测" : "无参考软探测";
    $("modelSelectionInfo").textContent = `待审：${describe($("targetInput").value.trim())}  |  ${method}只读取待审模型，不加载干净参考模型。`;
    return;
  }
  const target = state.models.find((item) => item.path === $("targetInput").value.trim());
  const reference = state.models.find((item) => item.path === $("referenceInput").value.trim());
  const compatibility = target?.base_model && reference?.base_model && target.base_model === reference.base_model
    ? ` · 同基座 ${target.base_model}`
    : " · 请选择同基座的干净参考 LoRA";
  $("modelSelectionInfo").textContent = `待审：${describe($("targetInput").value.trim())}  |  参考：${describe($("referenceInput").value.trim())}${compatibility}`;
}

function selectedScenario() {
  return document.querySelector('input[name="scenario"]:checked')?.value || "general";
}

function renderScenarioOptions() {
  const selected = selectedScenario();
  $("scenarioGrid").innerHTML = state.scenarios.map((scenario) => `<label class="scenario-choice ${scenario.id === selected ? "is-selected" : ""}">
    <input type="radio" name="scenario" value="${escapeHtml(scenario.id)}" ${scenario.id === selected ? "checked" : ""}>
    <span>${escapeHtml(scenario.short_label || scenario.label)}</span>
    <small>${escapeHtml((scenario.coverage_focus || []).join(" · "))}</small>
  </label>`).join("");
  document.querySelectorAll('input[name="scenario"]').forEach((input) => input.addEventListener("change", syncScanSetup));
  syncScanSetup();
}

function syncScanSetup() {
  const detectorMode = selectedDetectorMode();
  const competitionMode = isCompetitionMode(detectorMode);
  const requestedMode = competitionMode ? "coverage_audit" : "formal_blind";
  const modeInput = document.querySelector(`input[name="scanMode"][value="${requestedMode}"]`);
  const generalScenario = document.querySelector('input[name="scenario"][value="general"]');
  if (modeInput) modeInput.checked = true;
  if (competitionMode && generalScenario) generalScenario.checked = true;
  const scenario = state.scenarios.find((item) => item.id === selectedScenario());
  $("calibrationField").hidden = true;
  $("presetInput").disabled = competitionMode;
  $("dtypeInput").disabled = competitionMode;
  if (competitionMode) {
    $("presetInput").value = "exhaustive";
    $("dtypeInput").value = "float16";
  }
  document.querySelectorAll(".mode-choice").forEach((choice) => {
    choice.classList.toggle("is-selected", choice.querySelector("input")?.checked);
  });
  document.querySelectorAll(".scenario-choice").forEach((choice) => {
    choice.classList.toggle("is-selected", choice.querySelector("input")?.checked);
  });
  $("oracleTargetField").hidden = true;
  $("oracleTargetInput").required = false;
  $("referenceField").hidden = detectorMode !== "reference_assisted";
  $("referenceInput").required = detectorMode === "reference_assisted";
  $("referenceInput").disabled = detectorMode !== "reference_assisted" || !$("referenceInput").options.length;
  $("scenarioPicker").hidden = competitionMode;
  $("fixedRuntimeConfig").hidden = !competitionMode;
  $("runtimeFormGrid").hidden = competitionMode;
  $("modelPickerLabel").textContent = competitionMode ? "Competition Core 最终模型" : "待审模型与同基座参考";
  $("scanDialogTitle").textContent = competitionMode ? "启动隐式条件后门检测" : "启动词级触发器反演";
  $("methodEyebrow").textContent = competitionMode ? "推荐方法" : "增强取证";
  $("methodTitle").textContent = competitionMode ? "隐式条件后门检测" : "参考辅助 · 多起点 Beam HotFlip";
  $("methodDescription").textContent = competitionMode
    ? "单独检查一个待审模型：先挖掘异常强化输出，再用连续潜变量前缀与内部普通对照比较；不使用运行时干净参考模型、已知条件、训练目标或中毒数据。"
    : "先比较待审模型与同基座干净参考模型的异常输出，再从多个随机起点执行 Beam HotFlip，恢复可读词级触发器并在独立留出问题上复现。";
  $("methodFactOneLabel").textContent = competitionMode ? "模型输入" : "模型输入";
  $("methodFactOne").textContent = competitionMode ? "单个待审开放权重模型" : "待审模型 + 同基座干净参考模型";
  $("methodFactTwoLabel").textContent = competitionMode ? "结论依据" : "取证证据";
  $("methodFactTwo").textContent = competitionMode ? "对数似然差与候选族支持联合校准" : "恢复触发器、留出 ASR 与参考分离度";
  $("scenarioSummary").textContent = scenario
    ? `${scenario.label} · 探测 ${scenario.discovery_prompt_count} / 验证 ${scenario.validation_prompt_count}`
    : "等待场景";
  $("startScanBtn").textContent = competitionMode ? "开始隐式后门检测" : "开始 Beam HotFlip 反演";
  renderModelSelectionInfo();
}

async function refreshModels() {
  const [data, calibrations] = await Promise.all([api("/api/models"), api("/api/calibrations")]);
  state.models = data.items || [];
  state.modelRoots = data.search_roots || [];
  state.calibrations = calibrations.items || [];
  renderModelOptions();
  renderCalibrationOptions();
  toast(`发现 ${state.models.length} 个本地模型`);
}

async function addModelRoot() {
  const input = $("modelRootInput");
  const path = input.value.trim();
  if (!path) return;
  const data = await api("/api/model-roots", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  state.models = data.catalog.items || [];
  state.modelRoots = data.catalog.search_roots || [];
  input.value = "";
  renderModelOptions();
  toast(`已扫描 ${data.path}`);
}

function resetLiveState() {
  state.lastEventSequence = 0;
  state.live = {
    discovery: new Map(), validation: new Map(), targetStates: new Map(),
    refinements: new Map(), candidates: [], events: [], activeStage: "output_discovery", currentTarget: null,
    detectorMode: "reference_free_soft_probe", parameters: [], parameterSignature: "",
    rf: {
      responsePrefix: "", candidates: [], probes: new Map(), summary: null, miningProgress: null,
      shards: new Map(), probeInputs: [], batchSize: 0, softTokenCount: 0,
      activeCandidateRank: null, activeProbeRank: null, activeProbeStep: null,
    },
  };
}

function captureLiveEvents(events) {
  const changes = new Set();
  for (const event of events || []) {
    if (Number(event.sequence || 0) <= state.lastEventSequence) continue;
    state.lastEventSequence = Number(event.sequence || 0);
    if (event.type === "scan_configuration") {
      state.live.detectorMode = event.detector_mode || state.live.detectorMode;
      state.live.parameters = event.parameters || [];
      changes.add("configuration");
    }
    if (event.type === "model_response") {
      const key = `${event.round}:${event.question}`;
      const row = state.live.discovery.get(key) || { round: event.round, perturbation: event.perturbation, question: event.question, input: event.input, target_response: null, reference_response: null };
      row[event.model === "target" ? "target_response" : "reference_response"] = event.output;
      row.updatedAt = Date.now();
      state.live.discovery.set(key, row);
      changes.add("discovery");
    }
    if (event.type === "validation_response") {
      const key = `${event.target_text}:${event.round}:${event.question}`;
      const row = state.live.validation.get(key) || {
        target_text: event.target_text, round: event.round, question: event.question,
        input: event.input, target_response: null, reference_response: null,
      };
      if (event.model === "target" || event.model === "reference") {
        row[`${event.model}_response`] = event.output;
        row[`${event.model}_hit`] = event[`${event.model}_hit`];
      } else {
        Object.assign(row, event);
      }
      row.updatedAt = Date.now();
      state.live.validation.set(key, row);
      changes.add("validation");
    }
    if (event.type === "stage1_candidates") {
      state.live.candidates = event.candidates || [];
      changes.add("discovery");
      changes.add("inversion");
    }
    if (event.type === "soft_probe_candidates") {
      state.live.rf.responsePrefix = event.response_prefix || "";
      state.live.rf.candidates = event.candidates || [];
      state.live.rf.activeCandidateRank ||= Number(event.candidates?.[0]?.rank || 0) || null;
      changes.add("rf-discovery");
    }
    if (event.type === "competition_scan_started") {
      state.live.rf.responsePrefix = event.response_prefix || "";
      state.live.rf.miningProgress = event;
      const shardCount = Number(event.shard_count || 0);
      const vocabularySize = Number(event.vocabulary_size || 0);
      for (let index = 0; index < shardCount; index += 1) {
        state.live.rf.shards.set(index + 1, {
          shard_index: index + 1,
          vocabulary_start: Math.floor(vocabularySize * index / shardCount),
          vocabulary_end: Math.floor(vocabularySize * (index + 1) / shardCount),
          status: "pending",
        });
      }
      changes.add("rf-discovery");
    }
    if (["competition_shard_started", "competition_mining_progress", "competition_shard_completed", "competition_merge_started"].includes(event.type)) {
      state.live.rf.miningProgress = event;
      if (event.shard_index) {
        const shard = state.live.rf.shards.get(Number(event.shard_index)) || { shard_index: Number(event.shard_index) };
        Object.assign(shard, event, {
          status: event.type === "competition_shard_completed" ? "complete" : "running",
        });
        state.live.rf.shards.set(Number(event.shard_index), shard);
      }
      changes.add("rf-discovery");
    }
    if (event.type === "competition_probe_inputs") {
      state.live.rf.probeInputs = event.inputs || [];
      state.live.rf.batchSize = Number(event.batch_size || 0);
      state.live.rf.softTokenCount = Number(event.soft_token_count || 0);
      changes.add("rf-probe");
    }
    if (event.type === "competition_probe_steps") {
      const candidate = state.live.rf.candidates.find((item) => Number(item.rank) === Number(event.rank)) || {};
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank, candidate_output: candidate.text, steps: [] };
      const byStep = new Map((probe.steps || []).map((item) => [Number(item.step), item]));
      (event.steps || []).forEach((item) => byStep.set(Number(item.step), item));
      probe.steps = [...byStep.values()].sort((a, b) => Number(a.step) - Number(b.step));
      Object.assign(probe, { rank: event.rank });
      state.live.rf.probes.set(event.rank, probe);
      state.live.rf.activeProbeRank = Number(event.rank);
      state.live.rf.activeProbeStep = Math.max(0, probe.steps.length - 1);
      changes.add("rf-probe");
    }
    if (event.type === "competition_probe_progress") {
      const candidate = state.live.rf.candidates.find((item) => Number(item.rank) === Number(event.rank)) || {};
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank, candidate_output: candidate.text, steps: [] };
      Object.assign(probe, event);
      state.live.rf.probes.set(event.rank, probe);
      state.live.rf.activeProbeRank = Number(event.rank);
      changes.add("rf-probe");
    }
    if (event.type === "competition_soft_replay") {
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank, steps: [] };
      probe.replay = event.replay || {};
      state.live.rf.probes.set(event.rank, probe);
      state.live.rf.activeProbeRank = Number(event.rank);
      changes.add("rf-probe");
    }
    if (event.type === "competition_probe_result") {
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank };
      Object.assign(probe, event, { evidence: true, steps: event.steps || probe.steps || [] });
      state.live.rf.probes.set(event.rank, probe);
      state.live.rf.activeProbeRank ||= Number(event.rank);
      changes.add("rf-probe");
    }
    if (event.type === "competition_scan_summary") {
      state.live.rf.summary = event;
      changes.add("rf-verdict");
    }
    if (event.type === "soft_probe_started") {
      const probe = state.live.rf.probes.get(event.rank) || { steps: new Map() };
      Object.assign(probe, event);
      state.live.rf.probes.set(event.rank, probe);
      changes.add("rf-probe");
    }
    if (event.type === "soft_probe_step") {
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank, candidate_output: event.candidate_output, steps: new Map() };
      const key = `${event.role}:${event.baseline_index || 0}:${event.seed}`;
      probe.steps.set(key, event);
      state.live.rf.probes.set(event.rank, probe);
      changes.add("rf-probe");
    }
    if (event.type === "soft_trigger_probe") {
      const probe = state.live.rf.probes.get(event.rank) || { rank: event.rank, steps: new Map() };
      Object.assign(probe, event);
      state.live.rf.probes.set(event.rank, probe);
      changes.add("rf-probe");
    }
    if (event.type === "soft_probe_summary") {
      state.live.rf.summary = event;
      changes.add("rf-verdict");
    }
    if (event.type === "target_started") {
      state.live.currentTarget = event.target_text;
      state.live.searchProgress = null;
      state.live.targetStates.set(event.target_text, { status: "running", ...event });
      changes.add("inversion");
    }
    if (event.type === "target_completed") {
      state.live.targetStates.set(event.target_text, { ...event });
      changes.add("inversion");
    }
    if (event.type === "target_skipped") {
      state.live.targetStates.set(event.target_text, { ...event, status: "not_run_after_success" });
      changes.add("inversion");
    }
    if (event.type === "search_progress") {
      state.live.searchProgress = event;
      changes.add("inversion");
    }
    if (event.type === "search_iteration") {
      state.live.searchIterations.push(event);
      state.live.searchIterations = state.live.searchIterations.slice(-160);
      changes.add("inversion");
    }
    if (event.type === "alpha_refinement") {
      const key = event.target_text || state.live.currentTarget || "unknown";
      const refinement = state.live.refinements.get(key) || { candidates: [] };
      if (event.phase === "candidate_scored") {
        const existing = refinement.candidates.findIndex((item) => item.trigger === event.trigger);
        if (existing >= 0) refinement.candidates[existing] = event;
        else refinement.candidates.push(event);
        refinement.phase = event.phase;
        refinement.candidate_index = event.candidate_index;
        refinement.candidates_scored = event.candidates_scored;
      } else {
        Object.assign(refinement, event);
        if (event.top_candidates?.length) refinement.candidates = event.top_candidates;
      }
      state.live.refinements.set(key, refinement);
      changes.add("inversion");
    }
    if (event.type === "scan_summary") {
      changes.add("validation");
    }
    state.live.events.push(event);
  }
  state.live.events = state.live.events.slice(-240);
  return changes;
}

function latestEvent(type) {
  return [...state.live.events].reverse().find((event) => event.type === type);
}

function liveStage(job) {
  const stage = job.stage;
  if (isSingleModelProbe(job.detector_mode)) {
    if (["output_discovery", "soft_trigger_probe", "calibrated_verdict"].includes(stage)) return stage;
    if (job.status === "completed") return "calibrated_verdict";
    return "output_discovery";
  }
  if (["output_discovery", "trigger_inversion", "forward_reproduction"].includes(stage)) return stage;
  if (job.status === "completed") return state.live.activeStage || "forward_reproduction";
  return "output_discovery";
}

function setLiveStage(stage, detectorMode) {
  state.live.activeStage = stage;
  const referenceFree = isSingleModelProbe(detectorMode);
  const competition = isCompetitionMode(detectorMode);
  const stages = referenceFree
    ? ["output_discovery", "soft_trigger_probe", "calibrated_verdict"]
    : ["output_discovery", "trigger_inversion", "forward_reproduction"];
  const railKeys = ["output_discovery", "trigger_inversion", "forward_reproduction"];
  const currentIndex = stages.indexOf(stage);
  const labels = competition
    ? { output_discovery: "全词表扫描中", soft_trigger_probe: "逐候选探测中", calibrated_verdict: "双条件判定中" }
    : referenceFree
    ? { output_discovery: "候选生成中", soft_trigger_probe: "连续优化中", calibrated_verdict: "校准裁决中" }
    : { output_discovery: "双模型探测中", trigger_inversion: "逆向搜索中", forward_reproduction: "逐题验证中" };
  $("liveRailOneLabel").textContent = competition ? "全词表扫描" : referenceFree ? "输出候选生成" : "异常输出发现";
  $("liveRailTwoLabel").textContent = competition ? "潜变量前缀探测" : referenceFree ? "软触发对照" : "触发器逆向";
  $("liveRailThreeLabel").textContent = competition ? "校准判定" : referenceFree ? "校准与裁决" : "留出正向验证";
  document.querySelectorAll("[data-live-rail]").forEach((rail) => {
    const index = railKeys.indexOf(rail.dataset.liveRail);
    rail.classList.toggle("is-current", index === currentIndex);
    rail.classList.toggle("is-complete", index < currentIndex);
    const summary = rail.querySelector("small");
    if (summary) summary.textContent = index < currentIndex ? "已完成" : index === currentIndex ? labels[stage] : "等待";
  });
  $("liveDiscoveryPanel").hidden = referenceFree || stage !== "output_discovery";
  $("liveInversionPanel").hidden = referenceFree || stage !== "trigger_inversion";
  $("liveValidationPanel").hidden = referenceFree || stage !== "forward_reproduction";
  $("liveCompetitionPanel").hidden = !competition;
  $("liveRfDiscoveryPanel").hidden = !referenceFree || competition || stage !== "output_discovery";
  $("liveRfProbePanel").hidden = !referenceFree || competition || stage !== "soft_trigger_probe";
  $("liveRfVerdictPanel").hidden = !referenceFree || competition || stage !== "calibrated_verdict";
}

function renderLiveDiscovery() {
  const rows = [...state.live.discovery.values()].sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
  renderResponseStream("liveDiscoveryStream", rows, "等待双模型输出", 24);
  const candidates = state.live.candidates;
  $("liveCandidateCount").textContent = `${candidates.length} 个候选`;
  $("liveCandidateList").innerHTML = candidates.length
    ? candidates.map((candidate) => `<div><b>#${escapeHtml(candidate.rank)}</b><code>${escapeHtml(candidate.text)}</code><span>${Number(candidate.score || 0).toFixed(2)}</span></div>`).join("")
    : "模型响应完成后会在这里排序 target_text。";
}

function renderLiveInversion() {
  const candidates = state.live.candidates;
  const currentTarget = state.live.currentTarget || latestEvent("target_started")?.target_text;
  const current = state.live.targetStates.get(currentTarget) || latestEvent("target_started") || {};
  $("liveTargetRun").textContent = currentTarget
    ? `target_text = ${currentTarget} · ${current.run_index || 1}/${current.run_total || candidates.length || 1}`
    : "等待 target_text";
  $("liveTargetCount").textContent = candidates.length || state.live.targetStates.size;
  $("liveTargetList").innerHTML = candidates.length
    ? candidates.map((candidate) => {
      const entry = state.live.targetStates.get(candidate.text);
      return `<div class="live-target-row ${escapeHtml(entry?.status || "pending")}"><b>#${escapeHtml(candidate.rank)}</b><code>${escapeHtml(candidate.text)}</code><span>${escapeHtml(targetStatus(entry?.status || "pending"))}</span></div>`;
    }).join("")
    : "等待阶段一候选";
  const progress = state.live.searchProgress?.target_text === currentTarget ? state.live.searchProgress : null;
  const progressPanel = $("liveSearchProgress");
  if (progress) {
    const phaseLabels = { initialization: "评估随机起点", beam_evaluation: "评估梯度替换候选", length_growth: "评估加长触发器" };
    const modelLabel = progress.model === "reference" ? "干净参考模型前向" : "待审模型前向";
    const completed = Number(progress.completed || 0);
    const total = Number(progress.total || 0);
    const ratio = total ? Math.min(100, completed / total * 100) : 0;
    progressPanel.hidden = false;
    progressPanel.innerHTML = `<div><span>${escapeHtml(phaseLabels[progress.phase] || "评估触发器候选")} · 第 ${escapeHtml(progress.iteration ?? 0)} 轮</span><strong>${escapeHtml(modelLabel)} ${completed}/${total || "?"}</strong></div><p>${Number(progress.candidate_count || 0)} 个触发器候选 × ${Number(progress.question_count || 0)} 个搜索问题；只统计已完成的真实生成批次。</p><i><b style="width:${ratio}%"></b></i>`;
  } else {
    progressPanel.hidden = true;
    progressPanel.innerHTML = "";
  }
  const iterations = state.live.searchIterations.filter((event) => !currentTarget || event.target_text === currentTarget);
  const latest = iterations.at(-1);
  $("liveIteration").textContent = latest ? `#${latest.iteration}` : "0";
  $("liveTrace").innerHTML = iterations.length
    ? iterations.map((event) => `<div class="live-trace-row ${event.accepted ? "accepted" : ""}"><span>#${escapeHtml(event.iteration)}</span><span>${escapeHtml(event.position ?? "-")}</span><code>${escapeHtml(event.trigger || "∅")}</code><b>${Number(event.loss || 0).toFixed(3)}</b><i>${event.accepted ? "保留" : "淘汰"}</i></div>`).join("")
    : "等待梯度候选";

  const refinement = state.live.refinements.get(currentTarget);
  const panel = $("liveRefinement");
  if (!refinement) {
    panel.innerHTML = '<div class="live-panel-heading"><span>局部字母精修</span><b>等待 HotFlip 候选</b></div><div class="live-refinement-content">精修仅在短字母触发器形成后开始。</div>';
    return;
  }
  const candidatesScored = refinement.candidates || [];
  const selected = refinement.selected_trigger || "计算中";
  const generating = refinement.phase === "generation_progress";
  const refinementState = refinement.phase === "completed"
    ? "已完成"
    : generating
      ? `${refinement.model === "reference" ? "参考模型" : "待审模型"}前向 ${refinement.completed || 0}/${refinement.total || "?"}`
      : `已评分 ${candidatesScored.length}/${refinement.candidates_scored || "?"}`;
  const generationRatio = refinement.total ? Math.min(100, Number(refinement.completed || 0) / Number(refinement.total) * 100) : 0;
  panel.innerHTML = `<div class="live-panel-heading"><span>局部字母精修</span><b>${escapeHtml(refinementState)}</b></div>
    ${generating ? `<div class="live-refinement-progress"><span>${Number(refinement.candidate_count || 0)} 个局部变体 × ${Number(refinement.question_count || 0)} 个搜索问题</span><i><b style="width:${generationRatio}%"></b></i></div>` : ""}
    <div class="live-refinement-path"><code>${escapeHtml(refinement.seed_trigger || "-")}</code><i>→</i><code>${escapeHtml(selected)}</code><span>${escapeHtml(refinement.selection_metric === "reference_separation" ? "按参考分离度选择" : "按待审模型 ASR 选择")}</span></div>
    <div class="live-refinement-rankings">${candidatesScored.map((candidate, index) => `<div><b>#${escapeHtml(candidate.candidate_index || index + 1)}</b><code>${escapeHtml(candidate.trigger)}</code><span>${percent(candidate.target_asr)} / ${candidate.reference_asr == null ? "-" : percent(candidate.reference_asr)}</span><strong>${points(candidate.primary_score)}</strong></div>`).join("") || '<p class="empty-copy">正在生成局部变体。</p>'}</div>`;
}

function renderLiveValidation() {
  const validationTarget = latestEvent("validation_response")?.target_text || state.live.currentTarget;
  const rows = [...state.live.validation.values()]
    .filter((row) => !validationTarget || row.target_text === validationTarget)
    .sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
  renderResponseStream("liveValidationStream", rows, "等待第一条留出验证输出", 20);
  const targetDone = rows.filter((row) => row.target_response != null).length;
  const referenceDone = rows.filter((row) => row.reference_response != null).length;
  const targetHits = rows.filter((row) => row.target_hit).length;
  const referenceHits = rows.filter((row) => row.reference_hit).length;
  const summary = latestEvent("scan_summary");
  const trigger = summary?.best_trigger || state.live.refinements.get(validationTarget)?.selected_trigger || "候选触发器";
  $("liveVerdict").textContent = `${targetDone}/${rows.length || "?"} 待审 · ${referenceDone}/${rows.length || "?"} 参考`;
  $("liveValidationMetrics").innerHTML = `<div><span>验证 target_text</span><code>${escapeHtml(validationTarget || "-")}</code></div><div><span>触发输入</span><code>${escapeHtml(trigger)} + 留出问题</code></div><div><span>命中计数</span><strong>待审 ${targetHits} / 参考 ${referenceHits}</strong></div>`;
}

function renderLiveParameters(parameters) {
  const signature = JSON.stringify(parameters || []);
  if (signature === state.live.parameterSignature) return;
  state.live.parameterSignature = signature;
  $("liveParameterCount").textContent = parameters?.length ? `${parameters.length} 项已冻结` : "等待任务配置";
  $("liveParameterGrid").innerHTML = parameters?.length
    ? parameters.map((parameter) => `<div><span>${escapeHtml(parameter.label || parameter.key)}</span><code title="${escapeHtml(parameter.value)}">${escapeHtml(parameter.value)}</code></div>`).join("")
    : '<div class="parameter-empty">任务创建后在此显示实际传入检测器的参数。</div>';
}

function currentRfProbe() {
  const probes = [...state.live.rf.probes.values()].sort((a, b) => Number(a.rank || 0) - Number(b.rank || 0));
  return probes.at(-1) || null;
}

function renderReferenceFreeDiscovery() {
  const candidates = state.live.rf.candidates;
  const mining = state.live.rf.miningProgress;
  $("rfResponsePrefix").textContent = state.live.rf.responsePrefix || "等待响应分隔符";
  $("rfCandidateCount").textContent = candidates.length
    ? `${candidates.length} 个候选`
    : mining?.shard_index
      ? `分片 ${mining.shard_index}/${mining.shard_count} · ${mining.completed || 0}/${mining.total || "?"}`
      : "候选生成中";
  $("rfCandidateStream").innerHTML = candidates.length
    ? candidates.map((candidate, index) => `<div class="rf-candidate-row"><b>#${escapeHtml(candidate.rank || index + 1)}</b><code>${escapeHtml(candidate.text)}</code><span>后缀 ${(Number(candidate.suffix_probability || 0) * 100).toFixed(1)}% · 族支持 ${Number(candidate.family_support || 0)}</span><strong>${candidate.token_ids?.length || 0} tokens</strong></div>`).join("")
    : '<p class="empty-copy">等待待审模型生成满足后缀置信门限的输出候选。</p>';
}

function renderReferenceFreeProbe() {
  const probe = currentRfProbe();
  if (!probe) {
    $("rfProbeProgress").textContent = "等待候选";
    $("rfProbeInputs").innerHTML = "";
    $("rfCandidateTarget").innerHTML = '<p class="empty-copy">候选输出将在这里进入连续软提示反演。</p>';
    $("rfBaselineTargets").innerHTML = '<p class="empty-copy">等待候选长度确定后构造内部良性对照。</p>';
    $("rfTrajectory").innerHTML = "";
    $("rfStepCount").textContent = "等待首步损失";
    return;
  }
  const searchPrompts = (probe.optimization_prompts || []).join("\n");
  const holdoutPrompts = (probe.validation_prompts || []).join("\n");
  $("rfProbeInputs").innerHTML = `<div><span>优化输入问题</span><code>${escapeHtml(searchPrompts || "等待事件")}</code></div><div><span>留出评分问题</span><code>${escapeHtml(holdoutPrompts || "等待事件")}</code></div>`;
  $("rfCandidateTarget").innerHTML = `<code>${escapeHtml(probe.candidate_output || "-")}</code>`;
  const baselines = probe.baselines || probe.evidence?.baselines || [];
  $("rfBaselineTargets").classList.toggle("baseline", true);
  $("rfBaselineTargets").innerHTML = baselines.length
    ? baselines.map((baseline, index) => `<code title="良性对照 ${index + 1}">${escapeHtml(baseline.text)}</code>`).join("")
    : '<p class="empty-copy">正在构造无 token 重叠的等长良性输出。</p>';
  const steps = [...(probe.steps || new Map()).values()].sort((a, b) => String(a.role).localeCompare(String(b.role)) || Number(a.baseline_index || 0) - Number(b.baseline_index || 0) || Number(a.seed) - Number(b.seed));
  const maxNll = Math.max(...steps.map((item) => Number(item.nll || 0)), 1);
  const minNll = Math.min(...steps.map((item) => Number(item.nll || 0)), maxNll);
  $("rfProbeProgress").textContent = probe.evidence ? `候选 #${probe.rank} 已完成 · 分数 ${Number(probe.score || 0).toFixed(4)}` : `候选 #${probe.rank} 正在优化`;
  $("rfStepCount").textContent = `${steps.length} 条分段优化事件`;
  $("rfTrajectory").innerHTML = steps.length
    ? steps.map((item) => {
      const span = Math.max(6, ((maxNll - Number(item.nll || 0)) / Math.max(0.000001, maxNll - minNll)) * 100);
      const label = item.role === "candidate" ? `候选 · s${item.seed}` : `对照${item.baseline_index || ""} · s${item.seed}`;
      return `<div class="rf-trajectory-row"><code>${escapeHtml(label)}</code><span>${escapeHtml(item.step)}/${escapeHtml(item.total_steps)}</span><span><i style="width:${span}%"></i></span><strong>${Number(item.nll || 0).toFixed(3)}</strong></div>`;
    }).join("")
    : '<p class="empty-copy">等待第一段软提示优化损失。</p>';
}

function renderReferenceFreeVerdict() {
  const summary = state.live.rf.summary;
  const probe = currentRfProbe();
  if (!summary) {
    $("rfVerdictCode").textContent = "等待校准摘要";
    $("rfVerdictMetrics").innerHTML = '<p class="empty-copy">候选全部完成后，在此展示模型级最大分数与校准阈值。</p>';
    return;
  }
  const verdict = summary.verdict || "INCONCLUSIVE";
  const statusClass = verdict === "DETECTED" ? "is-detected" : "is-inconclusive";
  $("rfVerdictCode").textContent = verdict;
  $("rfVerdictMetrics").innerHTML = `<div class="${statusClass}"><span>裁决</span><strong>${escapeHtml(verdict)}</strong></div><div><span>模型级最大分数</span><strong>${summary.score == null ? "-" : Number(summary.score).toFixed(4)}</strong></div><div><span>冻结阈值</span><strong>${summary.threshold == null ? "未加载" : Number(summary.threshold).toFixed(4)}</strong></div><div><span>候选数 / 耗时</span><strong>${summary.candidate_count ?? probe?.rank ?? 0} / ${summary.elapsed_seconds == null ? "-" : `${Number(summary.elapsed_seconds).toFixed(1)} s`}</strong></div>`;
}

function renderReferenceFreeLive() {
  if (isCompetitionMode(state.live.detectorMode)) {
    renderCompetitionWorkbench();
  } else {
    renderReferenceFreeDiscovery();
    renderReferenceFreeProbe();
    renderReferenceFreeVerdict();
  }
}

function renderLiveMonitor(job) {
  const changes = captureLiveEvents(job.events);
  const detectorMode = job.detector_mode || state.live.detectorMode;
  state.live.detectorMode = detectorMode;
  const referenceFree = isSingleModelProbe(detectorMode);
  $("jobPanel").classList.toggle("is-competition", isCompetitionMode(detectorMode));
  const names = isCompetitionMode(detectorMode)
    ? { queued: "任务排队", loading_models: "正在载入待审模型", output_discovery: "阶段一 · 隐式输出候选挖掘", soft_trigger_probe: "阶段二 · 潜变量前缀探测", calibrated_verdict: "阶段三 · 双条件校准判定", completed: "检测完成", failed: "检测失败", cancelled: "检测已取消" }
    : referenceFree
    ? { queued: "任务排队", loading_models: "正在载入待审模型", output_discovery: "阶段一 · 输出候选生成", soft_trigger_probe: "阶段二 · 软触发对照", calibrated_verdict: "阶段三 · 校准与裁决", completed: "检测完成", failed: "检测失败", cancelled: "检测已取消" }
    : { queued: "任务排队", loading_models: "正在载入模型", output_discovery: "阶段一 · 异常输出发现", trigger_inversion: "阶段二 · 触发器逆向", forward_reproduction: "阶段三 · 正向验证", completed: "检测完成", failed: "检测失败", cancelled: "检测已取消" };
  $("jobStage").textContent = names[job.stage] || job.stage;
  $("jobProgress").textContent = `${job.progress}%`;
  $("liveDashboardLink").href = `/static/live.html?job=${encodeURIComponent(job.id)}`;
  $("liveDashboardLink").hidden = !isCompetitionMode(detectorMode);
  const scenario = state.scenarios.find((item) => item.id === job.scenario);
  const modeLabel = isCompetitionMode(detectorMode) ? "隐式条件后门检测" : referenceFree ? "无参考主检测" : "参考辅助取证";
  $("liveModeChip").textContent = `${modeLabel} · ${scanRoleText(job.scan_role)}${scenario ? ` · ${scenario.short_label}` : ""}`;
  $("jobProgressBar").style.width = `${job.progress}%`;
  $("jobLogs").textContent = (job.logs || []).slice(-25).join("\n") || "等待检测进程输出";
  renderLiveParameters(job.parameters || state.live.parameters);
  setLiveStage(liveStage(job), detectorMode);
  if (referenceFree) {
    if (changes.size || job.status === "completed") renderReferenceFreeLive();
  } else if (changes.size || job.status === "completed") {
    renderLiveDiscovery();
    renderLiveInversion();
    renderLiveValidation();
  }
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await api(`/api/scans/${state.jobId}`);
    renderLiveMonitor(job);
    if (job.status === "completed") {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      const report = await api(job.result_url);
      renderReport(report);
      const catalog = await api("/api/catalog");
      state.catalog = displayCatalogItems(catalog.items);
      renderCatalog();
      $("cancelJobBtn").hidden = true;
      $("closeScanBtn").disabled = false;
      toast("检测完成，独立报告已归档");
    } else if (["failed", "cancelled"].includes(job.status)) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      $("scanError").textContent = job.error || "检测未完成，请查看运行日志。";
      $("startScanBtn").disabled = false;
      $("cancelJobBtn").hidden = true;
      $("closeScanBtn").disabled = false;
    }
  } catch (error) { $("scanError").textContent = error.message; }
}

async function startScan(event) {
  event.preventDefault();
  $("scanError").textContent = "";
  $("startScanBtn").disabled = true;
  try {
    const detectorMode = selectedDetectorMode();
    const competitionMode = isCompetitionMode(detectorMode);
    const mode = competitionMode ? "coverage_audit" : "formal_blind";
    const payload = {
      target: $("targetInput").value.trim(),
      reference_lora: competitionMode ? null : $("referenceInput").value.trim(),
      detector_mode: detectorMode,
      config: competitionMode ? "competition_core/configs/gpt2_detection_4060.yaml" : "configs/detection.yaml",
      preset: $("presetInput").value,
      dtype: $("dtypeInput").value,
      scenario: competitionMode ? "general" : selectedScenario(),
      soft_probe_calibration: null,
      scan_mode: mode,
    };
    const job = await api("/api/scans", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.jobId = job.id;
    resetLiveState();
    $("scanSetup").hidden = true;
    $("jobPanel").hidden = false;
    $("cancelJobBtn").hidden = false;
    $("closeScanBtn").disabled = true;
    renderLiveMonitor(job);
    state.pollTimer = window.setInterval(pollJob, 1100);
  } catch (error) { $("scanError").textContent = error.message; $("startScanBtn").disabled = false; }
}

async function loadInitialData() {
  try {
    const [health, catalog, models, scenarios, calibrations] = await Promise.all([api("/api/health"), api("/api/catalog"), api("/api/models"), api("/api/scenarios"), api("/api/calibrations")]);
    $("serviceState").title = `API ${health.version} · Python ${health.python}`;
    state.catalog = displayCatalogItems(catalog.items);
    state.models = models.items || [];
    state.modelRoots = models.search_roots || [];
    state.scenarios = scenarios.items || [];
    state.calibrations = calibrations.items || [];
    renderCatalog();
    renderModelOptions();
    renderScenarioOptions();
    renderCalibrationOptions();
    const initial = state.catalog.find((item) => item.available && catalogMethod(item) === "implicit")
      || state.catalog.find((item) => item.available);
    if (initial) await loadReport(initial.id);
    else showImplicitEmptyState();
  } catch (error) {
    $("serviceState").classList.add("is-offline");
    $("loadingState").innerHTML = `<p>检测服务不可用：${escapeHtml(error.message)}</p>`;
  }
}

function showImplicitEmptyState() {
  $("reportView").hidden = true;
  $("loadingState").hidden = false;
  $("loadingState").innerHTML = `<section class="implicit-empty-state"><p class="eyebrow">隐式条件后门检测</p><h1>选择一个本地待审模型</h1><p>检测固定使用单模型全词表挖掘与潜变量前缀探测，不加载干净参考模型，也不读取训练条件或目标输出。</p><button id="emptyScanBtn" class="button button-primary" type="button">新建隐式后门检测</button></section>`;
  $("emptyScanBtn").addEventListener("click", openScanDialog);
}

function setSidebarOpen(open) {
  document.body.classList.toggle("sidebar-open", open);
  $("sidebarToggle").setAttribute("aria-expanded", String(open));
  $("sidebarBackdrop").hidden = !open;
}

function openScanDialog() {
  if (!state.pollTimer) {
    state.jobId = null;
    resetLiveState();
    $("scanSetup").hidden = false;
    $("jobPanel").hidden = true;
    $("startScanBtn").disabled = false;
    $("cancelJobBtn").hidden = true;
    $("closeScanBtn").disabled = false;
    $("scanError").textContent = "";
    syncScanSetup();
  }
  $("scanDialog").showModal();
}

$("openScanBtn").addEventListener("click", openScanDialog);
$("sidebarToggle").addEventListener("click", () => setSidebarOpen(!document.body.classList.contains("sidebar-open")));
$("sidebarBackdrop").addEventListener("click", () => setSidebarOpen(false));
$("closeScanBtn").addEventListener("click", () => $("scanDialog").close());
$("scanForm").addEventListener("submit", startScan);
$("refreshModelsBtn").addEventListener("click", () => refreshModels().catch((error) => { $("scanError").textContent = error.message; }));
$("addModelRootBtn").addEventListener("click", () => addModelRoot().catch((error) => { $("scanError").textContent = error.message; }));
$("targetInput").addEventListener("change", renderModelOptions);
$("referenceInput").addEventListener("change", renderModelSelectionInfo);
$("calibrationInput").addEventListener("change", () => {
  renderCalibrationInfo();
  syncScanSetup();
});
document.querySelectorAll('input[name="scanMode"]').forEach((input) => input.addEventListener("change", renderModelOptions));
document.querySelectorAll('input[name="detectorMode"]').forEach((input) => input.addEventListener("change", renderModelOptions));
document.querySelectorAll("[data-catalog-method]").forEach((button) => button.addEventListener("click", () => {
  state.catalogMethod = button.dataset.catalogMethod;
  renderCatalog();
}));
document.querySelectorAll("[data-report-view]").forEach((button) => button.addEventListener("click", () => setReportView(button.dataset.reportView)));
$("processPrevBtn").addEventListener("click", () => { stopProcessPlayer(); stepProcessPlayer(-1); });
$("processNextBtn").addEventListener("click", () => { stopProcessPlayer(); stepProcessPlayer(1); });
$("processPlayBtn").addEventListener("click", () => {
  if (state.processPlayer.playing) stopProcessPlayer();
  else {
    const stage = state.processPlayer.stages[state.processPlayer.stageIndex];
    if (state.processPlayer.stageIndex === state.processPlayer.stages.length - 1 && state.processPlayer.frameIndex === (stage?.frames.length || 1) - 1) {
      state.processPlayer.stageIndex = 0;
      state.processPlayer.frameIndex = 0;
      renderProcessFrame();
    }
    startProcessPlayer();
  }
});
$("processScrubber").addEventListener("input", (event) => {
  stopProcessPlayer();
  state.processPlayer.frameIndex = Number(event.target.value);
  renderProcessFrame();
});
document.querySelectorAll("[data-player-speed]").forEach((button) => button.addEventListener("click", () => {
  const wasPlaying = state.processPlayer.playing;
  state.processPlayer.speed = Number(button.dataset.playerSpeed);
  document.querySelectorAll("[data-player-speed]").forEach((item) => item.classList.toggle("is-active", item === button));
  if (wasPlaying) startProcessPlayer();
}));
$("openExperienceBtn").addEventListener("click", () => {
  setReportView("evidence");
  $("competitionExperienceStage").classList.add("is-open");
  document.body.classList.add("experience-open");
});
$("closeExperienceBtn").addEventListener("click", () => {
  $("competitionExperienceStage").classList.remove("is-open");
  document.body.classList.remove("experience-open");
});
$("refreshBtn").addEventListener("click", async () => { const data = await api("/api/catalog"); state.catalog = displayCatalogItems(data.items); renderCatalog(); if (!state.catalog.length) showImplicitEmptyState(); toast("检测记录已刷新"); });
$("cancelJobBtn").addEventListener("click", async () => { if (state.jobId) { await api(`/api/scans/${state.jobId}`, { method: "DELETE" }); await pollJob(); } });
$("competitionStepPrev").addEventListener("click", () => {
  const core = state.competitionReport.report?.evidence?.competition_core;
  if (!core) return;
  state.competitionReport.probeStep = Math.max(0, state.competitionReport.probeStep - 1);
  renderCompetitionProbeStep(core);
});
$("competitionStepNext").addEventListener("click", () => {
  const core = state.competitionReport.report?.evidence?.competition_core;
  if (!core) return;
  state.competitionReport.probeStep += 1;
  renderCompetitionProbeStep(core);
});
$("experienceRunBtn").addEventListener("click", () => { void runCompetitionExperience(); });
$("experienceStopBtn").addEventListener("click", () => state.experience.controller?.abort());
$("scanDialog").addEventListener("click", (event) => { if (event.target === $("scanDialog") && !state.pollTimer) $("scanDialog").close(); });

loadInitialData();

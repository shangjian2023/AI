"use strict";

(function exposeCompetitionLive(global) {
  function create(deps) {
    const {
      $, state, escapeHtml, fixed, probability, selectionModeText,
      softReplayExamplesHtml, candidateInteractions, candidateTokenTexts,
      currentRfProbe, calibratedCompetitionDecision, evidenceSummaryHtml,
      shardGridHtml,
    } = deps;

    function renderCompetitionProbe() {
      const probe = currentRfProbe();
      if (!probe) {
        $("rfProbeProgress").textContent = "等待 Top-4";
        $("rfProbeInputs").innerHTML = '<p class="empty-copy">候选合并后依次执行连续潜变量探测。</p>';
        $("rfCandidateTarget").innerHTML = '<p class="empty-copy">等待候选。</p>';
        $("rfBaselineTargets").innerHTML = '<p class="empty-copy">等待内部对照。</p>';
        $("rfTrajectory").innerHTML = "";
        return;
      }
      $("rfProbeProgress").textContent = probe.evidence
        ? `候选 #${probe.rank} 完成 · 概率差 ${Number(probe.max_probability_gap || 0).toFixed(4)}`
        : `候选 #${probe.rank} 正在探测`;
      $("rfProbeInputs").innerHTML = `<div><span>候选族支持度</span><code>${escapeHtml(probe.family_support ?? "等待结果")}</code></div><div><span>论文概率判据</span><code>${probe.criterion_met == null ? "计算中" : probe.criterion_met ? "已越过 0.25" : "未越过 0.25"}</code></div>`;
      $("rfCandidateTarget").innerHTML = `<code>${escapeHtml(probe.candidate_output || "-")}</code>`;
      $("rfBaselineTargets").classList.toggle("baseline", true);
      $("rfBaselineTargets").innerHTML = `<code>${escapeHtml(probe.control_output || "正在构造等长无重叠对照")}</code>`;
      const steps = probe.steps || [];
      $("rfStepCount").textContent = `${steps.length} 个轨迹采样点`;
      $("rfTrajectory").innerHTML = steps.length
        ? steps.map((item) => `<div class="rf-trajectory-row"><code>#${escapeHtml(item.step)}</code><span title="候选平均 token 概率"><i style="width:${Math.max(2, Number(item.candidate_probability || 0) * 100)}%"></i></span><span title="内部对照平均 token 概率"><i class="control" style="width:${Math.max(2, Number(item.control_probability || 0) * 100)}%"></i></span><strong>${Number(item.probability_gap || 0).toFixed(3)}</strong></div>`).join("")
        : '<p class="empty-copy">等待候选概率轨迹。</p>';
    }

    function renderCompetitionVerdict() {
      const summary = state.live.rf.summary;
      if (!summary) {
        $("rfVerdictCode").textContent = "检测中";
        $("rfVerdictMetrics").innerHTML = '<p class="empty-copy">Top-4 完成后按照对数似然差 + 候选族支持双条件给出检测结论。</p>';
        return;
      }
      const decision = calibratedCompetitionDecision(summary);
      const logMet = Boolean(summary.log_likelihood_criterion_met);
      const familyMet = Boolean(summary.family_log_likelihood_criterion_met);
      $("rfVerdictCode").textContent = `${decision.code} · ${decision.text}`;
      $("rfVerdictMetrics").innerHTML = `<div class="${decision.className}"><span>正式检测结论</span><strong>${decision.code}</strong></div><div><span>对数似然差判据</span><strong>${logMet ? "满足" : "未满足"} · ${fixed(summary.maximum_log_likelihood_gap)} / ${fixed(summary.log_likelihood_gap_threshold, 1)}</strong></div><div><span>同候选联合判据</span><strong>${familyMet ? "满足 · 双条件通过" : "未满足 · 不检出"}</strong></div><div><span>最大族支持 / 门槛</span><strong>${Number(summary.maximum_family_support || 0)} / ${Number(summary.minimum_family_support || 0)}</strong></div>`;
    }

    function renderLiveCompetitionCandidates() {
      const rf = state.live.rf;
      const candidates = rf.candidates || [];
      let active = candidates.find((item) => Number(item.rank) === Number(rf.activeCandidateRank));
      if (!active) active = candidates[0];
      if (!active) {
        $("liveCompetitionCandidates").innerHTML = '<p class="empty-copy">四个分片合并后，候选会出现在这里。</p>';
        $("liveCompetitionTokenTrace").innerHTML = "";
        return;
      }
      rf.activeCandidateRank = Number(active.rank);
      $("liveCompetitionCandidates").innerHTML = `<div class="panel-label"><span>已合并候选</span><small>选择一个候选核对逐 token 输入与输出</small></div><div class="live-candidate-buttons">${candidates.slice(0, 12).map((candidate) => `<button type="button" data-live-competition-candidate="${escapeHtml(candidate.rank)}" class="${Number(candidate.rank) === Number(active.rank) ? "is-active" : ""}"><b>#${escapeHtml(candidate.rank)}</b><span>${escapeHtml(candidate.text)}</span><small>后缀 ${probability(candidate.suffix_probability, 1)} · 族支持 ${Number(candidate.family_support || 0)}</small></button>`).join("")}</div>`;
      const interactions = candidateInteractions(active, rf.responsePrefix);
      const tokenTexts = candidateTokenTexts(active);
      $("liveCompetitionTokenTrace").innerHTML = `<div class="panel-label"><span>候选 #${escapeHtml(active.rank)} 的模型交互</span><small>首 token 是遍历种子；其后每一行对应一次真实前向输出</small></div><div class="live-token-table"><div class="token-interaction seed-row"><b>种子</b><code>${escapeHtml(rf.responsePrefix || "响应起点")}</code><div><code>${escapeHtml(tokenTexts[0] || active.token_ids?.[0] || "-")}</code><small>词表枚举</small></div><strong>遍历值</strong><span>不是生成输出</span></div>${interactions.map((item) => `<div class="token-interaction"><b>#${escapeHtml(item.step)}</b><code>${escapeHtml(item.input_text)}</code><div><code>${escapeHtml(item.output_token_text)}</code><small>token ${escapeHtml(item.output_token_id)}</small></div><strong>${probability(item.output_probability)}</strong><span>${escapeHtml(selectionModeText(item.selection_mode))}</span></div>`).join("")}</div>`;
      document.querySelectorAll("[data-live-competition-candidate]").forEach((button) => button.addEventListener("click", () => {
        rf.activeCandidateRank = Number(button.dataset.liveCompetitionCandidate);
        renderLiveCompetitionCandidates();
      }));
    }

    function renderLiveCompetitionProbe() {
      const rf = state.live.rf;
      const probes = [...rf.probes.values()].sort((a, b) => Number(a.rank || 0) - Number(b.rank || 0));
      let active = probes.find((item) => Number(item.rank) === Number(rf.activeProbeRank));
      if (!active) active = probes.at(-1);
      if (!active) {
        $("liveCompetitionProbeNav").innerHTML = '<p class="empty-copy">等待 Top-4 候选进入潜变量探测。</p>';
        $("liveCompetitionProbeDetail").innerHTML = "";
        return;
      }
      rf.activeProbeRank = Number(active.rank);
      const steps = active.steps || [];
      if (rf.activeProbeStep == null || rf.activeProbeStep >= steps.length) rf.activeProbeStep = Math.max(0, steps.length - 1);
      const step = steps[rf.activeProbeStep];
      $("liveCompetitionProbeNav").innerHTML = probes.map((probe) => `<button type="button" data-live-competition-probe="${escapeHtml(probe.rank)}" class="${Number(probe.rank) === Number(active.rank) ? "is-active" : ""}"><span>候选 #${escapeHtml(probe.rank)}</span><strong>${fixed(probe.max_probability_gap)}</strong><small>${probe.evidence ? "已完成" : "计算中"} · 最大概率差</small></button>`).join("");
      const inputs = new Map((rf.probeInputs || []).map((item) => [Number(item.index), item.text]));
      const promptIndices = step?.prompt_indices || [];
      const candidateText = active.candidate_output || rf.candidates.find((item) => Number(item.rank) === Number(active.rank))?.text || "等待候选输出";
      const controlText = active.control_output || "正在构造等长无重叠对照";
      const replay = active.replay || {};
      const batchInputs = promptIndices.length
        ? promptIndices.map((index) => `<li><b>#${Number(index) + 1}</b><code>${escapeHtml(inputs.get(Number(index)) || `输入索引 ${index}`)}</code></li>`).join("")
        : '<li class="empty-copy">候选完成后显示本步实际输入索引与文本。</li>';
      const trajectory = steps.length ? steps.map((item, index) => `<button type="button" data-live-competition-step="${index}" class="${index === rf.activeProbeStep ? "is-active" : ""}"><b>#${escapeHtml(item.step)}</b><span>${probability(item.candidate_probability, 2)}</span><span>${probability(item.control_probability, 2)}</span><strong>${fixed(item.probability_gap)}</strong></button>`).join("") : '<p class="empty-copy">正在等待逐步概率输出。</p>';
      const logThreshold = Number(rf.summary?.log_likelihood_gap_threshold ?? 2.0);
      $("liveCompetitionProbeDetail").innerHTML = `<div class="live-probe-targets"><section class="candidate-side"><span>候选输出</span><small>模型异常确信的片段</small><code>${escapeHtml(candidateText)}</code></section><section class="control-side"><span>内部对照</span><small>等长且 token 不重叠的普通片段</small><code>${escapeHtml(controlText)}</code></section></div><div class="live-probe-step"><div class="probe-batch-inputs"><div class="panel-label"><span>第 ${escapeHtml(step?.step || "-")} 步实际输入</span><small>Epoch ${escapeHtml(step?.epoch || "-")} · Batch ${escapeHtml(step?.batch || "-")} · ${promptIndices.length || rf.batchSize || "?"} 条问题</small></div><ol>${batchInputs}</ol></div><div class="live-forward-output"><div class="candidate-side"><span>模型输出 A · 候选平均概率</span><strong>${step ? probability(step.candidate_probability, 3) : "-"}</strong><small>损失 ${step ? fixed(step.candidate_loss, 5) : "-"}</small></div><div class="control-side"><span>模型输出 B · 对照平均概率</span><strong>${step ? probability(step.control_probability, 3) : "-"}</strong><small>损失 ${step ? fixed(step.control_loss, 5) : "-"}</small></div><div class="gap-side"><span>论文概率差 / 复现线</span><strong>${step ? fixed(step.probability_gap) : "-"} / 0.2500</strong><small>平均对数似然差 ${step ? fixed(step.log_likelihood_gap) : "-"} / ${logThreshold.toFixed(4)} · 当前联合判据</small></div></div></div><section class="live-soft-replay"><header><span>新输入白盒回放</span><strong>${replay.sample_count ? `${Number(replay.soft_trigger_exact_prefix_match_count || 0)} / ${Number(replay.sample_count)} 条复现` : "等待回放"}</strong><small>新输入对数似然差 ${replay.sample_count ? fixed(replay.log_likelihood_gap) : "-"} · 回放本身不参与最终裁决</small></header><div class="soft-replay-examples">${softReplayExamplesHtml(replay)}</div></section><div class="live-probe-trajectory"><div class="panel-label"><span>全部优化步</span><small>候选概率 / 对照概率 / 概率差；点一行核对对应输入</small></div><div>${trajectory}</div></div>`;
      document.querySelectorAll("[data-live-competition-probe]").forEach((button) => button.addEventListener("click", () => {
        rf.activeProbeRank = Number(button.dataset.liveCompetitionProbe);
        rf.activeProbeStep = null;
        renderLiveCompetitionProbe();
      }));
      document.querySelectorAll("[data-live-competition-step]").forEach((button) => button.addEventListener("click", () => {
        rf.activeProbeStep = Number(button.dataset.liveCompetitionStep);
        renderLiveCompetitionProbe();
      }));
    }

    function renderCompetitionWorkbench() {
      const rf = state.live.rf;
      const stageOrder = ["output_discovery", "soft_trigger_probe", "calibrated_verdict"];
      const currentIndex = Math.max(0, stageOrder.indexOf(state.live.activeStage));
      document.querySelectorAll("[data-live-competition-stage]").forEach((section) => {
        const index = stageOrder.indexOf(section.dataset.liveCompetitionStage);
        section.classList.toggle("is-current", index === currentIndex);
        section.classList.toggle("is-complete", index < currentIndex || Boolean(rf.summary));
      });
      $("liveCompetitionState").textContent = {
        output_discovery: "正在扫描完整词表",
        soft_trigger_probe: "正在逐候选比较概率",
        calibrated_verdict: "双条件校准结论已生成",
      }[state.live.activeStage] || "正在准备";
      $("liveCompetitionShards").innerHTML = shardGridHtml([...rf.shards.values()]);
      renderLiveCompetitionCandidates();
      renderLiveCompetitionProbe();
      $("liveCompetitionVerdict").innerHTML = rf.summary
        ? evidenceSummaryHtml(rf.summary)
        : '<p class="empty-copy">Top-4 全部完成后，按照对数似然差 + 候选族支持双条件给出竞赛检测结论。</p>';
    }

    return { renderCompetitionProbe, renderCompetitionVerdict, renderCompetitionWorkbench };
  }

  global.BdShieldCompetitionLive = Object.freeze({ create });
}(window));

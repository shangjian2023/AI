"use strict";

(function exposeCompetitionReport(global) {
  function create(deps) {
    const {
      $, state, escapeHtml, fixed, probability, selectionModeText,
      softReplayExamplesHtml, normalizedShards, shardGridHtml,
      calibratedCompetitionDecision, evidenceSummaryHtml,
      renderCompetitionExperience,
    } = deps;

    function candidateTokenTexts(candidate) {
      const ids = candidate?.token_ids || [];
      const texts = candidate?.token_texts || [];
      return ids.map((tokenId, index) => texts[index] == null ? `<token:${tokenId}>` : String(texts[index]));
    }

    function candidateInteractions(candidate, responsePrefix) {
      if (candidate?.interactions?.length) return candidate.interactions;
      const ids = candidate?.token_ids || [];
      const texts = candidateTokenTexts(candidate);
      const probabilities = candidate?.continuation_probabilities || [];
      const modes = candidate?.selection_modes || [];
      return probabilities.map((outputProbability, index) => ({
        step: index + 1,
        input_text: `${responsePrefix || ""}${texts.slice(0, index + 1).join("")}`,
        input_token_ids: ids.slice(0, index + 1),
        output_token_id: ids[index + 1],
        output_token_text: texts[index + 1],
        output_probability: outputProbability,
        selection_mode: modes[index] || (candidate?.used_beam ? "beam_assisted_route" : "greedy"),
      })).filter((item) => item.output_token_id != null);
    }

    function renderCompetitionCandidate(core) {
      const mining = core.mining || {};
      const candidates = mining.candidates || [];
      let active = candidates.find((item) => Number(item.rank) === Number(state.competitionReport.candidateRank));
      if (!active) active = candidates[0];
      if (!active) {
        $("competitionCandidateNav").innerHTML = '<p class="empty-copy">没有保存候选输出。</p>';
        $("competitionCandidateSummary").innerHTML = "";
        $("competitionTokenTrace").innerHTML = '<p class="empty-copy">没有可复核的逐 token 交互。</p>';
        return;
      }
      state.competitionReport.candidateRank = Number(active.rank);
      $("competitionCandidateNav").innerHTML = candidates.slice(0, 12).map((candidate) => `<button type="button" data-competition-candidate-rank="${escapeHtml(candidate.rank)}" class="${Number(candidate.rank) === Number(active.rank) ? "is-active" : ""}"><b>#${escapeHtml(candidate.rank)}</b><span>${escapeHtml(candidate.text)}</span><small>后缀 ${probability(candidate.suffix_probability, 1)} · 族支持 ${Number(candidate.family_support || 0)}</small></button>`).join("");
      $("competitionCandidateSummary").innerHTML = `<div><span>当前候选完整文本</span><code>${escapeHtml(active.text)}</code></div><div><span>token 数</span><strong>${Number(active.token_count || active.token_ids?.length || 0)}</strong><small>模型内部处理的最小文本单位数量</small></div><div><span>后缀最低概率</span><strong>${probability(active.suffix_probability)}</strong><small>尾部最没把握的一个 token 仍有多确信</small></div><div><span>生成路线</span><strong>${active.used_beam ? "Beam 辅助" : "Greedy"}</strong><small>${active.used_beam ? "中途保留过多条候选路线" : "每步直接取最高概率 token"}</small></div>`;
      const tokenTexts = candidateTokenTexts(active);
      const seedId = active.token_ids?.[0];
      const seed = seedId == null ? "" : `<div class="token-interaction seed-row"><b>种子</b><code>${escapeHtml(mining.response_prefix || "响应起点")}</code><div><code>${escapeHtml(tokenTexts[0])}</code><small>token ${escapeHtml(seedId)}</small></div><strong>遍历值</strong><span>首 token 枚举<small>不是模型生成输出</small></span></div>`;
      const interactions = candidateInteractions(active, mining.response_prefix);
      $("competitionTokenTrace").innerHTML = seed + interactions.map((item) => `<div class="token-interaction"><b>#${escapeHtml(item.step)}</b><code>${escapeHtml(item.input_text)}</code><div><code>${escapeHtml(item.output_token_text)}</code><small>token ${escapeHtml(item.output_token_id)}</small></div><strong>${probability(item.output_probability)}</strong><span>${escapeHtml(selectionModeText(item.selection_mode))}</span></div>`).join("");
      document.querySelectorAll("[data-competition-candidate-rank]").forEach((button) => button.addEventListener("click", () => {
        state.competitionReport.candidateRank = Number(button.dataset.competitionCandidateRank);
        renderCompetitionCandidate(core);
      }));
    }

    function renderCompetitionProbeStep(core) {
      const evidence = core.probe_evidence || [];
      let active = evidence.find((item) => Number(item.rank) === Number(state.competitionReport.probeRank));
      if (!active) active = evidence[0];
      if (!active) {
        $("competitionProbeNav").innerHTML = '<p class="empty-copy">报告未保存潜变量探测结果。</p>';
        $("competitionProbeBatchInputs").innerHTML = '<li>没有可复核的输入批次。</li>';
        return;
      }
      state.competitionReport.probeRank = Number(active.rank);
      const result = active.probe || {};
      const replay = active.replay || {};
      const steps = result.steps || [];
      state.competitionReport.probeStep = Math.max(0, Math.min(state.competitionReport.probeStep, Math.max(0, steps.length - 1)));
      const step = steps[state.competitionReport.probeStep];
      $("competitionProbeNav").innerHTML = evidence.map((item) => `<button type="button" data-competition-probe-rank="${escapeHtml(item.rank)}" class="${Number(item.rank) === Number(active.rank) ? "is-active" : ""}"><span>候选 #${escapeHtml(item.rank)}</span><strong>${fixed(item.probe?.max_log_likelihood_gap)}</strong><small>最大对数似然差 · 族支持 ${Number(item.family_support || 0)}</small></button>`).join("");
      $("competitionCandidateOutput").textContent = result.candidate_text || "-";
      $("competitionControlOutput").textContent = result.control_text || "-";
      $("competitionProbeMetric").textContent = `候选 #${active.rank} · ${steps.length} 次模型对照`;
      $("competitionStepPrev").disabled = !step || state.competitionReport.probeStep <= 0;
      $("competitionStepNext").disabled = !step || state.competitionReport.probeStep >= steps.length - 1;
      $("competitionStepPosition").textContent = step ? `${state.competitionReport.probeStep + 1} / ${steps.length} · Epoch ${step.epoch || "-"} · Batch ${step.batch || "-"}` : "未保存轨迹";
      const inputs = new Map((core.probe_inputs || []).map((item) => [Number(item.index), item.text]));
      const promptIndices = step?.prompt_indices || [];
      $("competitionProbeBatchInputs").innerHTML = promptIndices.length
        ? promptIndices.map((index) => `<li><b>#${Number(index) + 1}</b><code>${escapeHtml(inputs.get(Number(index)) || `输入索引 ${index}（文本未保存）`)}</code></li>`).join("")
        : '<li class="empty-copy">该历史轨迹未保存本步输入索引。</li>';
      const softTokens = Number(core.probe_config?.soft_token_count || 0);
      const inputCount = promptIndices.length || Number(core.probe_config?.batch_size || 0);
      $("competitionCandidateInputRecipe").textContent = `${inputCount} 条上列问题 + ${softTokens || "?"} 个连续潜变量向量 + 候选输出`;
      $("competitionControlInputRecipe").textContent = `${inputCount} 条相同问题 + ${softTokens || "?"} 个等长潜变量向量 + 内部对照`;
      $("competitionCandidateProbability").textContent = step ? probability(step.candidate_probability, 3) : "-";
      $("competitionControlProbability").textContent = step ? probability(step.control_probability, 3) : "-";
      $("competitionCandidateLoss").textContent = step ? fixed(step.candidate_loss, 5) : "-";
      $("competitionControlLoss").textContent = step ? fixed(step.control_loss, 5) : "-";
      $("competitionProbabilityGap").textContent = step ? `${fixed(step.probability_gap)} / 0.2500` : "-";
      $("competitionGapMeter").style.setProperty("--gap-width", step ? `${Math.min(100, Math.max(0, Number(step.probability_gap || 0) / 0.25 * 100))}%` : "0%");
      $("competitionLogLikelihoodGap").textContent = step ? fixed(step.log_likelihood_gap) : fixed(result.max_log_likelihood_gap);
      $("competitionReplayRate").textContent = replay.sample_count ? probability(replay.soft_trigger_exact_prefix_match_rate, 1) : "-";
      $("competitionReplayLogGap").textContent = replay.sample_count ? fixed(replay.log_likelihood_gap) : "-";
      $("competitionReplayMatch").textContent = replay.sample_count ? `${Number(replay.soft_trigger_exact_prefix_match_count || 0)} / ${Number(replay.sample_count)} 条完整复现` : "等待回放";
      const refinement = active.replay_refinement || {};
      $("competitionReplayRefinement").textContent = refinement.used ? `已启用 · ${Number(refinement.steps || 0)} 步` : "未启用";
      const artifact = active.soft_trigger_artifact || {};
      $("competitionReplayArtifact").textContent = artifact.sha256 ? `已保存 · ${String(artifact.sha256).slice(0, 10)}…` : "未保存";
      $("competitionReplayExamples").innerHTML = softReplayExamplesHtml(replay);
      $("competitionTrajectory").innerHTML = steps.length ? `<div class="trajectory-header"><span>步</span><span>Epoch / Batch</span><span>候选概率</span><span>对照概率</span><span>概率差</span><span>对数似然差</span></div>${steps.map((item, index) => `<button type="button" data-competition-step="${index}" class="${index === state.competitionReport.probeStep ? "is-active" : ""}"><b>#${escapeHtml(item.step)}</b><span>${escapeHtml(item.epoch || "-")} / ${escapeHtml(item.batch || "-")}</span><strong class="candidate-value">${probability(item.candidate_probability, 2)}</strong><strong class="control-value">${probability(item.control_probability, 2)}</strong><strong>${fixed(item.probability_gap)}</strong><strong>${fixed(item.log_likelihood_gap)}</strong></button>`).join("")}` : '<p class="empty-copy">没有保存逐步概率轨迹。</p>';
      document.querySelectorAll("[data-competition-probe-rank]").forEach((button) => button.addEventListener("click", () => {
        state.competitionReport.probeRank = Number(button.dataset.competitionProbeRank);
        state.competitionReport.probeStep = 0;
        renderCompetitionProbeStep(core);
      }));
      document.querySelectorAll("[data-competition-step]").forEach((button) => button.addEventListener("click", () => {
        state.competitionReport.probeStep = Number(button.dataset.competitionStep);
        renderCompetitionProbeStep(core);
      }));
    }

    function renderCompetitionReport(report) {
      const core = report.evidence?.competition_core || {};
      const mining = core.mining || {};
      const decision = calibratedCompetitionDecision(core.summary || {});
      state.competitionReport.report = report;
      state.competitionReport.candidateRank = Number(mining.candidates?.[0]?.rank || 1);
      state.competitionReport.probeRank = Number(core.probe_evidence?.[0]?.rank || 1);
      state.competitionReport.probeStep = 0;
      $("competitionVocabularyMetric").textContent = `${Number(mining.vocabulary_size || 0).toLocaleString()} 个 token · ${Number(mining.candidates?.length || 0)} 个候选`;
      $("competitionShardGrid").innerHTML = shardGridHtml(normalizedShards(core.shards, mining));
      $("competitionDecisionBadge").textContent = `${decision.code} · ${decision.text}`;
      $("competitionEvidenceSummary").innerHTML = evidenceSummaryHtml(core.summary || {});
      renderCompetitionCandidate(core);
      renderCompetitionProbeStep(core);
      renderCompetitionExperience(core);
    }

    return {
      candidateInteractions,
      candidateTokenTexts,
      renderCompetitionProbeStep,
      renderCompetitionReport,
    };
  }

  global.BdShieldCompetitionReport = Object.freeze({ create });
}(window));

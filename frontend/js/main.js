// 主入口：初始化视图路由、交互与数据加载
import * as API from './api.js?v=70';
import { showModal, initModal } from './ui.js';
import { ring, donut, hbar, bipartite, heatmap, kwBars } from './charts.js?v=17';

let questions = [];
let currentQuestion = null;
let pendingConfidence = 0;
let currentBank = 'all';

// ---------- 视图路由（二级菜单） ----------
function initRouting() {
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.addEventListener('click', () => {
      const view = item.dataset.view;
      showView(view);
    });
  });
  initNavGroups();
}

// 导航分组折叠：点击分组标题展开/收起，次级分区默认折叠以突出重点
function initNavGroups() {
  document.querySelectorAll('.nav-group-title').forEach((title) => {
    title.addEventListener('click', (e) => {
      e.stopPropagation();
      title.closest('.nav-group').classList.toggle('collapsed');
    });
  });
}

// 切换视图后自动展开当前视图所在分组，保证选中项始终可见
function revealGroup(view) {
  const item = document.querySelector(`.nav-item[data-view="${view}"]`);
  if (!item) return;
  const group = item.closest('.nav-group');
  if (group) group.classList.remove('collapsed');
}

function showView(view) {
  // 高亮导航
  document.querySelectorAll('.nav-item').forEach((i) => i.classList.toggle('active', i.dataset.view === view));
  // 切换视图
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.dataset.view === view));
  revealGroup(view);
  // 实时监控轮询：仅在学习总览页开启
  if (view === 'dashboard') startMonitor(); else if (view === 'dashboard') loadDashboard();
  else if (view === 'question-bank') renderQuestionList();
  else if (view === 'question-answer') setModeAndLoad();
  else if (view === 'analysis-article') { /* 划词已在文章区域默认生效 */ }
  else if (view === 'analysis-cluster') loadClusters();
  else if (view === 'analysis-error') loadErrorClusters();
  else if (view === 'analysis-confusion') loadConfusion();
  else if (view === 'analysis-friction') loadFrictionSelect();
  else if (view === 'analysis-reading') loadReadingHistory();
  else if (view === 'analysis-variant') loadVariantSelect();
  else if (view === 'analysis-graph') loadGraph();
  else if (view === 'settings-modules') loadModules();
  else if (view === 'settings-training') loadTrainingStatus();
  else if (view === 'train-keywords') loadTrainPage();
}

// ---------- 标注与训练（文章关键词 / 读题划词 两套标注共用同一工作台） ----------
const trainState = { items: [], idx: 0, current: null, inited: false, picking: true, mode: 'article', subj: '数学' };

function trainBind() {
  if (trainState.inited) return;
  trainState.inited = true;
  document.getElementById('btnTrainExport').addEventListener('click', trainRunExport);
  document.getElementById('btnTrainRun').addEventListener('click', trainRunModel);
  document.getElementById('btnTrainEval').addEventListener('click', trainRunEval);
  document.getElementById('btnTrainAdd').addEventListener('click', trainAddArticle);
  document.getElementById('btnImportFolder').addEventListener('click', trainImportFolder);
  document.getElementById('btnImportFiles').addEventListener('click', () => document.getElementById('inputImportTxt').click());
  document.getElementById('inputImportTxt').addEventListener('change', (e) => trainImportFiles(e.target.files));
  document.getElementById('btnPrev').addEventListener('click', () => trainNav(-1));
  document.getElementById('btnNext').addEventListener('click', () => trainNav(1));
  document.getElementById('btnSaveLabel').addEventListener('click', trainSave);
  document.getElementById('trainDoc').addEventListener('mouseup', trainDocPick);
  document.querySelectorAll('#trainMode .seg-btn').forEach((b) => b.addEventListener('click', () => trainSetMode(b.dataset.mode)));
  document.querySelectorAll('#huaciSubjectSeg .seg-btn').forEach((b) => b.addEventListener('click', () => trainSetSubject(b.dataset.sub)));
  document.getElementById('btnHuaciExport').addEventListener('click', trainHuaciExport);
}

// 模式切换：文章关键词 ↔ 读题划词，复用同一套标注卡片
function trainSetMode(mode) {
  trainState.mode = mode;
  document.querySelectorAll('#trainMode .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  const isH = mode === 'huaci';
  document.getElementById('toolbarArticle').style.display = isH ? 'none' : '';
  document.getElementById('toolbarHuaci').style.display = isH ? '' : 'none';
  document.getElementById('importCard').style.display = isH ? 'none' : '';
  document.getElementById('trainHuaciModel').style.display = isH ? '' : 'none';
  document.getElementById('trainModeDesc').textContent = isH
    ? '读题划词标注：选学科 → 判断候选词该不该划（0 不划 / 1 可划 / 2 必划）→ 保存 → 导出训练数据'
    : '导入语料 → 标注候选词 → 训练重排模型 → 一键评估，训练过程不影响文章分析主流程';
  document.getElementById('trainLegend').textContent = isH
    ? '0 不划·噪音　|　1 可划·次要　|　2 必划·重点'
    : '0 无关/压掉　|　1 相关·次要　|　2 关键·必上屏';
  document.getElementById('trainEvalTip').textContent = isH
    ? '读题划词模式：在已标注样本上留出 20% 评估「该不该划」0/1/2 的分类能力。'
    : '用 27 篇全新独立测试集（不参与训练）衡量提取效果，约需 1 分钟。';
  document.getElementById('pickTip').textContent = isH
    ? '划词模式已开启：在下方题目里用鼠标选中你认为该划的词，会自动加入标注列表（默认标为 2 必划）。'
    : '划词模式已开启：在下方正文里用鼠标选中你认为的中心词，会自动加入标注列表（默认标为 2 关键）。';
  loadTrainPage();
}

// 学科切换（仅划词模式）
function trainSetSubject(subj) {
  trainState.subj = subj;
  document.querySelectorAll('#huaciSubjectSeg .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.sub === subj));
  loadTrainPage();
}

async function loadTrainPage() {
  trainBind();
  try {
    if (trainState.mode === 'huaci') {
      const [s, itemRes] = await Promise.all([
        API.Train.huaci.status(),
        API.Train.huaci.items(trainState.subj),
      ]);
      renderHuaciStatus(s);
      renderHuaciModel(s);
      trainState.items = itemRes.items || [];
      trainState.idx = 0;
      trainShowDoc();
      document.getElementById('trainHint').textContent =
        itemRes.counts.need_label > 0
          ? `当前学科「${trainState.subj}」：${itemRes.counts.total} 题，已标注 ${itemRes.counts.labeled} 题。给候选词打 0/1/2 表示「该不该划」。`
          : `当前学科「${trainState.subj}」：${itemRes.counts.total} 题已全部标注，可点「导出划词训练数据」。`;
      return;
    }
    const [s, itemRes] = await Promise.all([API.Train.status(), API.Train.items()]);
    renderTrainStatus(s);
    trainState.items = itemRes.items || [];
    trainState.idx = 0;
    trainShowDoc();
    renderTrainModel(s.rerank);
    const hint = document.getElementById('trainHint');
    hint.textContent = s.rerank && s.rerank.ready
      ? `重排模型已启用，文章分析页将用模型打分排序。新训练后请刷新页面生效。`
      : `流程：添加文章 → 「扫描并生成候选词」→ 逐个给候选词打 0/1/2 → 「保存标注」→ 标注 ≥ 20 条后点「用标注训练模型」。`;
  } catch (e) {
    document.getElementById('trainStatus').textContent = '加载失败：' + e.message;
  }
}

function renderHuaciStatus(s) {
  const el = document.getElementById('trainStatus');
  const subs = s.subjects || {};
  const parts = Object.entries(subs).map(([k, v]) => `${k} <b>${v.total}</b> 题 · 已标 <b>${v.labeled}</b>`);
  el.innerHTML = `划词题源：${parts.join('　·　')}<div style="font-size:0.7rem;color:var(--slate)">共 <b>${s.total}</b> 题 · 已标注 <b>${s.labeled}</b> 题。切换学科从对应题源开始标注。</div>`;
}

function renderHuaciModel(s) {
  const el = document.getElementById('trainHuaciModel');
  const c = (s.subjects || {})[trainState.subj] || {};
  el.innerHTML = `当前「${trainState.subj}」已标注 <b>${c.labeled}</b>/${c.total} 题。给候选词打 0/1/2 表示该不该划，保存后点「导出划词训练数据」即可用于训练模型。`;
}

async function trainHuaciExport() {
  const btn = document.getElementById('btnHuaciExport');
  btn.disabled = true; btn.textContent = '导出中…';
  try {
    const res = await API.Train.huaci.export(trainState.subj);
    document.getElementById('trainHint').textContent = res.ok
      ? `已导出 ${res.samples} 条标注样本（${res.labeled_questions} 题）→ ${res.path}`
      : (res.error || '导出失败');
  } catch (e) {
    document.getElementById('trainHint').textContent = '导出失败：' + e.message;
  } finally { btn.disabled = false; btn.textContent = '导出划词训练数据'; }
}

function renderTrainStatus(s) {
  const el = document.getElementById('trainStatus');
  const needLabel = s.articles - s.labeled;
  el.innerHTML = `文章 <b>${s.articles}</b> 篇 · 已导出候选 <b>${s.exported}</b> 篇 · 已标注 <b>${s.labeled}</b> 篇</b> · 待标注 <b>${Math.max(0, needLabel)}</b> 篇
    <div style="font-size:0.7rem;color:var(--slate)">建议标注 10–20 篇文章（每篇给前几个候选打 2）后训练，效果最明显。</div>`;
}

function renderTrainModel(rr) {
  const el = document.getElementById('trainModel');
  if (rr && rr.ready) {
    const m = rr.metrics || {};
    el.innerHTML = `<b>已就绪</b>（版本 ${rr.version || '-'}）· 样本 ${m.n_samples ?? rr.n_samples ?? '-'} · 验证 Spearman ${m.val_spearman ?? '-'} · 档位命中 ${m.val_class_acc !== undefined ? (m.val_class_acc * 100).toFixed(0) + '%' : '-'}`;
  } else {
    el.innerHTML = '未训练。标注足够样本后点击「用标注训练模型」。';
  }
}

async function trainRunExport() {
  const btn = document.getElementById('btnTrainExport');
  btn.disabled = true; btn.textContent = '导出中…';
  try {
    const res = await API.Train.export();
    btn.textContent = '扫描并生成候选词';
    if (res.ok) {
      document.getElementById('trainHint').textContent = `已为 ${res.exported} 篇文章生成候选词${res.skipped_empty ? `，${res.skipped_empty} 篇未提取到有效候选` : ''}。开始标注吧。`;
    } else {
      document.getElementById('trainHint').textContent = res.error || '导出失败';
    }
    await loadTrainPage();
  } catch (e) {
    btn.textContent = '扫描并生成候选词';
    document.getElementById('trainHint').textContent = '导出失败：' + e.message;
  } finally { btn.disabled = false; }
}

async function trainRunModel() {
  const btn = document.getElementById('btnTrainRun');
  btn.disabled = true; btn.textContent = '训练中…';
  try {
    const res = await API.Train.train();
    document.getElementById('trainHint').textContent = res.ok
      ? `训练完成：样本 ${res.metrics.n_samples}，验证 Spearman ${res.metrics.val_spearman}。刷新文章分析页即可生效。`
      : (res.error || '训练失败');
    await loadTrainPage();
  } catch (e) {
    document.getElementById('trainHint').textContent = '训练失败：' + e.message;
  } finally { btn.disabled = false; btn.textContent = '用标注训练模型'; }
}

async function trainRunEval() {
  const btn = document.getElementById('btnTrainEval');
  const box = document.getElementById('trainEvalResult');
  const tip = document.getElementById('trainEvalTip');
  btn.disabled = true;
  box.classList.add('hidden');

  // 读题划词模式：评估划词模型（0/1/2 该不该划），而非文章关键词提取
  if (trainState.mode === 'huaci') {
    btn.textContent = '评估中…';
    tip.textContent = `正在「${trainState.subj}」划词模型上做留出评估，请稍候…`;
    try {
      const res = await API.Train.huaci.eval(trainState.subj);
      if (!res.ok) { tip.textContent = res.error || '评估失败'; return; }
      renderHuaciEval(res, box);
      tip.textContent = '评估完成。';
    } catch (e) {
      tip.textContent = '评估失败：' + e.message;
    } finally { btn.disabled = false; btn.textContent = '一键评估效果'; }
    return;
  }

  // 文章关键词模式：用独立测试集评估提取效果
  btn.textContent = '评估中…（约 1 分钟）';
  tip.textContent = '正在 27 篇独立测试集上提取关键词并评估，请稍候…';
  try {
    const res = await API.Train.eval();
    if (!res.ok) { tip.textContent = res.error || '评估失败'; return; }
    renderTrainEval(res, box);
    tip.textContent = '评估完成。';
  } catch (e) {
    tip.textContent = '评估失败：' + e.message;
  } finally { btn.disabled = false; btn.textContent = '一键评估效果'; }
}

// 读题划词模型的"一键评估"结果面板（区分于 renderTrainEval）
function renderHuaciEval(res, box) {
  const p = res.per_class || {};
  const row = (name, d) => `<tr>
    <td>${name}</td>
    <td>${pct(d.precision)}</td>
    <td>${pct(d.recall)}</td>
    <td>${(d['f1-score']).toFixed(2)}</td>
    <td>${d.support}</td></tr>`;
  const featRows = (res.top_features || []).map(([f, v]) => `<tr><td>${esc(f)}</td><td>${v.toFixed(3)}</td></tr>`).join('');
  box.innerHTML = `
    <div class="eval-summary">学科 <b>${esc(res.subject)}</b> · 样本 <b>${res.n_samples}</b> 条 · 留出评估 <b>${res.test_size}</b> 条（不参与训练）· 准确率 <b>${pct(res.accuracy)}</b> · F1(weighted) <b>${(res.f1_weighted).toFixed(2)}</b> · F1(macro) <b>${(res.f1_macro).toFixed(2)}</b></div>
    <table class="eval-table">
      <thead><tr><th>标签</th><th>精确率</th><th>召回率</th><th>F1</th><th>样本</th></tr></thead>
      <tbody>${row('不划(0)', p['不划'] || {precision:0,recall:0,'f1-score':0,support:0})}${row('可划(1)', p['可划'] || {precision:0,recall:0,'f1-score':0,support:0})}${row('必划(2)', p['必划'] || {precision:0,recall:0,'f1-score':0,support:0})}</tbody>
    </table>
    <details class="eval-detail"><summary>最重要特征</summary>
      <table class="eval-table eval-table-detail"><thead><tr><th>特征</th><th>重要度</th></tr></thead><tbody>${featRows || '<tr><td colspan="2">—</td></tr>'}</tbody></table>
    </details>
    <div class="eval-verdict">${esc(res.hint || '')}</div>`;
  box.classList.remove('hidden');
}

function pct(v) { return (v * 100).toFixed(0) + '%'; }

function renderTrainEval(res, box) {
  const s = res.summary || {};
  const m = res.model || {}, r = res.rule || {}, c = res.conclusion || {};
  const row = (label, d) => `<tr>
    <td>${label}</td>
    <td>${pct(d.hit1)}</td>
    <td>${pct(d.any_hit)}</td>
    <td>${(d.p8_gold).toFixed(2)}</td>
    <td>${(d.p8_rel).toFixed(2)}</td>
    <td>${(d.recall).toFixed(2)}</td></tr>`;
  const mm = s.model_metrics || {};
  const detailRows = (res.details || []).map((d) => `<tr>
    <td title="${esc(d.gold.join(' / '))}">${esc(d.title)}</td>
    <td>${esc(d.hit_gold.join('、') || '—')}</td>
    <td>${d.hit_gold.length}/${d.n_gold}</td></tr>`).join('');
  box.innerHTML = `
    <div class="eval-summary">测试集 ${s.test_size} 篇（科普 ${s.n_sci} / 文学 ${s.n_lit}）· 临时模型样本 ${mm.n_samples} · Spearman ${mm.val_spearman} · 档位命中 ${pct(mm.val_class_acc)}</div>
    <table class="eval-table">
      <thead><tr><th>版本</th><th>首名命中</th><th>篇级命中</th><th>P@8 关键</th><th>P@8 相关</th><th>Gold召回</th></tr></thead>
      <tbody>${row('重排模型版', m)}${row('规则版基线', r)}</tbody>
    </table>
    <div class="eval-verdict"><b>判断：</b>${esc(c.verdict || '')}<br/><b>对比：</b>${esc(c.compare || '')}</div>
    <details class="eval-detail"><summary>查看每篇命中明细</summary>
      <table class="eval-table eval-table-detail"><thead><tr><th>篇目</th><th>命中关键中心词</th><th>命中/金标</th></tr></thead><tbody>${detailRows}</tbody></table>
    </details>`;
  box.classList.remove('hidden');
}

async function trainAddArticle() {
  const title = document.getElementById('inputArtTitle').value;
  const text = document.getElementById('inputArtText').value;
  if (!text.trim()) return;
  const res = await API.Train.addArticle(title, text);
  if (res && res.error) { document.getElementById('trainHint').textContent = res.error; return; }
  document.getElementById('inputArtTitle').value = '';
  document.getElementById('inputArtText').value = '';
  document.getElementById('trainHint').textContent = '已添加文章，点击「扫描并生成候选词」后即可标注。';
  await loadTrainPage();
}

async function trainImportFolder() {
  const btn = document.getElementById('btnImportFolder');
  btn.disabled = true; btn.textContent = '导入中…';
  try {
    const res = await API.Train.importFolder();
    document.getElementById('trainHint').textContent = res.ok
      ? `从 ${res.path} 导入 ${res.imported} 篇${res.skipped ? `，跳过 ${res.skipped} 篇（重复或过短）` : ''}。点「扫描并生成候选词」后即可标注。`
      : (res.error || '导入失败');
    await loadTrainPage();
  } catch (e) {
    document.getElementById('trainHint').textContent = '导入失败：' + e.message;
  } finally { btn.disabled = false; btn.textContent = '导入 data/texts/ 文件夹'; }
}

async function trainImportFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const btn = document.getElementById('btnImportFiles');
  btn.disabled = true; btn.textContent = '导入中…';
  try {
    const res = await API.Train.importFiles(files);
    document.getElementById('trainHint').textContent = res.ok
      ? `已导入 ${res.imported} 个 TXT${res.skipped ? `，跳过 ${res.skipped} 个` : ''}。点「扫描并生成候选词」后即可标注。`
      : (res.error || '导入失败');
    document.getElementById('inputImportTxt').value = '';
    await loadTrainPage();
  } catch (e) {
    document.getElementById('trainHint').textContent = '导入失败：' + e.message;
  } finally { btn.disabled = false; btn.textContent = '选择 TXT 多选导入'; }
}

function trainNav(delta) {
  const arr = trainState.items;
  if (!arr.length) return;
  trainState.idx = (trainState.idx + delta + arr.length) % arr.length;
  trainShowDoc();
}

// LaTeX 渲染辅助（读题划词的公式与公式类候选）
function katexHtml(expr) {
  try {
    // 内联渲染：让公式与前后文字同排，避免 LaTeX 用 display block 把一个字母撑成整行
    return window.katex.renderToString(expr, { throwOnError: false, displayMode: false });
  } catch (e) { /* 渲染失败回退原文 */ }
  return esc(expr);
}
// 把含 $...$ 的文本按"普通文字 + 公式块"交错渲染
function renderMathText(t) {
  const parts = [];
  const re = /\$([^$]+)\$/g;
  let last = 0, m;
  while ((m = re.exec(t)) !== null) {
    parts.push(esc(t.slice(last, m.index)));
    parts.push(katexHtml(m[1]));
    last = m.index + m[0].length;
  }
  parts.push(esc(t.slice(last)));
  return parts.join('');
}
// 候选词若是整块 $...$ 公式则渲染，否则按普通词转义
function renderCandKw(kw) {
  return (kw.startsWith('$') && kw.endsWith('$') && kw.length > 2) ? katexHtml(kw.slice(1, -1)) : esc(kw);
}

function trainShowDoc() {
  const info = document.getElementById('trainNavInfo');
  const docEl = document.getElementById('trainDoc');
  const candEl = document.getElementById('trainCands');
  const arr = trainState.items;
  if (!arr.length) {
    info.textContent = '0 / 0';
    docEl.textContent = trainState.mode === 'huaci'
      ? '该学科暂无题目可标注，请先确认 data/huaci 下已生成题源。'
      : '暂无文章，请先添加文章并生成候选词。';
    docEl.classList.toggle('picking', false);
    candEl.innerHTML = '';
    trainState.current = null;
    return;
  }
  const it = arr[trainState.idx];
  trainState.current = it;
  info.textContent = `${trainState.idx + 1} / ${arr.length}`;
  if (trainState.mode === 'huaci') {
    // 题干 + 选项逐块渲染（LaTeX 公式可见，便于标注复核）
    const blocks = [it.question];
    (it.options || []).forEach((o, i) => { blocks.push(`${'ABCD'[i] || (i + 1)}. ${o}`); });
    docEl.innerHTML = blocks.map((b) => `<div style="margin:6px 0">${renderMathText(b)}</div>`).join('');
    renderTrainCands(it.candidates || [], true);
  } else {
    docEl.textContent = `${it.title}${it.text ? '\n\n' + it.text : ''}`;
    renderTrainCands(it.candidates || [], it.exported);
  }
}

function trainDocPick(e) {
  if (!trainState.picking || !trainState.current) return;
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const text = sel.toString().trim();
  if (!text) return;
  const range = sel.getRangeAt(0);
  const doc = document.getElementById('trainDoc');
  // 只处理在正文内的选择，且长度/字符合规
  if (!range || !doc.contains(range.commonAncestorContainer)) return;
  if (!/^[\u4e00-\u9fa5A-Za-z0-9+\-#·]{2,16}$/.test(text)) { sel.removeAllRanges(); return; }
  sel.removeAllRanges();
  trainAddPick(text);
}

function trainAddPick(word) {
  const c = trainState.current;
  if (!c.candidates) c.candidates = [];
  const exists = c.candidates.some((x) => x.kw === word);
  if (!exists) {
    if (trainState.mode === 'huaci') {
      // 划词题：新划词按概念类入池（读题时手动划取的缺失词）
      c.candidates.push({
        kw: word, basis: '划词标记', type: 'concept', length: word.length, freq: 1,
        first_pos_norm: 1.0, in_options: 0, is_formula: 0, is_number: 0, has_unit: 0,
        label: 2, note: '', user_pick: true,
      });
    } else {
      c.candidates.push({
        kw: word, basis: '划词标记', frequency: 0, coverage: 0, stat_score: 0,
        stat_norm: 0, semantic: 0, length: word.length, is_subsumed: 0,
        rule_conf: 0, label: 2, note: '', user_pick: true,
      });
    }
  } else {
    // 已经存在：把它的 label 默认推成关键 2（若还没标）
    const one = c.candidates.find((x) => x.kw === word);
    if (one.label === null || one.label === undefined) one.label = 2;
  }
  renderTrainCands(c.candidates, true);
}

function trainDeletePick(i) {
  const c = trainState.current;
  if (!c || !c.candidates) return;
  c.candidates.splice(i, 1);
  renderTrainCands(c.candidates, true);
}

function renderTrainCands(cands, exported) {
  const el = document.getElementById('trainCands');
  const TYPE_CN = { data: '数据', logic: '逻辑', concept: '概念' };
  if (!exported) { el.innerHTML = '<div style="font-size:0.78rem;color:var(--mist)">该文章尚未生成候选词，请先执行「扫描并生成候选词」。</div>'; return; }
  if (!cands.length) { el.innerHTML = '<div style="font-size:0.78rem;color:var(--mist)">未提取到候选词。</div>'; return; }
  el.innerHTML = cands.map((c, i) => {
    const meta = trainState.mode === 'huaci'
      ? `${TYPE_CN[c.type] || c.type || ''} · 频${c.freq ?? '-'}${c.has_unit ? ' · 单位' : ''}${c.is_formula ? ' · 公式' : ''}${c.in_options ? ' · 选项' : ''}${c.basis ? ' · ' + c.basis : ''}`
      : `频${c.frequency ?? '-'} · 语义${c.semantic ?? '-'} · 规则分${c.rule_conf ?? '-'}${c.basis ? ' · ' + c.basis : ''}`;
    return `
    <div class="train-cand" data-i="${i}">
      <div class="train-cand-main">
        <div class="train-cand-word">${renderCandKw(c.kw)}
          ${c.basis === '划词标记' ? '<span class="tag-user">划词</span>' : ''}
          <span class="meta">${meta}</span>
        </div>
        <input class="train-cand-note" data-note placeholder="备注（可选）" value="${esc(c.note || '')}" />
      </div>
      <div class="train-rating">
        <button data-v="0" class="r${c.label === 0 ? ' on-0' : ''}">0</button>
        <button data-v="1" class="r${c.label === 1 ? ' on-1' : ''}">1</button>
        <button data-v="2" class="r${c.label === 2 ? ' on-2' : ''}">2</button>
        ${c.basis === '划词标记' ? `<button class="cand-del" data-del title="删除该划词" aria-label="删除">×</button>` : ''}
      </div>
    </div>`}).join('');
  el.querySelectorAll('.train-rating').forEach((box) => {
    box.addEventListener('click', (e) => {
      const btn = e.target.closest('button'); if (!btn) return;
      if (btn.dataset.del !== undefined) {
        const row = box.closest('.train-cand');
        trainDeletePick(+row.dataset.i);
        return;
      }
      const row = box.closest('.train-cand');
      const i = +row.dataset.i;
      const v = +btn.dataset.v;
      const c = trainState.current.candidates[i];
      if (c.label === v) c.label = null; else c.label = v;
      row.querySelectorAll('button[data-v]').forEach((b) => b.className = 'r');
      if (c.label === 0) btn.className = 'r on-0';
      else if (c.label === 1) btn.className = 'r on-1';
      else if (c.label === 2) btn.className = 'r on-2';
    });
  });
  el.querySelectorAll('.train-cand-note[data-note]').forEach((inp) => {
    inp.addEventListener('input', (e) => {
      const row = inp.closest('.train-cand');
      const i = +row.dataset.i;
      trainState.current.candidates[i].note = inp.value;
    });
  });
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

async function trainSave() {
  const it = trainState.current;
  if (!it || !it.candidates) return;
  const res = trainState.mode === 'huaci'
    ? await API.Train.huaci.saveLabel(trainState.subj, it.qid, it.candidates)
    : await API.Train.saveLabel(it.doc_id, it.candidates);
  document.getElementById('trainHint').textContent = res.ok ? '已保存标注。' : (res.error || '保存失败');
  await loadTrainPage();
}

// ---------- 学习总览 ----------
async function loadDashboard() {
  const [boss, qRes] = await Promise.all([API.Boss.status(), API.Questions.list()]);
  questions = qRes.questions || [];
  renderBoss(boss);
  renderOverviewStats(questions, boss);
  renderDashClusters();
  renderDashMastery();
  renderDashConfusion();
  renderDashGraph();
  loadInsights();
}

// 概览条：综合健康度 + 关键统计
function renderOverviewStats(questions, boss) {
  const total = questions.length;
  const wrong = questions.filter((q) => q.is_wrong).length;
  const mastered = questions.filter((q) => q.is_mastered).length;
  const mastery = total ? Math.round((mastered / total) * 100) : 0;
  const stats = [
    { label: '题目', value: total, unit: '道' },
    { label: '错题待复盘', value: wrong, unit: '道', accent: wrong > 0 },
    { label: '掌握率', value: `${mastery}%`, unit: '' },
    { label: '连续答对', value: boss.consecutive_correct || 0, unit: '次' },
  ];
  const el = document.getElementById('oStats');
  el.innerHTML = stats.map((s) => `
    <div class="ov-stat${s.accent ? ' accent' : ''}">
      <div class="ov-stat-value">${s.value}<span class="ov-stat-unit">${s.unit}</span></div>
      <div class="ov-stat-label">${s.label}</div>
    </div>`).join('');
}

// ---------- 认知诊断 + 模型栈 ----------
async function loadInsights() {
  let res;
  try { res = await API.Monitor.insights(); } catch (e) { return; }
  renderHealth(res);
  renderDifficulty(res.difficulty);
  renderOverconfidence(res.overconfidence);
  renderClusters(res.error_clusters);
  renderModelStack(res.models);
}

function renderHealth(d) {
  const health = d.health != null ? Math.round(d.health) : 0;
  const cls = health >= 70 ? 'accent' : health >= 40 ? 'warn' : 'danger';
  // 概览条主环（紧凑）
  ring(document.getElementById('oHealthRing'), health, { size: 96, stroke: 9, cls, label: '健康度' });
  // 诊断导航段环
  ring(document.getElementById('dHealthRing'), health, { size: 84, stroke: 8, cls, label: '健康度' });
  const tag = document.getElementById('oHealthTag');
  tag.textContent = health >= 70 ? '状态良好' : health >= 40 ? '需要关注' : '存在风险';
  tag.className = 'overview-hero-tag ' + cls;
  const note = document.getElementById('oHealthNote');
  note.textContent = `未掌握率 ${Math.round((d.wrong_rate || 0) * 100)}% · 掌握率 ${Math.round((d.mastery_rate || 0) * 100)}%`;
  const dnote = document.getElementById('dHealthNote');
  dnote.textContent = `未掌握率 ${Math.round((d.wrong_rate || 0) * 100)}% · 掌握率 ${Math.round((d.mastery_rate || 0) * 100)}%`;
}

function renderDifficulty(d) {
  const valEl = document.getElementById('dDifficulty');
  const bar = document.getElementById('dDifficultyBar');
  const mean = d.mean;
  if (mean == null) { valEl.textContent = '--'; bar.textContent = '模型未训练'; return; }
  valEl.textContent = Math.round(mean * 100);
  const level = d.level || '适中';
  const active = Math.max(1, Math.min(5, Math.ceil(mean * 5)));
  let segs = '';
  for (let i = 1; i <= 5; i++) segs += `<span class="diff-seg ${i <= active ? 'on' : ''}"></span>`;
  bar.innerHTML = `<span class="diff-level">${level}</span><div class="diff-scale">${segs}</div>`;
}

function renderOverconfidence(o) {
  const el = document.getElementById('dOverconf');
  el.textContent = o && o.total ? `${o.risk_count} / ${o.total}` : '--';
  el.style.color = (o && o.warn) ? 'var(--danger)' : (o && o.total) ? 'var(--accent)' : '';
  document.getElementById('dOverconfNote').textContent =
    (o && o.warn) ? '存在过度自信错题，建议复盘' : (o && o.total) ? '风险评估正常' : '暂无错题样本';
}

function renderClusters(clusters) {
  const el = document.getElementById('dClusters');
  if (!clusters || !clusters.length) { el.textContent = '错题不足，暂未聚类'; return; }
  el.innerHTML = '';
  clusters.forEach((c) => {
    const row = document.createElement('div');
    row.className = 'diag-cluster-row';
    row.innerHTML = `<span>${c.label}</span><div class="mini-bar"><div class="mini-bar-fill" style="width:${Math.round(c.percentage * 100)}%"></div></div><span>${Math.round(c.percentage * 100)}%</span>`;
    el.appendChild(row);
  });
}

function renderModelStack(models) {
  const el = document.getElementById('stackList');
  if (!models || !models.length) { el.textContent = '暂无模型信息'; return; }
  el.innerHTML = '';
  models.forEach((m) => {
    const chip = document.createElement('span');
    chip.className = 'stack-chip ' + (m.ready ? 'ready' : 'pending');
    chip.innerHTML = `${m.name}<i>${m.ready ? '就绪' : '待训练'}</i>`;
    el.appendChild(chip);
  });
}

// 机器学习参数图表：参数量对比 + 架构类型分布
function renderMlCharts(models) {
  // 参数量对比（万为单位，ResNet 显著偏大）
  const params = models.map((m) => ({
    label: m.name,
    value: +(m.params / 10000).toFixed(2),
  })).sort((a, b) => b.value - a.value);
  const pEl = document.getElementById('mlParamsChart');
  pEl.innerHTML = '';
  hbar(pEl, params, { maxValue: 0, unit: '万' });

  // 架构类型分布
  const typeCount = {};
  models.forEach((m) => { typeCount[m.type || '其他'] = (typeCount[m.type || '其他'] || 0) + 1; });
  donut(document.getElementById('mlTypeChart'),
    Object.entries(typeCount).map(([label, value]) => ({ label, value })),
    { legend: 'pct' });
}

// ---------- 机器学习参数详情（隐藏二级菜单） ----------
function renderMlDetail(models) {
  const grid = document.getElementById('mlDetailGrid');
  if (!models || !models.length) { grid.textContent = '暂无模型参数'; return; }
  grid.innerHTML = '';
  models.forEach((m) => {
    const card = document.createElement('div');
    card.className = 'ml-card';
    card.innerHTML = `
      <div class="ml-card-head">
        <span class="ml-card-name">${m.name}</span>
        <span class="ml-card-state ${m.ready ? 'ready' : 'pending'}">${m.ready ? '已就绪' : '待训练'}</span>
      </div>
      <div class="ml-card-arch">${m.arch}</div>
      <div class="ml-card-row"><span>输入/输出</span><span>${m.io}</span></div>
      <div class="ml-card-row"><span>关键参数</span><span>${m.detail}</span></div>
    `;
    grid.appendChild(card);
  });
}

function initMlDetail() {
  document.getElementById('mlDetailToggle').addEventListener('click', () => {
    const body = document.getElementById('mlDetailBody');
    const caret = document.getElementById('mlDetailCaret');
    const open = body.classList.toggle('hidden');
    caret.textContent = open ? '▸' : '▾';
  });
}

function renderBoss(boss) {
  document.getElementById('bossLevel').textContent = `Lv.${boss.difficulty || 1}`;
  document.getElementById('bossHp').style.width = `${Math.max(0, Math.min(100, boss.hp || 100))}%`;
  document.getElementById('bossDiff').textContent = `难度：${boss.difficulty || 1} / 5`;
  document.getElementById('bossStreak').textContent = boss.consecutive_correct || 0;
  const dqnEl = document.getElementById('bossDqn');
  if (dqnEl) {
    const state = (boss.dqn_state || []).map((x) => Number(x).toFixed(2)).join(', ');
    dqnEl.textContent = `DQN 决策：${boss.dqn_action_label || '保持'} · 状态[${state}]`;
  }
  // 难度档位 pips（游戏化：点亮当前档位）
  const pipsEl = document.getElementById('bossPips');
  pipsEl.innerHTML = '';
  const diff = boss.difficulty || 1;
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement('span');
    pip.className = 'boss-pip' + (i <= diff ? ' on' : '');
    pipsEl.appendChild(pip);
  }
}

async function renderDashClusters() {
  const el = document.getElementById('dashClusterChart');
  let res;
  try { res = await API.Analysis.clusters(); } catch (e) { el.textContent = '数据加载失败'; return; }
  const clusters = res.clusters || [];
  if (!clusters.length) { el.innerHTML = '<div class="chart-empty">题目不足 5 道</div>'; return; }
  donut(el, clusters.map((c) => ({ label: c.cluster_label, value: c.count || Math.round(c.percentage * 100) })));
}

async function renderDashMastery() {
  const el = document.getElementById('dashMasteryChart');
  const total = questions.length;
  if (!total) { el.innerHTML = '<div class="chart-empty">暂无题目</div>'; return; }
  const wrong = questions.filter((q) => q.is_wrong).length;
  const mastered = questions.filter((q) => q.is_mastered).length;
  const other = Math.max(0, total - wrong - mastered);
  donut(el, [
    { label: '已掌握', value: mastered },
    { label: '错题', value: wrong },
    { label: '学习中', value: other },
  ]);
}

async function renderDashConfusion() {
  const el = document.getElementById('dashConfusion');
  let res;
  try { res = await API.Analysis.confusion(); } catch (e) { el.textContent = '数据加载失败'; return; }
  const pairs = res.pairs || [];
  if (!pairs.length) { el.textContent = '暂无易混淆概念'; el.className = 'cluster-list empty'; return; }
  el.className = 'cluster-list';
  el.innerHTML = '';
  pairs.slice(0, 4).forEach(([a, b, sim]) => {
    const item = document.createElement('div');
    item.className = 'cluster-item';
    item.innerHTML = `<div class="cluster-head"><span class="cluster-label">${a} ↔ ${b}</span><span class="cluster-pct">${sim}</span></div>`;
    el.appendChild(item);
  });
}

async function renderDashGraph() {
  const el = document.getElementById('dashGraph');
  let res;
  try { res = await API.Analysis.graph(); } catch (e) { el.textContent = '模型未训练'; return; }
  if (!res.ok) { el.textContent = res.error || '模型未训练'; return; }
  el.textContent = '';
  bipartite(el, { left: res.left_nodes, right: res.right_nodes, matrix: res.matrix, height: 220 });
}

// ---------- 实时监控反馈（状态灯） ----------
let monitorTimer = null;

function startMonitor() {
  stopMonitor();
  refreshMonitor();
  monitorTimer = setInterval(refreshMonitor, 2000);
}

function stopMonitor() {
  if (monitorTimer) { clearInterval(monitorTimer); monitorTimer = null; }
}

async function refreshMonitor() {
  let res;
  try { res = await API.Monitor.status(); } catch (e) { return; }
  renderMouseMonitor(res.mouse);
  renderScreenMonitor(res.screen);
}

function renderMouseMonitor(m) {
  const statusEl = document.getElementById('mMouseStatus');
  const barsEl = document.getElementById('mMouseBars');
  const noteEl = document.getElementById('mMouseNote');
  const confEl = document.getElementById('mMouseConf');
  noteEl.textContent = m.note || '';
  // 未开启：明确提示用户去采集模块开启
  if (!m.enabled) {
    statusEl.textContent = '未开启';
    statusEl.className = 'monitor-status off';
    barsEl.innerHTML = '';
    confEl.textContent = '';
    return;
  }
  // 已开启但模型未训练
  if (!m.model_ready) {
    statusEl.textContent = '待训练';
    statusEl.className = 'monitor-status warn';
    barsEl.innerHTML = '';
    confEl.textContent = '';
    return;
  }
  // 已开启且模型就绪，数据尚未积累足够：立即反馈"采集中"
  if (!m.status) {
    statusEl.textContent = '采集中';
    statusEl.className = 'monitor-status collecting';
    barsEl.innerHTML = '';
    confEl.textContent = '积累轨迹中…（需 50 帧）';
    return;
  }
  const cls = m.status === '卡壳' ? 'danger' : m.status === '犹豫' ? 'warn' : 'good';
  statusEl.textContent = m.status;
  statusEl.className = `monitor-status ${cls}`;
  const labels = ['流畅', '犹豫', '卡壳'];
  const probs = m.probs || [0, 0, 0];
  // 综合置信度：最大概率 + 对应类别
  const maxP = Math.max(...probs);
  const maxIdx = probs.indexOf(maxP);
  const confCls = maxP >= 0.7 ? 'high' : maxP >= 0.5 ? 'mid' : 'low';
  confEl.textContent = `置信度 ${Math.round(maxP * 100)}%（${labels[maxIdx]}）`;
  confEl.className = `monitor-conf ${confCls}`;
  barsEl.innerHTML = labels.map((label, i) => {
    const pct = Math.round((probs[i] || 0) * 100);
    const active = label === m.status ? ' active' : '';
    return `
      <div class="monitor-bar">
        <span class="monitor-bar-label">${label}</span>
        <div class="monitor-bar-track"><div class="monitor-bar-fill${active}" style="width:${pct}%"></div></div>
        <span class="monitor-bar-val">${pct}%</span>
      </div>`;
  }).join('');
}

function renderScreenMonitor(s) {
  const statusEl = document.getElementById('mScreenStatus');
  const simEl = document.getElementById('mScreenSim');
  const curveEl = document.getElementById('mScreenCurve');
  const noteEl = document.getElementById('mScreenNote');
  noteEl.textContent = s.note || '';
  // 未开启：明确提示用户去采集模块开启
  if (!s.enabled) {
    statusEl.textContent = '未开启';
    statusEl.className = 'monitor-status off';
    simEl.textContent = '—';
    curveEl.innerHTML = '';
    return;
  }
  // 已开启但模型未训练
  if (!s.model_ready) {
    statusEl.textContent = '待训练';
    statusEl.className = 'monitor-status warn';
    simEl.textContent = '—';
    curveEl.innerHTML = '';
    return;
  }
  // 已开启且模型就绪，截图尚未积累足够：立即反馈"采集中"
  if (!s.attention) {
    statusEl.textContent = '采集中';
    statusEl.className = 'monitor-status collecting';
    simEl.textContent = '—';
    curveEl.innerHTML = '';
    return;
  }
  const cls = s.attention === '漂移' ? 'danger' : 'good';
  statusEl.textContent = s.attention;
  statusEl.className = `monitor-status ${cls}`;
  simEl.textContent = s.similarity != null ? s.similarity.toFixed(3) : '—';
  // 注意力曲线：把历史相似度画成 SVG 折线，直观体现专注/漂移趋势
  const hist = s.history || [];
  curveEl.innerHTML = hist.length >= 2 ? renderAttentionCurve(hist) : '';
}

// 注意力曲线：history 相似度 → SVG 折线（含 0.8 专注阈值参考线）
function renderAttentionCurve(hist) {
  const w = 260, h = 46, pad = 4;
  const n = hist.length;
  const pts = hist.map((v, i) => {
    const x = pad + (i / (n - 1)) * (w - 2 * pad);
    const y = h - pad - (Math.min(1, Math.max(0, v)) * (h - 2 * pad));
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const ty = (h - pad - (0.8 * (h - 2 * pad))).toFixed(1);
  const last = hist[hist.length - 1];
  return `
    <svg class="sim-curve-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="0" y1="${ty}" x2="${w}" y2="${ty}" class="sim-threshold"/>
      <polyline points="${pts}" class="sim-polyline"/>
      <circle cx="${(pad + ((n - 1) / (n - 1)) * (w - 2 * pad)).toFixed(1)}" cy="${(h - pad - (Math.min(1, Math.max(0, last)) * (h - 2 * pad))).toFixed(1)}" r="2.5" class="sim-dot"/>
    </svg>`;
}

// ---------- 添加题目 ----------
function initEntry() {
  // 录入方式按钮：手动表单（新增）由 initEditQuestionControls 统一接管（含退出编辑态）
  // 上传识别：选择已有图片文件
  document.getElementById('btnOcrUpload').addEventListener('click', pickQuestionImage);
  // 拍照识别：调用本地摄像头
  document.getElementById('btnPhotoCapture').addEventListener('click', openPhotoCapture);
  // 框选识别：鼠标框选屏幕区域
  document.getElementById('btnOcrSelect').addEventListener('click', openScreenSelect);

  document.getElementById('questionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      question_text: document.getElementById('inputQuestion').value.trim(),
      options: document.getElementById('inputOptions').value.split('\n').map((s) => s.trim()).filter(Boolean),
      correct_answer: document.getElementById('inputCorrect').value.trim(),
      knowledge_tags: document.getElementById('inputTags').value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
      keywords: document.getElementById('inputKeywords').value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
      is_wrong: document.getElementById('inputWrong').checked,
    };
    if (editingQuestionId) payload.question_id = editingQuestionId;
    if (!payload.question_text || !payload.correct_answer) {
      showModal('提示', '题干与正确答案为必填项');
      return;
    }
    await API.Questions.add(payload);
    const wasEditing = !!editingQuestionId;
    cancelEditQuestion(false); // 回到新增态，并清空表单
    renderQuestionList();
    showModal('已保存', wasEditing ? '题目已更新' : '题目已加入题库');
  });

  // 一键划词：按当前题干+选项生成"该划关键词"，填进关键词框供人工确认/增删
  document.getElementById('btnAutoKeywords').addEventListener('click', async () => {
    const qtext = document.getElementById('inputQuestion').value.trim();
    if (!qtext) { showModal('提示', '请先填写题干再一键划词'); return; }
    const options = document.getElementById('inputOptions').value.split('\n').map((s) => s.trim()).filter(Boolean);
    const btn = document.getElementById('btnAutoKeywords');
    const old = btn.textContent;
    btn.disabled = true; btn.textContent = '生成中…';
    try {
      const r = await API.Questions.suggestKeywords({ question_text: qtext, options });
      const kws = r.keywords || [];
      document.getElementById('inputKeywords').value = kws.join('，');
      showModal('一键划词', kws.length ? `已生成 ${kws.length} 个关键词，可增删后保存：${kws.join('、')}` : '未生成关键词，可手填');
    } catch (e) {
      showModal('提示', '一键划词失败，可手填关键词');
    } finally {
      btn.disabled = false; btn.textContent = old;
    }
  });
}

// ---------- 编辑已保存题目：把题目载回表单（公式以富文本渲染），改后覆盖保存 ----------
let editingQuestionId = null;

function enterEditQuestion(q) {
  editingQuestionId = q.question_id;
  document.getElementById('inputQuestion').value = mathifyText(q.question_text || '');
  document.getElementById('inputOptions').value = (q.options || []).map((o) => mathifyText(o)).join('\n');
  document.getElementById('inputCorrect').value = q.correct_answer || '';
  document.getElementById('inputTags').value = (q.knowledge_tags || []).join('，');
  document.getElementById('inputKeywords').value = (q.keywords || []).join('，');
  document.getElementById('inputWrong').checked = !!q.is_wrong;
  rtfSyncAll();
  const form = document.getElementById('questionForm');
  form.classList.remove('hidden');
  const banner = document.getElementById('editQuestionBanner');
  banner.classList.remove('hidden');
  const short = (q.question_text || '题目').slice(0, 18);
  document.getElementById('editQuestionLabel').textContent = `正在编辑：${short}${short.length >= 18 ? '…' : ''}（点以下「保存题目」覆盖）`;
  showView('question-add');
  form.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function cancelEditQuestion(resetForm) {
  editingQuestionId = null;
  document.getElementById('editQuestionBanner').classList.add('hidden');
  if (resetForm) {
    document.getElementById('questionForm').reset();
    rtfSyncAll();
  }
}

function initEditQuestionControls() {
  document.getElementById('cancelEditQuestion').addEventListener('click', () => {
    cancelEditQuestion(true);
    // 取消编辑后回到题目列表，让已保存内容保持可见（避免停留在被清空的表单上误以为丢失）
    showView('question-bank');
  });
  // 点「新增题目」时退出编辑态，避免把新题覆盖到旧题上
  document.getElementById('btnManualForm').addEventListener('click', () => {
    cancelEditQuestion(true);
    document.getElementById('questionForm').classList.remove('hidden');
    document.getElementById('inputQuestion').focus();
  });
}

// OCR 结果填入后：滚动到表单并高亮，提示用户去核对修正（避免结果落在屏外被忽略）
function focusAddForm() {
  const form = document.getElementById('questionForm');
  form.classList.remove('hidden');
  form.scrollIntoView({ behavior: 'smooth', block: 'center' });
  form.classList.add('ocr-flash');
  setTimeout(() => form.classList.remove('ocr-flash'), 1500);
}

// 把图片交给 OCR 后端识别，结果填入表单（带并发保护，避免重复识别）
let ocrBusy = false;
async function recognizeImage(blob) {
  if (ocrBusy) return;
  ocrBusy = true;
  showModal('OCR 识别', '正在识别，请稍候…');
  try {
    const res = await API.Questions.ocr(blob);
    if (!res.ok) { showModal('OCR 提示', res.error || '识别失败，可手动录入'); return false; }
    document.getElementById('inputQuestion').value = mathifyText(res.question_text || '');
    document.getElementById('inputOptions').value = (res.options || []).map((o) => mathifyText(o)).join('\n');
    rtfSyncAll();
    focusAddForm();
    showModal('OCR 完成', '识别结果已填入表单，请核对修正后保存');
    return true;
  } catch (e) {
    showModal('OCR 提示', '识别请求失败，可手动录入');
    return false;
  } finally {
    ocrBusy = false;
  }
}

// 上传识别：选择已有图片文件
function pickQuestionImage() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    await recognizeImage(file);
  };
  input.click();
}

// ---------- 添加题目：可视化公式编辑器（点选结构/符号，全程不显示 LaTeX 代码） ----------
const MATH_FIELDS = {
  inputQuestion: '题干',
  inputOptions: '选项',
  inputCorrect: '正确答案',
  inputTags: '知识点',
};
let mathTarget = 'inputQuestion';

// 叶子：{ type:'txt'|'sym', v }；结构盒：{ type:'box', kind, slots:{槽名:[叶子] } }
// 用户只看到公式的“形状”，LaTeX 由模型在“插入”时才静默生成。
const VE = { root: [], rootIdx: 0, editLeaf: null, rtChipEl: null, rtSel: {} };
const VE_GLYPH = {
  '\\alpha':'α','\\beta':'β','\\theta':'θ','\\pi':'π','\\lambda':'λ','\\mu':'μ',
  '\\omega':'ω','\\partial':'∂','\\infty':'∞','\\times':'×','\\div':'÷','\\pm':'±',
  '\\leq':'≤','\\geq':'≥','\\neq':'≠','\\circ':'°','\\cdot':'·','\\Delta':'Δ','\\nabla':'∇',
};

const veBoL = (t, v) => ({ type: t, v });

function veLeafLatex(l) { return l.v; }
function veBoxLatex(b) {
  const L = (a) => a.map((x) => (x.type === 'box' ? veBoxLatex(x) : veLeafLatex(x))).join(' ');
  switch (b.kind) {
    case 'frac': return `\\frac{${L(b.slots.num)}}{${L(b.slots.den)}}`;
    case 'sqrt': return `\\sqrt{${L(b.slots.body)}}`;
    case 'sup': return `{${L(b.slots.base)}}^{${L(b.slots.exp)}}`;
    case 'sub': return `{${L(b.slots.base)}}_{${L(b.slots.idx)}}`;
    case 'sum': return `\\sum_{${L(b.slots.low)}}^{${L(b.slots.hi)}} ${L(b.slots.body)}`;
    case 'int': return `\\int_{${L(b.slots.low)}}^{${L(b.slots.hi)}} ${L(b.slots.body)}`;
    case 'grp': return `\\left(${L(b.slots.body)}\\right)`;
    default: return '';
  }
}
function veLatex() {
  return VE.root.map((t) => (t.type === 'box' ? veBoxLatex(t) : veLeafLatex(t))).join(' ');
}

// DFS 收集叶子，用于焦点跳转
function veCollect(arr) {
  const out = [];
  const rec = (t) => { if (t.type === 'box') for (const k of Object.keys(t.slots)) t.slots[k].forEach(rec); else out.push(t); };
  arr.forEach(rec);
  return out;
}
function veBoxLeavesOf(box) { return veCollect([box]); }

function veGlyphHtml(late) {
  if (!late) return null;
  if (window.katex) { try { return window.katex.renderToString(late, { throwOnError: false }); } catch (e) { /* 走回退 */ } }
  return esc(late);
}
function veSymText(v) { return VE_GLYPH[v] || (v[0] === '\\' ? veGlyphHtml(v) : v); }

function veFocusLeaf(leaf) { VE.editLeaf = leaf; veRender(); }
function veRender() {
  const c = document.getElementById('veCanvas');
  c.innerHTML = '';
  const kids = VE.root.slice();
  // 未编辑任何格子时，在插入位置渲染一个闪烁游标，提示下一个符号落点
  if (!VE.editLeaf) kids.splice(Math.min(VE.rootIdx, kids.length), 0, null);
  for (const t of kids) c.appendChild(t === null ? veCaretEl() : veEl(t));
  renderVePreview();
  if (VE.editLeaf) { const inp = c.querySelector('input.ve-inp'); if (inp) inp.focus(); }
}
function veCaretEl() { const s = document.createElement('span'); s.className = 've-caret'; return s; }
function renderVePreview() {
  const p = document.getElementById('vePreview');
  const latex = veLatex();
  if (!latex) { p.innerHTML = '<span class="math-preview-empty">空公式</span>'; return; }
  if (window.katex) { try { window.katex.render(latex, p, { throwOnError: false, displayMode: true }); return; } catch (e) { /* 回退 */ } }
  p.innerHTML = esc(latex);
}

function veLeafEl(l) {
  const box = document.createElement('span');
  box.className = 've-l';
  if (l === VE.editLeaf) {
    const inp = document.createElement('input');
    inp.className = 've-inp';
    inp.value = l.type === 'sym' ? (VE_GLYPH[l.v] || l.v) : l.v;
    inp.setAttribute('autocomplete', 'off');
    const reflow = () => { inp.style.width = Math.max(22, (inp.value.length * 11) + 8) + 'px'; inp.focus(); const p = inp.value.length; try { inp.setSelectionRange(p, p); } catch (_) {} };
    const commit = () => { VE.editLeaf = null; veRootCaretAfter(l); veRender(); };
    inp.addEventListener('input', () => {
      l.type = 'txt'; l.v = inp.value || ''; renderVePreview(); // 仅更新预览与宽度，避免光标跳动
      reflow();
    });
    inp.addEventListener('keydown', (ev) => {
      ev.stopPropagation();
      if (ev.key === 'Enter' || ev.key === 'Escape') { ev.preventDefault(); inp.blur(); return; }
      if (ev.key === 'Tab') { ev.preventDefault(); commit(); return; }
      // 运算符自动退出当前格子，落到基座层接着输（x^2+3 → 填完指数按 + 直接回到基座）
      if (ev.key.length === 1 && '+*/=-<>'.indexOf(ev.key) >= 0) {
        ev.preventDefault();
        VE.editLeaf = null; veRootCaretAfter(l); veRender();
        VE.root.splice(VE.rootIdx, 0, veBoL('txt', ev.key)); VE.rootIdx++;
        veRender();
        const c = document.getElementById('veCanvas'); if (c) c.focus();
        return;
      }
    });
    inp.addEventListener('blur', commit);
    box.appendChild(inp);
    setTimeout(reflow, 0);
    return box;
  }
  if (l.type === 'txt') {
    if (l.v === '') { box.className += ' ve-empty'; box.textContent = '▢'; }
    else box.innerHTML = veGlyphHtml(l.v);
  } else {
    box.className += ' ve-sym';
    box.innerHTML = veSymText(l.v);
  }
  box.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); veFocusLeaf(l); });
  return box;
}

function veSlotEl(arr) {
  const s = document.createElement('span'); s.className = 've-slot';
  arr.forEach((t) => s.appendChild(veEl(t)));
  return s;
}
function veScriptEl(cls, arr) {
  const s = document.createElement('span'); s.className = 've-script ve-' + cls;
  s.appendChild(veSlotEl(arr)); return s;
}
function veLimOp(low, hi, sym) {
  const c = document.createElement('span'); c.className = 've-limop';
  const up = document.createElement('span'); up.className = 've-lim-up'; hi.forEach((t) => up.appendChild(veEl(t)));
  const mid = document.createElement('span'); mid.className = 've-lim-mid'; mid.textContent = sym;
  const dn = document.createElement('span'); dn.className = 've-lim-dn'; low.forEach((t) => dn.appendChild(veEl(t)));
  c.appendChild(up); c.appendChild(mid); c.appendChild(dn);
  return c;
}
function veEl(t) {
  if (t.type !== 'box') return veLeafEl(t);
  const w = document.createElement('span'); w.className = 've-bx ve-' + t.kind;
  w.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest('.ve-l') || ev.target.closest('input')) return;
    const arr = veBoxLeavesOf(t); if (arr.length) { veFocusLeaf(arr[0]); ev.stopPropagation(); }
  });
  const s = t.slots;
  if (t.kind === 'frac') {
    const st = document.createElement('span'); st.className = 've-stack';
    st.appendChild(veSlotEl(s.num));
    const ln = document.createElement('span'); ln.className = 've-line'; st.appendChild(ln);
    st.appendChild(veSlotEl(s.den));
    w.appendChild(st);
  } else if (t.kind === 'sqrt') {
    const root = document.createElement('span'); root.className = 've-sqrt';
    const bar = document.createElement('span'); bar.className = 've-sqrtbar'; bar.textContent = '√';
    const body = veSlotEl(s.body); body.className = 've-sqrtbody';
    root.appendChild(bar); root.appendChild(body); w.appendChild(root);
  } else if (t.kind === 'sup' || t.kind === 'sub') {
    const base = veSlotEl(s.base);
    w.appendChild(base);
    w.appendChild(veScriptEl(t.kind === 'sup' ? 'exp' : 'sbt', t.kind === 'sup' ? s.exp : s.idx));
  } else if (t.kind === 'sum' || t.kind === 'int') {
    w.appendChild(veLimOp(s.low, s.hi, t.kind === 'sum' ? 'Σ' : '∫'));
    w.appendChild(veSlotEl(s.body));
  } else if (t.kind === 'grp') { w.appendChild(veSlotEl(s.body)); }
  return w;
}

function veMakeBox(kind) {
  const E = () => veBoL('txt', '');
  switch (kind) {
    case 'frac': return { type: 'box', kind, slots: { num: [E()], den: [E()] } };
    case 'sqrt': return { type: 'box', kind, slots: { body: [E()] } };
    case 'sum': return { type: 'box', kind, slots: { low: [E()], hi: [E()], body: [E()] } };
    case 'int': return { type: 'box', kind, slots: { low: [E()], hi: [E()], body: [E()] } };
    case 'grp': return { type: 'box', kind, slots: { body: [E()] } };
    default: return undefined;
  }
}
function veInsertStructure(kind) {
  let box;
  if (kind === 'sup' || kind === 'sub') {
    const prevIdx = VE.rootIdx - 1;
    const prev = prevIdx >= 0 ? VE.root[prevIdx] : null;
    const base = (prev && prev.type !== 'box') ? prev : veBoL('txt', 'x');
    if (prev && prev.type !== 'box') { VE.root.splice(prevIdx, 1); VE.rootIdx = prevIdx; }
    box = { type: 'box', kind, slots: { base: [base], ...(kind === 'sup' ? { exp: [veBoL('txt', '')] } : { idx: [veBoL('txt', '')] }) } };
  } else {
    box = veMakeBox(kind);
    // 分数：若前面紧跟一个数字/符号，把它的“分子”复用为分数分子（1/2 → \frac{1}{2}）
    if (kind === 'frac') {
      const prevIdx = VE.rootIdx - 1;
      const prev = prevIdx >= 0 ? VE.root[prevIdx] : null;
      if (prev && prev.type !== 'box') { box.slots.num = [prev]; VE.root.splice(prevIdx, 1); VE.rootIdx = prevIdx; }
    }
  }
  VE.root.splice(VE.rootIdx, 0, box); VE.rootIdx++;
  veRender();
  const leaves = veBoxLeavesOf(box);
  if (leaves.length) {
    // 上标/下标默认聚焦指数/下标位；分数/根式等聚焦第一个待填的空格
    let tgt = leaves[0];
    if (kind === 'sup' || kind === 'sub') tgt = leaves[leaves.length - 1] || leaves[0];
    else { const empty = leaves.find((ld) => ld.type === 'txt' && ld.v === ''); if (empty) tgt = empty; }
    veFocusLeaf(tgt);
  }
}
function veInsertSymbol(latex) {
  VE.root.splice(VE.rootIdx, 0, veBoL('sym', latex)); VE.rootIdx++;
  veRender();
  const c = document.getElementById('veCanvas'); if (c) c.focus(); // 保持连续输入
}

// 找到包含某叶子的根级令牌下标，用于编辑完成后把游标放回其后方
function veRootCaretAfter(leaf) {
  for (let i = 0; i < VE.root.length; i++) {
    const t = VE.root[i];
    if (t === leaf) { VE.rootIdx = i + 1; return; }
    if (t.type === 'box') { if (veBoxLeavesOf(t).indexOf(leaf) >= 0) { VE.rootIdx = i + 1; return; } }
  }
  VE.rootIdx = VE.root.length;
}

function veReset() { VE.root = []; VE.rootIdx = 0; VE.editLeaf = null; veRender(); }

// 键盘：未编辑某格时，字母/数字直接插入为文字，退格/方向键移动游标
function veKey(e) {
  if (VE.editLeaf) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Backspace') {
    if (VE.rootIdx > 0) { VE.root.splice(VE.rootIdx - 1, 1); VE.rootIdx--; veRender(); }
    e.preventDefault(); return;
  }
  if (e.key === 'Delete') { if (VE.rootIdx < VE.root.length) { VE.root.splice(VE.rootIdx, 1); veRender(); } e.preventDefault(); return; }
  if (e.key === 'ArrowLeft') { VE.rootIdx = Math.max(0, VE.rootIdx - 1); veRender(); e.preventDefault(); return; }
  if (e.key === 'ArrowRight') { VE.rootIdx = Math.min(VE.root.length, VE.rootIdx + 1); veRender(); e.preventDefault(); return; }
  if (e.key === 'Enter') { e.preventDefault(); return; }
  if (e.key === 'Tab') { e.preventDefault(); veEnterNextCell(); return; }
  // 数学快捷键：^ 上标、_ 下标、/ 分数
  if (e.key === '^') { e.preventDefault(); veInsertStructure('sup'); return; }
  if (e.key === '_') { e.preventDefault(); veInsertStructure('sub'); return; }
  if (e.key === '/') { e.preventDefault(); veInsertStructure('frac'); return; }
  // 空格：数学模式下普通空格会被 KaTeX 折叠，故插入一个可见间距（\ 转义空格）
  if (e.key === ' ') {
    e.preventDefault();
    VE.root.splice(VE.rootIdx, 0, veBoL('sym', '\\ ')); VE.rootIdx++;
    veRender();
    return;
  }
  if (e.key.length === 1) {
    if (VE.rootIdx > 0 && VE.root[VE.rootIdx - 1].type === 'txt') { VE.root[VE.rootIdx - 1].v += e.key; veRender(); }
    else { VE.root.splice(VE.rootIdx, 0, veBoL('txt', e.key)); VE.rootIdx++; veRender(); }
    e.preventDefault();
  }
}

// Tab：把编辑焦点推进到光标之后第一个可编辑格子（默认是下一个符号/结构的首个输入位）
function veEnterNextCell() {
  const leaves = veCollect(VE.root);
  if (!leaves.length) return;
  let target = null;
  for (let i = VE.rootIdx; i < VE.root.length && !target; i++) {
    const t = VE.root[i];
    const ls = t.type === 'box' ? veBoxLeavesOf(t) : (['txt', 'sym'].includes(t.type) ? [t] : []);
    if (ls.length) target = ls[0];
  }
  if (!target) target = leaves[0];
  veFocusLeaf(target);
}

// 载入已有公式：从目标字段的 $…$ 中解析（仅为一小类常用结构，失败则保留“整段不可分”叶子，仍能渲染）
function veParse(source) {
  source = (source || '').replace(/^\$\s*|\s*\$$/g, '');
  let i = 0, n = source.length;
  const B = (kind, slots) => ({ type: 'box', kind, slots });
  const ws = () => { while (i < n && /\s/.test(source[i])) i++; };
  function seq(arr) {
    while (i < n) {
      ws();
      const c = source[i];
      if (c === '}' || c === ')' || c === undefined) { if (c === '}' || c === ')') i++; return; }
      if (/[0-9]/.test(c)) { let s = ''; while (i < n && /[0-9.,]/.test(source[i])) s += source[i++]; arr.push(veBoL('txt', s)); continue; }
      if (/[a-zA-Z]/.test(c)) { let s = ''; while (i < n && /[a-zA-Z]/.test(source[i])) s += source[i++]; arr.push(veBoL('txt', s)); continue; }
      if (c === '\\') {
        const m = source.slice(i).match(/^\\([a-zA-Z]+|.)/);
        if (!m) { i++; continue; }
        const cmd = m[1]; i += m[0].length;
        if (cmd === 'frac') {
          const num = []; const den = [];
          if (source[i] === '{') { i++; seq(num); }
          if (source[i] === '{') { i++; seq(den); }
          arr.push(B('frac', { num, den })); continue;
        }
        if (cmd === 'sqrt') { const b = []; if (source[i] === '{') { i++; seq(b); } arr.push(B('sqrt', { body: b })); continue; }
        if (cmd === 'sum' || cmd === 'int') { const low = [], hi = [], body = [];
          if (source[i] === '_') { i++; if (source[i] === '{') { i++; seq(low); } }
          if (source[i] === '^') { i++; if (source[i] === '{') { i++; seq(hi); } }
          if (source[i] === '{') { i++; seq(body); }
          arr.push(B(cmd, { low, hi, body })); continue;
        }
        if (cmd === 'left') { const b = []; if (source[i] === '(') { i++; seq(b); } arr.push(B('grp', { body: b })); continue; }
        if (cmd === 'right' || cmd === ')') continue;
        arr.push(veBoL('sym', '\\' + cmd)); continue;
      }
      if (c === '^' || c === '_') {
        const op = c; i++;
        const exp = [];
        if (source[i] === '{') { i++; seq(exp); }
        else if (source[i] === '\\') {
          const m = source.slice(i).match(/^\\([a-zA-Z]+|.)/);
          if (m) { exp.push(veBoL('sym', '\\' + m[1])); i += m[0].length; }
        }
        else if (i < n) { exp.push(veBoL('txt', source[i++])); }
        const base = arr.pop() || veBoL('txt', 'x');
        if (op === '^') arr.push(B('sup', { base: [base], exp }));
        else arr.push(B('sub', { base: [base], idx: exp }));
        continue;
      }
      if (c === '(') { i++; const b = []; seq(b); arr.push(B('grp', { body: b })); continue; }
      if (c === '+=' || '+-=,'.indexOf(c) >= 0) { arr.push(veBoL('sym', c)); i++; continue; }
      // 未识别内容：保留为整段文字叶子（仍由 KaTeX 渲染，可整体改）
      let s = ''; while (i < n && '+-=,()^{}'.indexOf(source[i]) < 0 && !/[a-zA-Z0-9\\]/.test(source[i])) s += source[i++];
      if (s) arr.push(veBoL('txt', s));
      else { arr.push(veBoL('txt', source[i])); i++; }
    }
  }
  const out = [];
  // 顶层：遇到 ')' 不应来自顶层，直接容纳
  while (i < n) { ws(); if ('})'.indexOf(source[i]) >= 0) { i++; continue; } if (i >= n) break; const before = out.length; seq(out); if (out.length === before && i < n) i++; }
  return out;
}
function veLoadFromField() {
  // 从输入框的 $…$ 原始文本里取公式：优先取当前光标附近的，找不到就取该框第一个；
  // 当前框没有时，轮流扫描其余输入框，提升可用性。
  const find = (el) => {
    const v = el.value || '';
    if (!v) return '';
    const at = el.selectionStart != null ? el.selectionStart : v.length;
    const start = v.lastIndexOf('$', at - 1);
    const end = start >= 0 ? v.indexOf('$', start + 1) : -1;
    if (start >= 0 && end > start) return v.slice(start + 1, end);
    const m = v.match(/\$([^$]+)\$/);
    return m ? m[1] : '';
  };
  let srcField = null, exp = null;
  for (const id of Object.keys(MATH_FIELDS)) {
    const got = find(document.getElementById(id));
    if (got) { srcField = id; exp = got; break; }
  }
  if (!exp) { showModal('提示', '输入框里还没有可编辑的公式。请先点「插入」把拼好的公式写入题目，再点「从题目载入」就能继续修改。'); return; }
  document.getElementById('mathTargetLabel').textContent = MATH_FIELDS[srcField];
  VE.root = veParse(exp); VE.rootIdx = VE.root.length; VE.editLeaf = null;
  veRender();
}

function initMathEditor() {
  // 记录当前聚焦的目标字段
  Object.keys(MATH_FIELDS).forEach((id) => {
    document.getElementById(id).addEventListener('focusin', () => {
      mathTarget = id;
      document.getElementById('mathTargetLabel').textContent = MATH_FIELDS[id];
    });
  });

  const panel = document.getElementById('mathPanel');
  const toggle = () => {
    panel.classList.toggle('hidden');
    if (!panel.classList.contains('hidden')) { VE.rtChipEl = null; veRender(); document.getElementById('veCanvas').focus(); }
  };
  document.getElementById('btnMathEditor').addEventListener('click', toggle);

  // 结构按钮
  document.getElementById('veStruct').addEventListener('click', (e) => {
    const b = e.target.closest('[data-kind]'); if (!b) return;
    veInsertStructure(b.dataset.kind);
  });
  // 符号按钮
  document.getElementById('veSym').addEventListener('click', (e) => {
    const b = e.target.closest('[data-sym]'); if (!b) return;
    veInsertSymbol(b.dataset.sym);
  });

  const canvas = document.getElementById('veCanvas');
  canvas.addEventListener('keydown', veKey);
  canvas.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest('.ve-l') || ev.target.closest('input')) return;
    canvas.focus();
  });

  document.getElementById('mathInsert').addEventListener('click', () => {
    const latex = veLatex();
    if (latex) {
      if (VE.rtChipEl) {
        // 正在编辑表单里已渲染的某个公式 → 就地更新该公式
        VE.rtChipEl.setAttribute('data-latex', latex);
        VE.rtChipEl.innerHTML = veGlyphHtml(latex) || latex;
        const rich = VE.rtChipEl.closest('.rt-field'); VE.rtChipEl = null;
        if (rich) rtfSync(rich);
      } else if (RTF_FIELDS.includes(mathTarget)) {
        // 可视化公式插入题干 / 选项目前的光标处 → 以渲染块形式呈现
        const rich = rtfEl(mathTarget);
        insertChipAtCaret(rich, latex);
        rtfSync(rich);
      } else {
        insertText(document.getElementById(mathTarget), '$' + latex + '$');
      }
    }
    panel.classList.add('hidden');
  });
  document.getElementById('mathCancel').addEventListener('click', () => panel.classList.add('hidden'));
  document.getElementById('mathClear').addEventListener('click', veReset);
  document.getElementById('veLoad').addEventListener('click', veLoadFromField);
}

// 光标处插入文本（textarea / input 通用）
function insertText(el, text) {
  const s = el.selectionStart != null ? el.selectionStart : el.value.length;
  const e = el.selectionEnd != null ? el.selectionEnd : el.value.length;
  const v = el.value;
  el.value = v.slice(0, s) + text + v.slice(e);
  el.focus();
  const pos = s + text.length;
  try { el.setSelectionRange(pos, pos); } catch (_) { /* 忽略 */ }
}

// ---------- 题干/选项：公式富文本字段 ----------
// 用 contenteditable 代替隐藏的 textarea：$…$ 以渲染块展示，点公式可回到可视化编辑器；
// 保存时仍以 $…$ 文本写入，兼容既有渲染链路。
const RTF_FIELDS = ['inputQuestion', 'inputOptions'];

function rtfEl(fieldId) { return document.getElementById(fieldId + '-rt'); }
function rtfChip(latex) {
  const sp = document.createElement('span');
  sp.className = 'rt-fml'; sp.setAttribute('data-latex', latex); sp.contentEditable = 'false';
  sp.innerHTML = veGlyphHtml(latex) || latex;
  sp.addEventListener('pointerdown', (ev) => { ev.preventDefault(); ev.stopPropagation(); rtfEditChip(sp); });
  return sp;
}
// 把 $…$ 原文渲染成语义块（普通文字 + 公式块 + 初始占位）
function rtfRenderFromText(ta) {
  const rich = rtfEl(ta.id); if (!rich) return;
  const v = ta.value || '';
  const frag = document.createDocumentFragment();
  const re = /(\$[^$]+\$|\n)/g; let last = 0, m;
  while ((m = re.exec(v)) !== null) {
    if (m.index > last) frag.appendChild(document.createTextNode(v.slice(last, m.index)));
    const tok = m[1];
    if (tok === '\n') frag.appendChild(document.createElement('br'));
    else if (tok.length > 2 && tok[0] === '$') frag.appendChild(rtfChip(tok.slice(1, -1)));
    else frag.appendChild(document.createTextNode(tok));
    last = m.index + tok.length;
  }
  if (last < v.length) frag.appendChild(document.createTextNode(v.slice(last)));
  rich.innerHTML = '';
  rich.appendChild(frag);
}
// 语义块 DOM → $…$ 文本（写回隐藏 textarea）
function rtfSerialize(rich) {
  let out = '';
  (function walk(node) {
    for (const ch of node.childNodes) {
      if (ch.nodeType === 3) out += ch.textContent;
      else if (ch.nodeType === 1) {
        if (ch.classList && ch.classList.contains('rt-fml')) { out += '$' + ch.getAttribute('data-latex') + '$'; continue; }
        if (ch.tagName === 'BR') { out += '\n'; continue; }
        walk(ch);
      }
    }
  })(rich);
  return out;
}
function rtfSync(rich) {
  const ta = document.getElementById(rich.dataset.field);
  if (ta) ta.value = rtfSerialize(rich);
}
function rtfSyncAll() { RTF_FIELDS.forEach((id) => rtfRenderFromText(document.getElementById(id))); }

// ---------- 自动识别数学文本 → LaTeX ----------
// OCR / 粘贴进来的公式其实是普通文本（无 $…$），这里把"算式/方程/带运算符或上下标的数字式子"
// 自动转成 LaTeX 并包成 $…$，从而在题干/选项里直接渲染成可点击编辑的公式块。
const VE_SUPER = { '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁺':'+','⁻':'-' };
const VE_SUB = { '₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9' };
const VE_MATH_SYM = { '×':'\\times', '÷':'\\div', '±':'\\pm', '≤':'\\leq', '≥':'\\geq', '≠':'\\neq', '≈':'\\approx', '∞':'\\infty', '·':'\\cdot', '−':'-', 'π':'\\pi', 'α':'\\alpha', 'β':'\\beta', 'θ':'\\theta', 'λ':'\\lambda', 'μ':'\\mu', 'Δ':'\\Delta', '∑':'\\sum', '∫':'\\int' };

function veIsMathStrong(core) {
  // 强数学信号：等号/乘除/比较/上下标/根式/求和积分/希腊字母/反斜杠命令；
  // 或"含数字且含 + - * /"（算式），避免把“2024”这种纯年份或普通英文误判
  return (/[=×÷±≤≥≠^√∑∫∞°%\\]/.test(core))
    || (/[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉αβγθλμπδ]/.test(core))
    || (/\d/.test(core) && /[+\-*/=]/.test(core));
}
// 把一个已判定为数学的片段规范化成 LaTeX（unicode 上下标/符号 → \命令）
function veMathToLatex(expr) {
  let s = expr;
  s = s.replace(/√\{([^{}]*)\}/g, '\\sqrt{$1}');
  s = s.replace(/√([A-Za-z0-9]+)/g, '\\sqrt{$1}');
  s = s.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+/g, (m) => `^{${m.split('').map((c) => VE_SUPER[c]).join('')}}`);
  s = s.replace(/[₀₁₂₃₄₅₆₇₈₉]+/g, (m) => `_{${m.split('').map((c) => VE_SUB[c]).join('')}}`);
  for (const k of Object.keys(VE_MATH_SYM)) { const v = VE_MATH_SYM[k]; s = s.split(k).join(v + (v.startsWith('\\') ? ' ' : '')); }
  s = s.replace(/\^\{([^{}]*)\}/g, '^{$1}');
  s = s.replace(/\^([A-Za-z0-9]+)/g, '^{$1}');
  s = s.replace(/\_\{([^{}]*)\}/g, '_{$1}');
  s = s.replace(/_([A-Za-z0-9]+)/g, '_{$1}');
  s = s.replace(/\*/g, '\\times ');
  s = s.replace(/(\d+)\/(\d+)/g, '\\frac{$1}{$2}');
  return s.trim();
}
// 在文本里找到数学片段并包裹，已存在的 $…$ 原样保留（可重复调用，幂等）
function mathifyText(text) {
  if (!text) return text;
  const parts = []; let last = 0, m;
  const re = /\$[^$]+\$/g;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(mathifyRun(text.slice(last, m.index)));
    parts.push(m[0]);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(mathifyRun(text.slice(last)));
  return parts.join('');
}
function mathifyRun(seg) {
  // 以中文字符/标点为界切出"非中文片段"，再判断其中是否真为数学
  const out = []; let buf = '';
  const isCjk = (c) => /[\u4e00-\u9fff，。、；：！？…—”“’‘（）【】《》]/.test(c);
  for (const ch of seg) {
    if (isCjk(ch)) { if (buf) { out.push(buf); buf = ''; } out.push(ch); }
    else buf += ch;
  }
  if (buf) out.push(buf);
  return out.map((t) => {
    if (isCjk(t) || t === '') return t;
    const core = t.replace(/\s+/g, '');
    return veIsMathStrong(core) ? ('$' + veMathToLatex(t) + '$') : t;
  }).join('');
}
// 在光标处插入一个可换行的文字节点（兼容块级容器，Enter 用\n渲染）
function insertAtCaret(el, insertNode) {
  const sel = window.getSelection(); let r;
  if (sel.rangeCount && el.contains(sel.getRangeAt(0).startContainer)) r = sel.getRangeAt(0);
  else { r = document.createRange(); r.setStart(el, el.childNodes.length); r.collapse(true); }
  r.deleteContents();
  r.insertNode(insertNode);
  const c = document.createRange(); c.setStartAfter(insertNode); c.collapse(true);
  sel.removeAllRanges(); sel.addRange(c);
}
function insertTextAtCaret(el, text) { if (text) insertAtCaret(el, document.createTextNode(text)); }
// 在富文本字段当前光标处插入公式块（优先使用此前记录的光标范围，避免焦点被面板按钮抢占后插错位置）
function insertChipAtCaret(el, latex) {
  const range = VE.rtSel[el.dataset.field];
  if (range && el.contains(range.startContainer) && el.contains(range.endContainer)) {
    const chip = rtfChip(latex);
    range.deleteContents();
    range.insertNode(chip);
    const sel = document.getSelection();
    const c = document.createRange(); c.setStartAfter(chip); c.collapse(true);
    sel.removeAllRanges(); sel.addRange(c);
    return;
  }
  insertAtCaret(el, rtfChip(latex));
}
function rtfStoreSel() {
  RTF_FIELDS.forEach((id) => {
    const rich = rtfEl(id);
    const sel = document.getSelection();
    if (sel.rangeCount && rich.contains(sel.anchorNode) && rich.contains(sel.focusNode)) {
      VE.rtSel[id] = sel.getRangeAt(0).cloneRange();
    } else {
      delete VE.rtSel[id];
    }
  });
}
// 点击公式块 → 载入可视化编辑器就地修改
function rtfEditChip(sp) {
  VE.rtChipEl = sp;
  const fieldId = sp.closest('.rt-field').dataset.field;
  mathTarget = fieldId;
  const label = document.getElementById('mathTargetLabel'); if (label) label.textContent = MATH_FIELDS[fieldId];
  VE.root = veParse(sp.getAttribute('data-latex')); VE.rootIdx = VE.root.length; VE.editLeaf = null;
  veRender();
  document.getElementById('mathPanel').classList.remove('hidden');
  document.getElementById('veCanvas').focus();
}
function initRTFields() {
  RTF_FIELDS.forEach((id) => {
    const ta = document.getElementById(id);
    ta.style.display = 'none';
    const rich = rtfEl(id);
    rtfRenderFromText(ta);
    // Enter 用换行文本 + pre-wrap 渲染，避免浏览器插入<div>破坏序列化
    rich.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); insertTextAtCaret(rich, '\n'); rtfSync(rich); }
    });
    // 粘贴仅接受纯文本，避免塞入<…>污染结构
    rich.addEventListener('paste', (ev) => {
      ev.preventDefault();
      const t = (ev.clipboardData || window.clipboardData).getData('text');
      insertTextAtCaret(rich, mathifyText(t)); rtfSync(rich);
    });
    rich.addEventListener('input', () => rtfSync(rich));
    rich.addEventListener('focusin', () => {
      mathTarget = id;
      const label = document.getElementById('mathTargetLabel'); if (label) label.textContent = MATH_FIELDS[id];
    });
  });
  // 记录题干/选项里最近的选中范围，供插入公式时准确定位
  document.addEventListener('selectionchange', rtfStoreSel);
}

// ---------- 划选未识别片段 → 用公式编辑器打开 ----------
// 题干/选项里自动识别不出的纯字母+数字片段，也可以划选后一键扔进公式编辑器，强行按公式改。
let rtFmlBtn = null, rtFmlRange = null;
const RT_FML_OK = /^[0-9A-Za-z+\-*/=×÷±≤≥≠≈√·^_.()，。、；： $%\u00a0]+$/;

function rtFmlField(el) { return el ? el.closest('.rt-field') : null; }

function hideRtfmlBtn(saveRange) {
  if (rtFmlBtn) rtFmlBtn.classList.add('hidden');
  if (!saveRange) rtFmlRange = null;
}

function showRtfmlBtn() {
  const sel = window.getSelection();
  // 选区被收起（例如点了按钮本身）时不显示，但保留已记录的范围供按钮点击使用
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) { hideRtfmlBtn(true); return; }
  const r = sel.getRangeAt(0);
  const rich = rtFmlField(r.commonAncestorContainer && r.commonAncestorContainer.nodeType === 1
    ? r.commonAncestorContainer : r.startContainer);
  if (!rich) { hideRtfmlBtn(); return; }
  const text = sel.toString().replace(/[\n\r]/g, ' ').trim();
  if (!text || text.length > 40 || !RT_FML_OK.test(text)) { hideRtfmlBtn(); return; }
  rtFmlRange = r.cloneRange();
  const rect = r.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) { hideRtfmlBtn(); return; }
  rtFmlBtn.style.left = Math.max(8, rect.left) + 'px';
  rtFmlBtn.style.top = (rect.bottom + 6) + 'px';
  rtFmlBtn.classList.remove('hidden');
}

function onClickRtfmlBtn() {
  if (!rtFmlRange) return;
  const rich = rtFmlField(rtFmlRange.commonAncestorContainer && rtFmlRange.commonAncestorContainer.nodeType === 1
    ? rtFmlRange.commonAncestorContainer : rtFmlRange.startContainer);
  if (!rich) { hideRtfmlBtn(); return; }
  const raw = rtFmlRange.toString().replace(/[\n\r]/g, ' ').trim();
  const latex = veMathToLatex(raw) || (raw || 'x');
  const chip = rtfChip(latex);
  rtFmlRange.deleteContents();
  rtFmlRange.insertNode(chip);
  const sel = window.getSelection();
  const c = document.createRange(); c.setStartAfter(chip); c.collapse(true);
  sel.removeAllRanges(); sel.addRange(c);
  hideRtfmlBtn();
  rtfSync(rich);
  rtfEditChip(chip); // 直接打开公式编辑器继续拼
}

function initRTSelectEdit() {
  rtFmlBtn = document.getElementById('rtFmlBtn');
  if (!rtFmlBtn) return;
  rtFmlBtn.addEventListener('click', onClickRtfmlBtn);
  document.addEventListener('mouseup', () => { setTimeout(showRtfmlBtn, 0); });
  document.addEventListener('mousedown', (e) => {
    if (!rtFmlBtn.contains(e.target)) hideRtfmlBtn(); // 新选区开始时清空旧范围
  });
  window.addEventListener('resize', () => hideRtfmlBtn(true));
  window.addEventListener('scroll', () => hideRtfmlBtn(true), true);
}

// 拍照识别：调用本地摄像头，实时预览后拍照
// 用 onclick 赋值 + teardown 统一清理，避免重复打开导致监听器叠加、重复识别
function openPhotoCapture() {
  const mask = document.getElementById('captureMask');
  const video = document.getElementById('captureVideo');
  const cancelBtn = document.getElementById('captureCancel');
  const shootBtn = document.getElementById('captureShoot');
  let stream = null;

  const teardown = () => {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    video.srcObject = null;
    mask.classList.add('hidden');
    cancelBtn.onclick = null;
    shootBtn.onclick = null;
  };

  cancelBtn.onclick = teardown;
  shootBtn.onclick = () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      teardown();
      await recognizeImage(blob);
    }, 'image/jpeg', 0.92);
  };

  (async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    } catch (e) {
      teardown();
      showModal('提示', '无法访问摄像头，请检查浏览器权限');
      return;
    }
    video.srcObject = stream;
    mask.classList.remove('hidden');
    try { await video.play(); } catch (e) { /* autoplay 已触发，忽略 */ }
  })();
}

// 框选识别：先弹提示让用户把题目调到屏幕上，确认就绪后再截屏。
// 避免“一点按钮就立刻抓当前屏”，用户来不及把目标题目摆出来。
function openScreenSelect() {
  const readyMask = document.getElementById('selectReadyMask');
  const goBtn = document.getElementById('selectReadyGo');
  const cancelBtn = document.getElementById('selectReadyCancel');
  const start = () => { readyMask.classList.add('hidden'); doScreenSelect(); };
  goBtn.onclick = start;
  cancelBtn.onclick = () => { readyMask.classList.add('hidden'); };
  readyMask.classList.remove('hidden');
}

// 截屏 + 整屏压暗 + 框选聚光 + 确认/重选（用户就绪后进入）
// screenSelectActive 保证任意时刻只有一个框选会话，避免重复截屏/重复监听导致“确认”连发多次 OCR
let screenSelectActive = false;

async function doScreenSelect() {
  if (screenSelectActive) return;
  screenSelectActive = true;
  let shot;
  try { shot = await API.Screen.capture(); } catch (e) { screenSelectActive = false; showModal('提示', '屏幕捕获失败'); return; }
  if (!shot || !shot.ok) { screenSelectActive = false; showModal('提示', (shot && shot.error) || '屏幕捕获失败'); return; }

  const mask = document.getElementById('selectMask');
  const img = document.getElementById('selectImg');
  const stage = document.getElementById('selectStage');
  const box = document.getElementById('selectBox');
  const confirmBar = document.getElementById('selectConfirm');
  const tip = document.getElementById('selectTip');
  const okBtn = document.getElementById('selectOk');
  const redoBtn = document.getElementById('selectRedo');

  img.src = shot.image;
  await new Promise((r) => { img.onload = r; });

  // 关键：必须先显示覆盖层让 stage 完成布局，再测量尺寸。
  // 若在 hidden（display:none）状态下读 clientWidth/clientHeight 会得到 0，
  // 图片会被设成 0×0，导致整屏发黑且框选失效。
  mask.classList.remove('hidden');
  const stageW = stage.clientWidth, stageH = stage.clientHeight;
  const nw = img.naturalWidth, nh = img.naturalHeight;
  const scale = Math.min(stageW / nw, stageH / nh);
  const cw = nw * scale, ch = nh * scale;
  img.style.left = ((stageW - cw) / 2) + 'px';
  img.style.top = ((stageH - ch) / 2) + 'px';
  img.style.width = cw + 'px';
  img.style.height = ch + 'px';

  // 聚光：框选盒用同一张屏图做背景并随框偏移，框内显示原图、框外由压暗层盖住
  // （全局样式：整屏被 .select-dim 压暗，只有框选区域以正常亮度显示）
  box.style.backgroundImage = `url(${shot.image})`;

  const toFull = (nx, ny) => ({
    x: Math.round((nx / img.naturalWidth) * shot.full_w),
    y: Math.round((ny / img.naturalHeight) * shot.full_h),
  });
  const mapPoint = (e) => {
    const rect = img.getBoundingClientRect();
    return {
      x: Math.min(img.naturalWidth, Math.max(0, ((e.clientX - rect.left) / rect.width) * img.naturalWidth)),
      y: Math.min(img.naturalHeight, Math.max(0, ((e.clientY - rect.top) / rect.height) * img.naturalHeight)),
    };
  };

  let dragging = false, start = null, cur = null, doing = false;

  function reset() {
    box.style.display = 'none';
    confirmBar.classList.add('hidden');
    tip.style.display = '';
    tip.textContent = '按住鼠标拖拽，框选要识别的区域';
  }

  function drawBox() {
    const stageRect = stage.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    const ox = imgRect.left - stageRect.left;
    const oy = imgRect.top - stageRect.top;
    const sxp = ox + (start.x / img.naturalWidth) * imgRect.width;
    const syp = oy + (start.y / img.naturalHeight) * imgRect.height;
    const cxp = ox + (cur.x / img.naturalWidth) * imgRect.width;
    const cyp = oy + (cur.y / img.naturalHeight) * imgRect.height;
    const l = Math.min(sxp, cxp), t = Math.min(syp, cyp);
    box.style.left = l + 'px';
    box.style.top = t + 'px';
    box.style.width = Math.abs(cxp - sxp) + 'px';
    box.style.height = Math.abs(cyp - syp) + 'px';
    box.style.backgroundSize = `${imgRect.width}px ${imgRect.height}px`;
    box.style.backgroundPosition = `${-(l - ox)}px ${-(t - oy)}px`;
    box.style.display = 'block';
  }

  function close() {
    screenSelectActive = false;
    reset();
    mask.classList.add('hidden');
    img.src = '';
    box.style.backgroundImage = 'none';
    stage.removeEventListener('pointerdown', onDown);
    stage.removeEventListener('pointermove', onMove);
    stage.removeEventListener('pointerup', onUp);
    stage.removeEventListener('pointercancel', onUp);
    // 释放共用按钮，避免多次框选累加监听导致“确认”连发多次 OCR
    okBtn.onclick = null;
    redoBtn.onclick = null;
    document.getElementById('selectCancel').removeEventListener('click', close);
  }

  // 用 Pointer Events + setPointerCapture：把指针“锁定”在覆盖层上，
  // 即使拖出窗口/覆盖层也能持续收到移动与抬起，框选不会中途丢失或“一闪就没了”。
  function onDown(e) {
    // 确认/重选按钮、提示、压暗层都在 stage 内，按下后会冒泡触发 drag。
    // 若不忽略，点“确认”会被当作开始拖拽而 reset，导致“框选完了但确定无效”。
    if (e.target.closest('#selectConfirm, #selectTip, #selectDim')) return;
    e.preventDefault();
    reset();
    dragging = true;
    start = mapPoint(e);
    cur = { ...start };
    try { stage.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
  }
  function onMove(e) {
    if (!dragging) return;
    cur = mapPoint(e);
    drawBox();
  }
  function onUp() {
    if (!dragging) return;
    dragging = false;
    // 太小的拖拽视为误触，清空重来；否则保留选框并给出确认/重选
    if (Math.abs(cur.x - start.x) < 8 || Math.abs(cur.y - start.y) < 8) { reset(); return; }
    // 隐藏顶部重复文案，确认信息由底部确认条统一接管，避免两处“相同内容”叠着显示
    tip.style.display = 'none';
    confirmBar.classList.remove('hidden');
  }

  okBtn.onclick = async () => {
    if (doing) return;
    doing = true;
    const x0 = Math.min(start.x, cur.x), y0 = Math.min(start.y, cur.y);
    const w = Math.abs(cur.x - start.x), h = Math.abs(cur.y - start.y);
    const p0 = toFull(x0, y0), p1 = toFull(x0 + w, y0 + h);
    close();
    showModal('OCR 识别', '正在识别，请稍候…');
    try {
      const res = await API.Screen.ocr({ x: p0.x, y: p0.y, w: p1.x - p0.x, h: p1.y - p0.y });
      if (!res || !res.ok) { showModal('OCR 提示', (res && res.error) || '识别失败，可手动录入'); return; }
      document.getElementById('inputQuestion').value = mathifyText(res.question_text || '');
      document.getElementById('inputOptions').value = (res.options || []).map((o) => mathifyText(o)).join('\n');
      rtfSyncAll();
      focusAddForm();
      showModal('OCR 完成', '识别结果已填入表单，请核对修正后保存');
    } catch (e) { showModal('OCR 提示', '识别请求失败，可手动录入'); }
  };
  redoBtn.onclick = reset;

  stage.addEventListener('pointerdown', onDown);
  stage.addEventListener('pointermove', onMove);
  stage.addEventListener('pointerup', onUp);
  stage.addEventListener('pointercancel', onUp);
  document.getElementById('selectCancel').addEventListener('click', close);
}

// ---------- 题目列表 ----------
function initBank() {
  document.querySelectorAll('[data-bank]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-bank]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentBank = btn.dataset.bank;
      renderQuestionList();
    });
  });
}

async function renderQuestionList() {
  const qRes = await API.Questions.list();
  questions = qRes.questions || [];
  const filtered = questions.filter((q) => {
    if (currentBank === 'wrong') return q.is_wrong;
    if (currentBank === 'mastered') return q.is_mastered;
    return true;
  });
  document.getElementById('bankCount').textContent = `${filtered.length} 道`;
  const list = document.getElementById('questionList');
  if (!filtered.length) { list.innerHTML = '<div class="cluster-list empty">暂无题目</div>'; return; }
  list.innerHTML = '';
  filtered.forEach((q) => {
    const item = document.createElement('div');
    item.className = `question-item ${q.is_wrong ? 'is-wrong' : q.is_mastered ? 'is-mastered' : ''}`;
    const tags = (q.knowledge_tags || []).map((t) => `<span class="q-tag">${t}</span>`).join('');
    const stateTag = q.is_wrong ? '<span class="q-tag wrong">错题</span>' : q.is_mastered ? '<span class="q-tag">已掌握</span>' : '';
    const opts = (q.options || []).map((o, i) => `<div class="q-opt">${renderMathText(o)}</div>`).join('');
    item.innerHTML = `
      <div class="q-text">${renderMathText(q.question_text)}</div>
      ${opts ? `<div class="q-opts">${opts}</div>` : ''}
      <div class="q-meta">${stateTag}${tags}<span>来源：${q.source}</span></div>
      <div class="q-edit"><button type="button" class="btn btn-ghost q-edit-btn">编辑</button></div>
    `;
    const editBtn = item.querySelector('.q-edit-btn');
    if (editBtn) editBtn.addEventListener('click', () => enterEditQuestion(q));
    list.appendChild(item);
  });
}

// ---------- 答题 ----------
async function loadQuestionSelect() {
  document.getElementById('answerResult').classList.add('hidden');
  document.getElementById('answerResult').innerHTML = '';
  const qRes = await API.Questions.list();
  questions = (qRes.questions || []).filter((q) => !q.is_wrong);
  const sel = document.getElementById('selectQuestion');
  sel.innerHTML = '';
  if (questions.length === 0) {
    document.getElementById('answerArea').textContent = '题库暂无未掌握题目，请先添加题目';
    sel.innerHTML = '<option>暂无题目</option>';
    return;
  }
  questions.forEach((q) => {
    const opt = document.createElement('option');
    opt.value = q.question_id;
    opt.textContent = q.question_text.slice(0, 30);
    sel.appendChild(opt);
  });
  sel.onchange = () => showQuestion(questions.find((q) => q.question_id === sel.value));
  showQuestion(questions[0]);
}

async function showQuestion(q) {
  currentQuestion = q;
  const area = document.getElementById('answerArea');
  let text = q.question_text;
  if (q.options && q.options.length) text += '\n\n' + q.options.join('\n');
  // 渲染 $...$ 公式（读者可直接看到公式）
  area.innerHTML = renderMathText(text);
  // 答题口径提示：明确告知用户以怎样的形式作答，避免无从下手/瞎填
  document.getElementById('answerFormatHint').textContent = answerFormatHint(q);
  document.getElementById('answerActions').classList.remove('hidden');
  // 切换题目时清空读题划词记录
  readingPath = [];
  renderReadingPath();
  document.getElementById('readingResult').classList.add('hidden');
  // 提交前不做任何高亮/输出，关键词在提交后统一揭示
}

// 提交答案后才高亮题干关键词（提交前界面不做任何输出）
async function revealKeywords(area) {
  if (!currentQuestion) return;
  let kws = currentQuestion.keywords || [];
  if (!kws.length) {
    try {
      const r = await API.Questions.suggestKeywords({
        question_text: currentQuestion.question_text || '',
        options: currentQuestion.options || [],
      });
      kws = r.keywords || [];
      currentQuestion.keywords = kws;
    } catch (e) { kws = []; }
  }
  highlightKeywords(area, kws);
}

// 在已渲染的题面 DOM 上，只对文本节点里的关键词打 <mark> 高亮
// （不触碰标签/公式结构，避免误伤 LaTeX 渲染出的 HTML）
function highlightKeywords(el, words) {
  if (!el || !words || !words.length) return;
  const escWords = words.filter((w) => w && w.length).sort((a, b) => b.length - a.length); // 长词优先，减少子串误包
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const text = node.nodeValue || '';
    if (!text.trim()) return;
    const hits = [];
    escWords.forEach((w) => {
      let i = text.indexOf(w);
      while (i >= 0) { hits.push({ s: i, e: i + w.length, w }); i = text.indexOf(w, i + w.length); }
    });
    if (!hits.length) return;
    hits.sort((a, b) => a.s - b.s);
    // 合并重叠命中，避免 <mark> 嵌套
    const spans = [];
    let cur = null;
    hits.forEach((h) => {
      if (cur && h.s < cur.e) { if (h.e > cur.e) cur.e = h.e; return; }
      cur = { s: h.s, e: h.e }; spans.push(cur);
    });
    const frag = document.createDocumentFragment();
    let pos = 0;
    spans.forEach((sp) => {
      if (sp.s > pos) frag.appendChild(document.createTextNode(text.slice(pos, sp.s)));
      const mark = document.createElement('mark');
      mark.className = 'kw-hl';
      mark.textContent = text.slice(sp.s, sp.e);
      frag.appendChild(mark);
      pos = sp.e;
    });
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    node.parentNode.replaceChild(frag, node);
  });
}

// 根据题目与标准答案推断"答题口径"，让用户知道怎么写答案
function answerFormatHint(q) {
  const ca = (q.correct_answer || '').trim();
  if (!ca) return '此题无标准答案设定，请直接作答。';
  const isOption = q.options && q.options.length > 0 && !/\d/.test(ca) && /[A-Da-d]/.test(ca);
  if (isOption) return '此题从下方选项中选择，请填写选项字母（如：C）。';
  if (/最小|最大/.test(ca)) {
    const nums = (ca.match(/-?\d+(\.\d+)?/g) || []);
    return `此题求最值，请填写两个数值，用「,」或「和」分隔，如：${nums.join(',')}`;
  }
  if (/\d/.test(ca)) {
    const m = (ca.match(/-?\d+(\.\d+)?/) || ['']);
    return `此题答案为数值，直接填写${m[0] ? '，例如：' + m[0] : ''}（单位可省略）。`;
  }
  return '请按题目要求直接填写答案文字。';
}

// F1-4：捕获用户划选关键词的顺序，形成读题路径
let readingPath = [];
// 答题/审题 双模式状态
let answerMode = 'answer';        // 'answer' 作答 | 'reading' 审题
let stkReadingPath = [];          // 审题模式的划词轨迹（与作答模式的 readingPath 隔离）
let stkCurrentQuestion = null;
let stkMetaReadyFor = '';         // 当前已加载筛选选项的学科

function initReadingCapture() {
  document.addEventListener('mouseup', (e) => {
    // 读文章划词（默认开启）：在文章文本框内划选文字即静默采集读文章关键词，不干扰复制/粘贴
    const articleTa = document.getElementById('inputArticle');
    if (articleTa && articleTa.contains(e.target)) {
      const s = articleTa.selectionStart >> 0, en = articleTa.selectionEnd >> 0;
      const text = articleTa.value.slice(s, en).trim();
      if (text && text.length <= 40) API.Reading.capture([text], 'article').catch(() => {});
    }
    // 框选识别覆盖层打开时不采集划词、也不清除选择，避免干扰框选
    const selectMask = document.getElementById('selectMask');
    if (selectMask && !selectMask.classList.contains('hidden')) return;
    // 关键修复：在输入框/文本域/下拉框等可编辑控件内抬起鼠标时，
    // 不采集也不清除选区，保证粘贴、ctrl+a 全选、复制等操作不被干扰。
    const t = e.target;
    if (t && (t.closest('input, textarea, select, [contenteditable="true"], [contenteditable=""]'))) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;
    const text = sel.toString().trim();
    if (!text || text.length > 40) return;
    // 读题场景：题干默认生效划词，按当前模式记录到对应划词路径
    if (answerMode === 'reading') {
      stkReadingPath.push(text);
      renderStkPath();
    } else if (document.getElementById('selectQuestion')) {
      readingPath.push(text);
      renderReadingPath();
    }
    sel.removeAllRanges();
  });
}

// 读题划词分析页：汇总划词记录（不再做"数据/逻辑优先"三分类）
async function loadReadingHistory() {
  const summaryEl = document.getElementById('readingSummary');
  const listEl = document.getElementById('readingHistory');
  let res;
  try { res = await API.Reading.history(); } catch (e) { return; }
  const records = res.records || [];
  if (!records.length) {
    summaryEl.innerHTML = '<div class="chart-empty">暂无划词记录，请在「采集模块」开启划词采集后，于答题/读文章时划选关键词</div>';
    listEl.innerHTML = '';
    return;
  }
  const total = records.length;
  const totalWords = records.reduce((s, r) => s + (r.words || []).length, 0);
  summaryEl.innerHTML = `
    <div class="reading-summary-head">已采集 <b>${total}</b> 次划词 · 共 <b>${totalWords}</b> 个词</div>`;
  // 历史列表
  listEl.innerHTML = records.map((r, i) => `
    <div class="reading-history-item">
      <div class="reading-history-head">
        <span class="reading-history-tag">${r.source === 'article' ? '读文章' : '读题'}</span>
        <span class="reading-history-time">${(r.timestamp || '').slice(0, 16).replace('T', ' ')}</span>
      </div>
      <div class="reading-history-words">${(r.words || []).map((w) => `<span class="reading-chip">${w}</span>`).join('')}</div>
    </div>`).join('');
}

// 划词轨迹渲染（统一函数）：作答/审题两模式共用，只展示轨迹 chips，不做提前输出
// （质量/讲评一律在提交或完成审题后统一揭示，符合"提交前不剧透"的交互约定）
function renderPathChips(el, arr) {
  if (!el) return;
  el.innerHTML = arr.length
    ? arr.map((w, i) => `<span class="reading-chip">${i + 1}. ${w}</span>`).join('')
    : '<span class="reading-empty">尚未划词</span>';
}
function renderReadingPath() { renderPathChips(document.getElementById('readingPath'), readingPath); }
function renderStkPath() { renderPathChips(document.getElementById('stkPath'), stkReadingPath); }

// 归一化划词/关键词，便于忽略标点/空格/大小写/公式符号的差异后比对
function readingNorm(s) {
  return String(s || '').replace(/[\s$，。、；：,.；;:，'"“”（）()]/g, '').toLowerCase();
}
// 计算学生划词 vs 本题标准关键词的覆盖/命中/漏划
function assessReadingQuality(std, picked) {
  const stdN = (std || []).map(readingNorm).filter(Boolean);
  const picks = (picked || []).map(readingNorm).filter(Boolean);
  if (!stdN.length) return null;
  const hitIdx = new Set();
  picks.forEach((p) => {
    stdN.forEach((s, i) => { if (!hitIdx.has(i) && (s.includes(p) || p.includes(s))) hitIdx.add(i); });
  });
  const hit = hitIdx.size;
  const missed = stdN.map((s, i) => (hitIdx.has(i) ? null : std[i])).filter(Boolean);
  return {
    hit, stdTotal: stdN.length, pickTotal: picks.length,
    recall: stdN.length ? hit / stdN.length : 0,
    precision: picks.length ? hit / picks.length : 0,
    missed,
  };
}

async function analyzeReading() {
  // 划词分析：直接用"学生划词 vs 本题存储的关键词"做对比（放弃原"数据/逻辑优先"三分类）
  const el = document.getElementById('readingResult');
  if (!readingPath.length) { el.classList.add('hidden'); return; }
  renderReadingResult(el);
}

function renderReadingResult(el) {
  const std = currentQuestion && currentQuestion.keywords && currentQuestion.keywords.length
    ? currentQuestion.keywords : null;
  if (!std) {
    el.innerHTML = '<div class="reading-off">本题未存关键词，无法对比。可在「添加题目」里点「一键划词」补上。</div>';
    el.classList.remove('hidden');
    return;
  }
  const a = assessReadingQuality(std, readingPath);
  if (!a) { el.classList.add('hidden'); return; }
  const ok = Math.round(a.recall * 100);
  const grade = ok >= 80 ? 'good' : (ok >= 50 ? 'warn' : 'weak');
  el.innerHTML = `
    <div class="reading-quality-card ${grade}">
      <div class="rq-head">划词质量：命中 <b>${a.hit}/${a.stdTotal}</b> 个该划词（覆盖 ${ok}%）</div>
      <div class="rq-metrics">
        <span>识别的关键词：<b>${a.hit}</b> 个</span>
        <span>你划的词：<b>${a.pickTotal}</b> 个</span>
        <span>多余划词：<b>${Math.max(0, a.pickTotal - a.hit)}</b> 个</span>
      </div>
      ${a.missed.length ? `<div class="rq-missed">漏划：${a.missed.map((m) => `<b>${m}</b>`).join('、')}</div>` : '<div class="rq-missed no">没漏划，全划到了</div>'}
    </div>`;
  el.classList.remove('hidden');
}

// ---------- 审题模式：逐词讲评 + 题干高亮 ----------
// 依据标准词把学生划词归为 命中/漏划/多余，并对题面做三色高亮
function renderStressReview(area, resultEl, a, std, picks) {
  const stdN = (std || []).map(readingNorm).filter(Boolean);
  const used = new Array(std.length).fill(false);
  picks.forEach((p) => {
    const pn = readingNorm(p);
    stdN.forEach((sn, i) => {
      if (!used[i] && sn && pn && (sn.includes(pn) || pn.includes(sn))) used[i] = true;
    });
  });
  const missed = std.map((s, i) => (used[i] ? null : s)).filter(Boolean);
  const extra = (picks || []).filter((p) => !std.some((s) => {
    const pn = readingNorm(p), sn = readingNorm(s);
    return pn && sn && (sn.includes(pn) || pn.includes(sn));
  }));
  // 高亮注释：命中绿 / 漏划橙 / 多余灰
  const ann = std.map((s, i) => ({ w: s, cls: used[i] ? 'kw-hit' : 'kw-missed' }));
  extra.forEach((p) => ann.push({ w: p, cls: 'kw-extra' }));
  if (area) highlightReadingKeywords(area, ann);
  const ok = Math.round((a ? a.recall : 0) * 100);
  const grade = ok >= 80 ? 'good' : (ok >= 50 ? 'warn' : 'weak');
  resultEl.innerHTML = `
    <div class="reading-quality-card ${grade}">
      <div class="rq-head">审题划词：命中 <b>${a ? a.hit : 0}/${a ? a.stdTotal : 0}</b> 个该划词（覆盖 ${ok}%）</div>
      <div class="rq-metrics">
        <span>该划（标准）：<b>${a ? a.stdTotal : 0}</b> 个</span>
        <span>你划了：<b>${(picks || []).length}</b> 个</span>
        <span>漏划：<b>${missed.length}</b> · 多余：<b>${extra.length}</b></span>
      </div>
      ${missed.length ? `<div class="rq-missed">漏划：${missed.map((m) => `<b>${esc(m)}</b>`).join('、')}</div>` : '<div class="rq-missed no">没漏划，全部划到了</div>'}
      ${extra.length ? `<div class="rq-extra">多余划词：${extra.map((m) => `<b>${esc(m)}</b>`).join('、')}</div>` : ''}
    </div>
    <div class="rq-legend">
      <span class="lg lg-hit">■ 命中</span><span class="lg lg-missed">■ 漏划</span><span class="lg lg-extra">■ 多余</span>
      <span class="lg-tip">题干中已按上述标签高亮对应词，可回看题面</span>
    </div>`;
  resultEl.classList.remove('hidden');
}

// 在已渲染题面上，对文本节点打多色 <mark>（命中绿/漏划橙/多余灰），不破坏 LaTeX 结构
function highlightReadingKeywords(el, ann) {
  if (!el || !ann || !ann.length) return;
  const items = ann.filter((x) => x && x.w && x.w.trim()).sort((a, b) => b.w.length - a.w.length); // 长词优先
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const text = node.nodeValue || '';
    if (!text.trim()) return;
    const hits = [];
    items.forEach((it) => {
      let i = text.indexOf(it.w);
      while (i >= 0) { hits.push({ s: i, e: i + it.w.length, cls: it.cls }); i = text.indexOf(it.w, i + it.w.length); }
    });
    if (!hits.length) return;
    hits.sort((a, b) => a.s - b.s);
    const spans = [];
    let cur = null;
    hits.forEach((h) => {
      if (cur && h.s < cur.e) { if (h.e > cur.e) cur.e = h.e; return; }
      cur = { s: h.s, e: h.e, cls: h.cls }; spans.push(cur);
    });
    const frag = document.createDocumentFragment();
    let pos = 0;
    spans.forEach((sp) => {
      if (sp.s > pos) frag.appendChild(document.createTextNode(text.slice(pos, sp.s)));
      const mark = document.createElement('mark');
      mark.className = 'kw-hl ' + sp.cls;
      mark.textContent = text.slice(sp.s, sp.e);
      frag.appendChild(mark);
      pos = sp.e;
    });
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    node.parentNode.replaceChild(frag, node);
  });
}

// ---------- 审题模式：模式切换 + 题库加载 ----------
// 进入答题/审题页时按当前模式载入
function setModeAndLoad() {
  if (answerMode === 'reading') initReadingBank(true);
  else loadQuestionSelect();
}

// 切换 作答 / 审题 模式
function switchAnswerMode(mode, bootstrapping) {
  answerMode = mode;
  const pAnswer = document.getElementById('qaAnswerPanel');
  const pReading = document.getElementById('qaReadingPanel');
  document.querySelectorAll('#answerModeSeg .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  const toAnswer = mode === 'answer';
  if (pAnswer) pAnswer.classList.toggle('hidden', !toAnswer);
  if (pReading) pReading.classList.toggle('hidden', toAnswer);
  if (toAnswer) {
    loadQuestionSelect();
  } else {
    stkReadingPath = []; stkCurrentQuestion = null;
    renderStkPath();
    if (!bootstrapping) loadReadingQuestion();
  }
}

async function ensureStkMeta(subject) {
  const diffSel = document.getElementById('stkDifficulty');
  const typeSel = document.getElementById('stkType');
  if (stkMetaReadyFor === subject && diffSel && diffSel.options.length) return;
  let res;
  try { res = await API.Stk.status(); } catch (e) { return; }
  const sb = (res.subjects || {})[subject] || {};
  if (diffSel) diffSel.innerHTML = ['全部'].concat(sb.difficulties || []).map((d) => `<option>${d}</option>`).join('');
  if (typeSel) typeSel.innerHTML = ['全部'].concat(sb.types || []).map((t) => `<option>${t}</option>`).join('');
  stkMetaReadyFor = subject;
  return sb;
}

function stkFilters() {
  const seg = document.getElementById('stkSubjectSeg');
  const sub = (seg.querySelector('.seg-btn.active') || {}).dataset ? seg.querySelector('.seg-btn.active').dataset.sub : '物理';
  const diffSel = document.getElementById('stkDifficulty');
  const typeSel = document.getElementById('stkType');
  return {
    sub,
    diff: diffSel && diffSel.value !== '全部' ? diffSel.value : '',
    type: typeSel && typeSel.value !== '全部' ? typeSel.value : '',
  };
}

async function loadReadingQuestion() {
  const area = document.getElementById('stkArea');
  const actions = document.getElementById('stkActions');
  const meta = document.getElementById('stkMeta');
  const f = stkFilters();
  await ensureStkMeta(f.sub);
  document.getElementById('stkResult').classList.add('hidden');
  stkReadingPath = []; renderStkPath();
  if (actions) actions.classList.remove('hidden');
  area.innerHTML = '<div class="loading">加载中...</div>';
  try {
    const res = await API.Stk.random(f.sub, f.diff, f.type);
    if (!res.ok) {
      area.innerHTML = `<div class="error">${res.error || '获取题目失败'}</div>`;
      if (actions) actions.classList.add('hidden');
      if (meta) meta.textContent = '';
      return;
    }
    stkCurrentQuestion = res.question;
    area.innerHTML = renderMathText((res.question.question || '') + (res.question.options && res.question.options.length ? '\n\n' + res.question.options.join('\n') : ''));
    if (meta) meta.textContent = `— ${res.question.ques_type || ''} · ${res.question.difficulty || ''} —`;
  } catch (e) {
    area.innerHTML = '<div class="error">网络错误，请重试</div>';
  }
}

// 审题模式：绑定筛选、换一题、完成审题、查看标准划词（只初始化一次）
function initReadingBank(bootstrapping) {
  const subjectSeg = document.getElementById('stkSubjectSeg');
  if (!subjectSeg || subjectSeg.__stkBound) { if (bootstrapping) loadReadingQuestion(); return; }
  subjectSeg.__stkBound = true;

  const reset = () => { stkReadingPath = []; stkCurrentQuestion = null; renderStkPath(); document.getElementById('stkResult').classList.add('hidden'); };

  subjectSeg.addEventListener('click', async (e) => {
    const btn = e.target.closest('.seg-btn');
    if (!btn) return;
    subjectSeg.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', b === btn));
    stkMetaReadyFor = '';
    await ensureStkMeta(btn.dataset.sub);
    reset();
    loadReadingQuestion();
  });

  const onFilter = () => { reset(); loadReadingQuestion(); };
  document.getElementById('stkDifficulty').addEventListener('change', onFilter);
  document.getElementById('stkType').addEventListener('change', onFilter);
  document.getElementById('btnStkNext').addEventListener('click', () => { reset(); loadReadingQuestion(); });

  document.getElementById('btnStkSubmit').addEventListener('click', () => {
    const std = (stkCurrentQuestion && stkCurrentQuestion.keywords) || [];
    const resultEl = document.getElementById('stkResult');
    if (!std.length) {
      resultEl.innerHTML = '<div class="reading-off">本题暂无标准划词，无法对比。</div>';
      resultEl.classList.remove('hidden');
      return;
    }
    const a = assessReadingQuality(std, stkReadingPath);
    // 先恢复未高亮的题面，再逐词讲评高亮
    document.getElementById('stkArea').innerHTML = renderMathText(
      (stkCurrentQuestion.question || '') + (stkCurrentQuestion.options && stkCurrentQuestion.options.length ? '\n\n' + stkCurrentQuestion.options.join('\n') : '')
    );
    renderStressReview(document.getElementById('stkArea'), resultEl, a, std, stkReadingPath);
  });

  document.getElementById('btnStkShow').addEventListener('click', () => {
    const std = (stkCurrentQuestion && stkCurrentQuestion.keywords) || [];
    const resultEl = document.getElementById('stkResult');
    if (!std.length) {
      resultEl.innerHTML = '<div class="reading-off">本题暂无标准划词。</div>';
      resultEl.classList.remove('hidden');
      return;
    }
    // 恢复题面并仅高亮标准词（中性）
    document.getElementById('stkArea').innerHTML = renderMathText(
      (stkCurrentQuestion.question || '') + (stkCurrentQuestion.options && stkCurrentQuestion.options.length ? '\n\n' + stkCurrentQuestion.options.join('\n') : '')
    );
    highlightReadingKeywords(document.getElementById('stkArea'), std.map((w) => ({ w, cls: 'kw-std' })));
    resultEl.innerHTML = `
      <div class="reading-quality-card good">
        <div class="rq-head">标准划词：${std.map((w) => `<b>${esc(w)}</b>`).join('、')}</div>
      </div>`;
    resultEl.classList.remove('hidden');
  });

  if (bootstrapping) loadReadingQuestion();
}

function initAnswer() {
  // 模式切换：作答 / 审题
  document.getElementById('answerModeSeg').addEventListener('click', (e) => {
    const btn = e.target.closest('.seg-btn');
    if (btn) switchAnswerMode(btn.dataset.mode);
  });
  // 审题模式的筛选与操作（绑定一次，后续复用）
  initReadingBank(false);
  document.querySelectorAll('[data-confidence]').forEach((btn) => {
    btn.addEventListener('click', () => {
      pendingConfidence = Number(btn.dataset.confidence);
      document.querySelectorAll('[data-confidence]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
  document.getElementById('btnSubmit').addEventListener('click', async () => {
    if (!currentQuestion) return;
    const userAnswer = document.getElementById('inputAnswer').value.trim();
    if (!userAnswer) { showModal('提示', '请输入你的答案'); return; }
    const res = await API.Questions.answer({
      question_id: currentQuestion.question_id,
      user_answer: userAnswer,
      confidence_self: pendingConfidence,
    });
    // 仅在提交后才高亮题干关键词（提交前界面不做任何输出）
    await revealKeywords(document.getElementById('answerArea'));
    // 结果持久展示（不再一闪而过）
    renderAnswerResult(res, userAnswer);
    document.getElementById('answerActions').classList.add('hidden');
    // 静默触发读题策略分析，不打断作答
    analyzeReading();
    loadDashboard();
  });
}

// 答题结果：持久面板，清晰呈现对错 / 你的答案 vs 正确答案 / 进步反馈，支持再答或下一题
function renderAnswerResult(res, userAnswer) {
  const el = document.getElementById('answerResult');
  const ok = !!res.is_correct;
  const ap = res.analyses && res.analyses.overconfidence;
  let note = ok
    ? '正确，已记入「已掌握」。'
    : (ap && ap.warn ? '答错且可能存在过度自信，建议复盘错因。' : '对比你的答案与正确答案，理清思路后再继续。');
  // 进步反馈（闪烁状态被持续呈现替代）：连续答对 / 难度 / Boss 等级
  const boss = res.boss || {};
  const streak = boss.consecutive_correct;
  const progress = streak != null
    ? `<div class="answer-result-progress">
        <span class="arp-badge">连续答对 <b>${streak}</b> 题</span>
        <span class="arp-badge">难度 <b>${boss.difficulty != null ? boss.difficulty + '/5' : '—'}</b></span>
        <span class="arp-badge">Boss <b>Lv.${boss.level != null ? boss.level : '—'}</b></span>
      </div>`
    : '';
  const matchCls = ok ? 'match' : 'diff';
  // 涉及知识点（若有）单独成行呈现，便于复盘时定位掌握方向
  const tags = (currentQuestion.knowledge_tags || []).filter(Boolean).map((t) => `<span class="arp-badge arp-tag">${esc(t)}</span>`).join('');
  const tagsRow = tags ? `<div class="answer-result-tags">涉及知识点：${tags}</div>` : '';
  el.innerHTML = `
    <div class="answer-result-card ${ok ? 'ok' : 'no'}">
      <div class="answer-result-verdict">
        <span class="answer-result-icon">${ok ? '✓' : '✕'}</span>
        <div class="answer-result-verdict-text">
          <span class="answer-result-verdict-title">${ok ? '回答正确' : '答错了'}</span>
          <span class="answer-result-verdict-sub">${ok ? '知识点掌握度提升了' : '这道题已记入错题，可复盘后重做'}</span>
        </div>
      </div>
      <div class="answer-result-compare">
        <div class="answer-result-row"><span class="lbl">你的答案</span><span class="val ${matchCls}">${esc(userAnswer)}</span></div>
        <div class="answer-result-row"><span class="lbl">正确答案</span><span class="val">${esc(res.correct_answer)}</span></div>
      </div>
      ${progress}
      ${tagsRow}
      <div class="answer-result-note">${note}</div>
    </div>
    <div class="answer-result-actions">
      <button class="btn btn-secondary" id="btnRetryAnswer">再答一次</button>
      <button class="btn btn-primary" id="btnNextQuestion">下一题</button>
    </div>`;
  el.classList.remove('hidden');
  // 自动滚动到结果区，避免结果落在视口之外被忽略
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  document.getElementById('inputAnswer').value = '';
  document.getElementById('btnRetryAnswer').addEventListener('click', () => {
    hideAnswerResult();
    document.getElementById('answerActions').classList.remove('hidden');
    document.getElementById('inputAnswer').focus();
  });
  document.getElementById('btnNextQuestion').addEventListener('click', () => {
    hideAnswerResult();
    loadQuestionSelect();
  });
}

// 收起答题结果面板并恢复到干净状态
function hideAnswerResult() {
  const el = document.getElementById('answerResult');
  el.classList.add('hidden');
  el.innerHTML = '';
  document.getElementById('readingResult').classList.add('hidden');
}

// ---------- 文章分析 ----------
function initArticle() {
  document.getElementById('btnAnalyze').addEventListener('click', async () => {
    const text = document.getElementById('inputArticle').value.trim();
    if (!text) { showModal('提示', '请先粘贴文章内容'); return; }
    const title = document.getElementById('inputArticleTitle').value.trim();
    const progEl = document.getElementById('kwProgress');
    const resultEl = document.getElementById('kwResult');
    resultEl.classList.add('hidden');
    progEl.classList.remove('hidden');
    const steps = progEl.querySelectorAll('.kw-progress-step');
    steps.forEach((s) => { s.className = 'kw-progress-step'; const i = s.querySelector('.kw-step-info'); if (i) i.remove(); });
    let si = 0;
    const markStep = (info) => {
      if (si < steps.length && steps[si]) {
        steps[si].classList.add('active');
        if (info) steps[si].insertAdjacentHTML('beforeend', `<span class="kw-step-info">${info}</span>`);
        if (si > 0) { steps[si - 1].classList.remove('active'); steps[si - 1].classList.add('done'); }
        si++;
      }
    };
    const finishProg = () => { steps.forEach((s) => { s.classList.remove('active'); s.classList.add('done'); }); si = steps.length; };
    try {
      // 优先流式：每一阶段完成即点亮真实步骤
      await API.Analysis.articleStream(
        text, title,
        (ev) => markStep(`${ev.items}项`),
        (res) => {
          finishProg();
          progEl.classList.add('hidden');
          if (!res || res.status === 'error') { showModal('提示', (res && res.error) || '中心词提取失败'); return; }
          renderKeywordResult(res);
          resultEl.classList.remove('hidden');
          animateKwChips();
        },
        (err) => { finishProg(); progEl.classList.add('hidden'); showModal('提示', err || '中心词提取失败'); },
      );
    } catch (e) {
      // 流式不可用时回退普通接口
      steps.forEach((s) => { s.className = 'kw-progress-step'; const i = s.querySelector('.kw-step-info'); if (i) i.remove(); });
      let st = 0;
      const timer = setInterval(() => { if (st < steps.length) { steps[st].classList.add('active'); if (st > 0) steps[st - 1].classList.add('done'); st++; } }, 400);
      try {
        const res = await API.Analysis.article(text, title);
        clearInterval(timer); steps.forEach((s) => s.classList.add('done'));
        if (res.status === 'error') { showModal('提示', res.error); return; }
        renderKeywordResult(res); resultEl.classList.remove('hidden'); animateKwChips();
      } catch (e2) { clearInterval(timer); showModal('提示', '中心词提取失败'); }
      finally { progEl.classList.add('hidden'); }
    }
  });
}

// 关键词逐个浮现（让结果出来更有"过程感"）
function animateKwChips() {
  const chips = document.querySelectorAll('.kw-chip');
  chips.forEach((c, i) => {
    c.style.opacity = '0'; c.style.transform = 'translateY(6px)'; c.style.transition = 'opacity .35s ease, transform .35s ease';
    setTimeout(() => { c.style.opacity = '1'; c.style.transform = 'none'; }, i * 90);
  });
}

function renderKeywordResult(res) {
  const q = res.quality || {};
  const levelCls = q.level === '较好' ? 'good' : q.level === '中等' ? 'warn' : 'weak';
  document.getElementById('kwQuality').innerHTML =
    `<span class="kw-quality-badge ${levelCls}">${q.level || '—'}</span>
     <span class="kw-quality-tip">${q.assessment || ''}</span>
     <span class="kw-quality-stats">
       平均语义贴合 ${(q.avg_semantic * 100).toFixed(0)}% · 整体置信度 ${(q.avg_confidence * 100).toFixed(0)}%
     </span>`;

  // 中心词 chips：置信度 + 语义贴合 + 频次 + 词源
  const kwEl = document.getElementById('kwKeywords');
  kwEl.innerHTML = '';
  (res.keywords || []).forEach((k, i) => {
    const chip = document.createElement('div');
    chip.className = 'kw-chip';
    chip.innerHTML = `
      <div class="kw-chip-rank">#${i + 1}</div>
      <div class="kw-chip-word">${k.word}<span class="kw-chip-basis">${k.basis || ''}</span></div>
      <div class="kw-chip-metrics">
        <span title="置信度（语义×统计融合）">${Math.round(k.confidence * 100)}%</span>
        <span title="语义贴合度（与全文余弦）">语义 ${(k.semantic_score * 100).toFixed(0)}</span>
        <span title="出现频次">×${k.frequency}</span>
      </div>`;
    kwEl.appendChild(chip);
  });

  // 中心词置信度条形图
  kwBars(document.getElementById('kwConfBars'), res.keywords || []);

  // 模型信息
  const m = res.model || {};
  document.getElementById('kwModel').innerHTML =
    `<div class="kw-detail-title">语义模型</div>
     <div class="kw-model-row"><span>编码器</span><span>${(m.embedder || '').split('/').pop()}</span></div>
     <div class="kw-model-row"><span>嵌入维度</span><span>${m.dim || '—'} 维</span></div>
     <div class="kw-model-row"><span>融合策略</span><span>${m.blend || '—'}</span></div>
     <div class="kw-model-row"><span>方法</span><span>${m.method || '—'}</span></div>`;

  // 分析过程可视化：stepper + 每阶段明细（可折叠开关，默认展开）
  const pipe = res.pipeline || [];
  const doc = res.document || {};
  const stages = [
    { t: '分词', d: (doc.tokens != null ? doc.tokens + ' 词' : `jieba`) },
    { t: '统计 TextRank+PMI', d: (pipe[1] ? pipe[1].items : '') + ' 进语义' },
    { t: '语义精排 MiniLM', d: (pipe[2] ? pipe[2].items : '') + ' 项' },
    { t: '融合排序', d: (res.keywords || []).length + ' 个上屏' },
  ];
  const pipeEl = document.getElementById('kwPipeline');
  const rows = pipe.map((p) => `<div class="proc-row"><span>${esc(p.step)}</span><span>${p.items} 项 · ${p.ms}ms</span></div>`).join('');
  pipeEl.innerHTML =
    `<div class="proc-title">分析过程</div>
     <div class="proc-body">
       ${procStepper(stages)}
       ${rows ? `<div class="proc-rows">${rows}</div>` : ''}
     </div>`;
}

function procStepper(stages) {
  if (!stages || !stages.length) return '';
  return `<div class="proc-stepper">${stages.map((s, i) =>
    `<div class="proc-stage"><b>${esc(s.t)}</b><i>${esc(String(s.d))}</i></div>` +
    (i < stages.length - 1 ? '<span class="proc-arrow">→</span>' : '')).join('')}</div>`;
}

// ---------- 语音分析（F1-3 录音 + F3-3 卡壳检测） ----------
let voiceRec = null;

function initVoice() {
  document.getElementById('btnVoiceRecord').addEventListener('click', startVoiceRecording);
  document.getElementById('btnVoiceStop').addEventListener('click', stopVoiceRecording);
}

async function startVoiceRecording() {
  // 请求麦克风权限并建立音频采集链
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { showModal('提示', '无法访问麦克风，请检查浏览器权限'); return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  voiceRec = { ctx, source, processor, stream, data: [], seconds: 0, timerId: null };
  processor.onaudioprocess = (e) => {
    voiceRec.data.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(ctx.destination);
  document.getElementById('btnVoiceRecord').disabled = true;
  document.getElementById('btnVoiceStop').disabled = false;
  document.getElementById('voiceTimer').textContent = '00:00';
  voiceRec.timerId = setInterval(() => {
    voiceRec.seconds++;
    document.getElementById('voiceTimer').textContent = fmtClock(voiceRec.seconds);
  }, 1000);
}

async function stopVoiceRecording() {
  if (!voiceRec) return;
  clearInterval(voiceRec.timerId);
  const { ctx, source, processor, stream, data } = voiceRec;
  source.disconnect();
  processor.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  document.getElementById('btnVoiceRecord').disabled = false;
  document.getElementById('btnVoiceStop').disabled = true;

  // 合并 PCM → 降采样到 16kHz → 编码 WAV
  const total = data.reduce((s, a) => s + a.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  data.forEach((a) => { merged.set(a, off); off += a.length; });
  const target = 16000;
  const ratio = ctx.sampleRate / target;
  const outLen = Math.max(1, Math.floor(merged.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = merged[Math.floor(i * ratio)];
  const wav = encodeWav(out, target);

  const statusEl = document.getElementById('voiceStatus');
  const resultEl = document.getElementById('voiceResult');
  statusEl.classList.remove('hidden');
  resultEl.classList.add('hidden');
  try {
    const res = await API.Voice.analyze(wav);
    statusEl.classList.add('hidden');
    if (!res.ok) { showModal('提示', res.error || '语音分析失败'); return; }
    renderVoiceResult(res);
  } catch (e) {
    statusEl.classList.add('hidden');
    showModal('提示', '语音分析失败');
  }
}

function fmtClock(s) {
  const m = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${m}:${ss}`;
}

// 将 Float32 PCM 编码为 16-bit 单声道 WAV
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // 单声道
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function renderVoiceResult(res) {
  const el = document.getElementById('voiceResult');
  el.classList.remove('hidden');
  const stallsHtml = res.stalls.length
    ? res.stalls.map(([t, d]) => `<span class="voice-stall">${t}s 停顿 ${d}s</span>`).join('')
    : '<span class="voice-clean">未检测到明显卡壳停顿</span>';
  const modelTag = res.model === 'whisper'
    ? '<span class="voice-model whisper">Whisper 转写</span>'
    : '<span class="voice-model heuristic">能量检测（启发式）</span>';
  const transcriptHtml = res.transcript
    ? `<div class="voice-transcript"><div class="card-tag">转写文本</div><div class="voice-transcript-text">${res.transcript}</div></div>`
    : '';
  el.innerHTML = `
    ${modelTag}
    <div class="voice-metrics">
      <div class="voice-metric"><div class="voice-metric-value">${res.duration}s</div><div class="voice-metric-label">录音时长</div></div>
      <div class="voice-metric"><div class="voice-metric-value">${Math.round(res.speech_ratio * 100)}%</div><div class="voice-metric-label">语音占比</div></div>
      <div class="voice-metric"><div class="voice-metric-value">${res.stall_count}</div><div class="voice-metric-label">卡壳次数</div></div>
    </div>
    <div class="voice-assessment">${res.assessment}</div>
    <div class="voice-stalls">${stallsHtml}</div>
    ${transcriptHtml}
  `;
}

// ---------- 编程刷题（F7-1 漂亮错误检测 + F7-2 压力测试） ----------
function initCoding() {
  document.getElementById('btnCodeAnalyze').addEventListener('click', async () => {
    const code = document.getElementById('inputCode').value.trim();
    if (!code) { showModal('提示', '请先粘贴代码'); return; }
    const tle = document.getElementById('inputCodeTle').checked;
    const el = document.getElementById('codeResult');
    el.innerHTML = '<div class="chart-empty">正在分析…</div>';
    el.classList.remove('hidden');
    try {
      const res = await API.Code.analyze(code, tle);
      if (res.error) { showModal('提示', res.error); el.classList.add('hidden'); return; }
      renderCodeResult(res);
    } catch (e) {
      el.classList.add('hidden');
      showModal('提示', '代码分析失败');
    }
  });
}

function renderCodeResult(res) {
  const el = document.getElementById('codeResult');
  const ec = res.error_class || {};
  const st = res.stress || {};
  const cls = ec.innovative ? 'good' : 'warn';
  const hints = (st.hints || []).map((h) => `<li>${h}</li>`).join('');
  const conf = ec.confidence != null ? `<div class="code-conf">语义置信度：${(ec.confidence * 100).toFixed(1)}%</div>` : '';
  const modelTag = ec.model === 'rule+sbert'
    ? '<span class="code-model">规则 + SBERT</span>'
    : '<span class="code-model rule">规则判定（启发式）</span>';
  el.innerHTML = `
    <div class="code-section">
      <div class="card-tag">错误类型判断</div>
      <div class="code-badge ${cls}">${ec.label}</div>
      ${modelTag}
      ${ec.innovative ? `<div class="code-patterns">写法亮点：${(ec.patterns || []).join(' / ')}</div>` : ''}
      ${conf}
    </div>
    <div class="code-section">
      <div class="card-tag">压力测试建议</div>
      <div class="code-stress-metrics">
        <div class="code-stop"><div class="code-stop-value">${st.scale}</div><div class="code-stop-label">建议规模上限</div></div>
        <div class="code-stop"><div class="code-stop-value">${st.time_limit}ms</div><div class="code-stop-label">时间限制参考</div></div>
      </div>
      <ul class="code-hints">${hints}</ul>
    </div>`;
  el.classList.remove('hidden');
}

// ---------- 分析视图 ----------
async function loadClusters() {
  const res = await API.Analysis.clusters();
  const el = document.getElementById('clusterList');
  const clusters = res.clusters || [];
  const chartEl = document.getElementById('clusterChart');
  if (!clusters.length) {
    el.textContent = '题目不足 5 道'; el.className = 'cluster-list empty';
    chartEl.innerHTML = '<div class="chart-empty">题目不足 5 道</div>';
    return;
  }
  el.className = 'cluster-list';
  el.innerHTML = '';
  donut(chartEl, clusters.map((c) => ({ label: c.cluster_label, value: c.count || Math.round(c.percentage * 100) })));
  clusters.forEach((c) => {
    const item = document.createElement('div');
    item.className = 'cluster-item';
    item.innerHTML = `
      <div class="cluster-head"><span class="cluster-label">${c.cluster_label}</span><span class="cluster-pct">${Math.round(c.percentage * 100)}%</span></div>
      <div class="cluster-kw">${(c.keywords || []).join(' / ')}</div>
    `;
    el.appendChild(item);
  });
}

// ---------- 错因聚类 ----------
async function loadErrorClusters() {
  const res = await API.Analysis.errorClusters();
  const el = document.getElementById('errorClusterList');
  const clusters = res.clusters || [];
  const chartEl = document.getElementById('errorChart');
  if (!clusters.length) {
    el.textContent = res.message || '暂无错题数据'; el.className = 'cluster-list empty';
    chartEl.innerHTML = '<div class="chart-empty">' + (res.message || '暂无错题数据') + '</div>';
    return;
  }
  el.className = 'cluster-list';
  el.innerHTML = '';
  donut(chartEl, clusters.map((c) => ({ label: c.cluster_label, value: c.count || Math.round(c.percentage * 100) })));
  clusters.forEach((c) => {
    const item = document.createElement('div');
    item.className = 'cluster-item';
    item.innerHTML = `
      <div class="cluster-head"><span class="cluster-label">${c.cluster_label}</span><span class="cluster-pct">${Math.round(c.percentage * 100)}%</span></div>
      <div class="cluster-kw">涵盖 ${c.wrong_answer_ids.length} 道错题</div>
    `;
    el.appendChild(item);
  });
}

// ---------- 认知摩擦（悬停权重 + 反常识） ----------
async function loadFrictionSelect() {
  const qRes = await API.Questions.list();
  const list = (qRes.questions || []).filter((q) => q.question_text);
  const sel = document.getElementById('selectFriction');
  sel.innerHTML = '';
  if (!list.length) {
    sel.innerHTML = '<option>暂无题目</option>';
    return;
  }
  list.forEach((q) => {
    const opt = document.createElement('option');
    opt.value = q.question_id;
    opt.textContent = q.question_text.slice(0, 30);
    sel.appendChild(opt);
  });
  sel.onchange = () => loadFriction(sel.value);
  loadFriction(sel.value);
}

async function loadFriction(id) {
  const res = await API.Analysis.friction(id);
  if (res.error) { showModal('提示', res.error); return; }
  document.getElementById('frictionEmpty').classList.add('hidden');
  document.getElementById('frictionBlock').classList.remove('hidden');
  // 悬停权重
  const heList = document.getElementById('hesitationList');
  heList.innerHTML = '';
  const he = Object.entries(res.hesitation || {}).slice(0, 8);
  if (!he.length) { heList.textContent = '暂无线索词'; }
  he.forEach(([word, w]) => {
    const row = document.createElement('div');
    row.className = 'hesitation-row';
    row.innerHTML = `<span>${word}</span><div class="bar"><div class="bar-fill" style="width:${Math.round(w * 100)}%"></div></div><span>${w}</span>`;
    heList.appendChild(row);
  });
  // 反常识词对
  const anList = document.getElementById('antonymList');
  anList.innerHTML = '';
  const ants = res.antonyms || [];
  if (!ants.length) { anList.textContent = '本题未发现反常识词对'; }
  ants.forEach(([a, b]) => {
    const chip = document.createElement('span');
    chip.className = 'keyword-chip';
    chip.textContent = `${a} ↔ ${b}`;
    anList.appendChild(chip);
  });
}

// ---------- 变体生成 ----------
async function loadVariantSelect() {
  const qRes = await API.Questions.list();
  const list = (qRes.questions || []).filter((q) => q.question_text);
  const sel = document.getElementById('selectVariant');
  sel.innerHTML = '';
  if (!list.length) { sel.innerHTML = '<option>暂无题目</option>'; return; }
  list.forEach((q) => {
    const opt = document.createElement('option');
    opt.value = q.question_id;
    opt.textContent = q.question_text.slice(0, 30);
    sel.appendChild(opt);
  });
}

function initVariant() {
  document.getElementById('btnVariantGen').addEventListener('click', async () => {
    const id = document.getElementById('selectVariant').value;
    if (!id) { showModal('提示', '请先选择一道题'); return; }
    const q = questions.find((x) => x.question_id === id) || (await API.Questions.list()).questions.find((x) => x.question_id === id);
    const res = await API.Analysis.variant(id);
    if (res.error) { showModal('提示', res.error); return; }
    document.getElementById('variantOrigBlock').classList.remove('hidden');
    document.getElementById('variantNewBlock').classList.remove('hidden');
    document.getElementById('variantOrig').textContent = q ? q.question_text : '（原题缺失）';
    document.getElementById('variantNew').textContent = res.variant_text;
    const domainMap = { physics: '物理', math: '数学', general: '通用' };
    document.getElementById('variantDomain').textContent = `领域识别：${domainMap[res.domain] || res.domain}`;
    const chEl = document.getElementById('variantChanges');
    chEl.innerHTML = (res.changes || []).map((c) => `<span class="variant-change">${c}</span>`).join('');
    chEl.classList.toggle('hidden', !(res.changes || []).length);
  });
}

// ---------- 易混淆概念 ----------
async function loadConfusion() {
  const res = await API.Analysis.confusion();
  const el = document.getElementById('confusionList');
  const pairs = res.pairs || [];
  if (!pairs.length) { el.textContent = '暂无易混淆概念'; el.className = 'cluster-list empty'; return; }
  el.className = 'cluster-list';
  el.innerHTML = '';
  pairs.forEach(([a, b, sim]) => {
    const item = document.createElement('div');
    item.className = 'cluster-item';
    item.innerHTML = `<div class="cluster-head"><span class="cluster-label">${a} ↔ ${b}</span><span class="cluster-pct">${sim}</span></div>`;
    el.appendChild(item);
  });
}

async function loadGraph() {
  const res = await API.Analysis.graph();
  const sum = document.getElementById('graphSummary');
  const el = document.getElementById('graphArea');
  const hm = document.getElementById('graphHeatmap');
  if (!res.ok) {
    sum.textContent = res.error || '认知图谱未就绪';
    el.textContent = ''; el.className = 'graph-area';
    hm.innerHTML = '<div class="chart-empty">认知图谱未就绪</div>';
    return;
  }
  // 生成易懂的说明文字
  const leftStr = res.left_nodes.slice(0, 4).join('、') + (res.left_nodes.length > 4 ? '…' : '');
  const rightStr = res.right_nodes.join('、');
  sum.innerHTML = `<div class="graph-summary-text">
    <b>认知图谱已就绪</b>：${res.left_nodes.length} 个知识点簇 × ${res.right_nodes.length} 个行为标签
    <div class="graph-summary-detail">
      <b>知识点簇</b>：语义相近的题目归为一组，编号 0~${res.left_nodes.length - 1} 是聚类标签，不代表难度。例：「${leftStr}」
    </div>
    <div class="graph-summary-detail"><b>行为标签</b>：${rightStr}</div>
    <div class="graph-summary-hint">
      <b>看图</b>：左簇到右行的连线<b>越粗</b>越易表现出该行为；矩阵颜色越深、数字越大，关联越强。
    </div>
  </div>`;
  el.className = 'graph-area';
  el.textContent = '';
  bipartite(el, { left: res.left_nodes, right: res.right_nodes, matrix: res.matrix, height: 260 });
  heatmap(hm, { left: res.left_nodes, right: res.right_nodes, matrix: res.matrix });
}

// ---------- 设置：采集模块（滑块开关） ----------
async function loadModules() {
  const res = await API.Modules.status();
  document.querySelectorAll('[data-module]').forEach((input) => {
    input.checked = !!res[input.dataset.module];
  });
}

function initModules() {
  document.querySelectorAll('[data-module]').forEach((input) => {
    input.addEventListener('change', async () => {
      const res = await API.Modules.toggle(input.dataset.module);
      if (res.error) { showModal('提示', res.error); input.checked = !input.checked; return; }
      input.checked = res.enabled;
    });
  });
}

// ---------- 设置：模型中心（就绪状态 + 参数详情） ----------
async function loadTrainingStatus() {
  let res;
  try { res = await API.Monitor.insights(); } catch (e) { showModal('提示', '模型数据加载失败'); return; }
  const models = res.models || [];
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  if (!models.length) { list.innerHTML = '<div class="cluster-list empty">暂无模型信息</div>'; }
  models.forEach((m) => {
    const row = document.createElement('div');
    row.className = 'model-row';
    row.innerHTML = `
      <div><div class="model-name">${m.name}</div><div class="model-code">${m.type || ''}</div></div>
      <div class="model-state ${m.ready ? 'ready' : 'pending'}">
        <span class="status-dot" style="${m.ready ? '' : 'background:var(--warn);animation:none'}"></span>${m.ready ? '已就绪' : '待训练'}
      </div>`;
    list.appendChild(row);
  });
  renderMlCharts(models);
  renderMlDetail(models);
}

// ---------- 学习总览：快捷入口 ----------
function initQuickActions() {
  document.querySelectorAll('.qa-btn').forEach((btn) => {
    btn.addEventListener('click', () => showView(btn.dataset.goto));
  });
}

// ---------- 启动 ----------
async function init() {
  initModal();
  initRouting();
  initEntry();
  initBank();
  initAnswer();
  initReadingCapture();
  initMathEditor();
  initRTFields();
  initRTSelectEdit();
  initEditQuestionControls();
  initArticle();
  initVoice();
  initCoding();
  initModules();
  initVariant();
  initMlDetail();
  initQuickActions();
  showView('analysis-article');  // 主打：进入即用文章分析
}

init();
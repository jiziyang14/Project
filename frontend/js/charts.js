// 图形化组件库：纯 SVG 生成，无外部依赖，随主题（CSS 变量）自适应。
// 供各视图把"数字/列表"升级为"图表"展示。

// 主题图表色（与 main.css :root 保持一致）
const C = {
  1: 'var(--chart-1)',
  2: 'var(--chart-2)',
  3: 'var(--chart-3)',
  4: 'var(--chart-4)',
  5: 'var(--chart-5)',
  6: 'var(--chart-6)',
  7: 'var(--chart-7)',
  8: 'var(--chart-8)',
};

// ---------- 环形仪表盘（认知健康度等 0~100 指标） ----------
export function ring(el, value, { size = 120, stroke = 11, cls = 'accent', label = '', suffix = '' } = {}) {
  const v = Math.max(0, Math.min(100, value == null ? 0 : value));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const filled = c * (v / 100);
  el.innerHTML = `
    <svg viewBox="0 0 ${size} ${size}" class="chart-ring" role="img" aria-label="${label}${v}">
      <circle class="ring-track" cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke-width="${stroke}"/>
      <circle class="ring-fill ring-${cls}" cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none"
        stroke-width="${stroke}" stroke-dasharray="${filled.toFixed(1)} ${c.toFixed(1)}"
        stroke-linecap="round" transform="rotate(-90 ${size / 2} ${size / 2})"/>
      <text class="ring-val" x="${size / 2}" y="${size / 2 + 6}" text-anchor="middle">${v}${suffix}</text>
      ${label ? `<text class="ring-label" x="${size / 2}" y="${size - 6}" text-anchor="middle">${label}</text>` : ''}
    </svg>`;
}

// ---------- 环形饼图（题型/错因/掌握/架构占比） ----------
// legend: 'both' 显示 数量·百分比，'pct' 仅百分比，'count' 仅数量。
// 自动按数值降序排列分段，并分配互不重复的系列色，使其呈现标准饼图形态。
export function donut(el, items, { size = 150, showTotal = true, legend = 'both' } = {}) {
  const sorted = items
    .map((i) => ({ ...i }))
    .filter((i) => (i.value || 0) > 0)
    .sort((a, b) => b.value - a.value);
  const total = sorted.reduce((s, i) => s + i.value, 0);
  if (!total) { el.textContent = '暂无数据'; return; }
  const r = size / 2 - 13;
  const c = 2 * Math.PI * r;
  let acc = 0;
  let segs = '';
  sorted.forEach((it, idx) => {
    const len = c * (it.value / total);
    const col = it.color || C[Math.min(idx, 7) + 1];
    segs += `<circle class="donut-seg" cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none"
      stroke="${col}" stroke-width="15" stroke-dasharray="${len.toFixed(1)} ${(c - len).toFixed(1)}"
      stroke-dashoffset="${(-acc).toFixed(1)}" transform="rotate(-90 ${size / 2} ${size / 2})"/>`;
    acc += len;
  });
  const legendFmt = (it, idx) => {
    const col = it.color || C[Math.min(idx, 7) + 1];
    const pct = Math.round((it.value / total) * 100);
    const val = legend === 'pct' ? `${pct}%` : legend === 'count' ? `${it.value}` : `${it.value} · ${pct}%`;
    return `
      <span class="legend-row"><i class="legend-dot" style="background:${col}"></i>
        <span class="legend-label">${it.label}</span>
        <span class="legend-val">${val}</span></span>`;
  };
  el.innerHTML = `
    <div class="chart-donut-wrap">
      <svg viewBox="0 0 ${size} ${size}" class="chart-donut" role="img">${segs}
        ${showTotal ? `<text class="donut-total" x="${size / 2}" y="${size / 2 + 7}" text-anchor="middle">${total}</text>` : ''}
      </svg>
    </div>
    <div class="donut-legend">${sorted.map(legendFmt).join('')}</div>`;
}

// ---------- 水平条形图（错因分布 / 悬停权重等） ----------
export function hbar(el, items, { maxValue = 1, unit = '', minPct = 2 } = {}) {
  if (!items.length) { el.textContent = '暂无数据'; return; }
  const max = maxValue || Math.max(...items.map((i) => i.value), 1);
  el.innerHTML = items.map((it, idx) => {
    let pct = Math.round(((it.value || 0) / max) * 100);
    if (pct < minPct) pct = minPct; // 保证极小值仍可见
    return `
      <div class="hbar-row">
        <span class="hbar-label">${it.label}</span>
        <div class="hbar-track"><div class="hbar-fill" style="width:${pct}%;background:${it.color || C[(idx % 4) + 1]}"></div></div>
        <span class="hbar-val">${it.value}${unit}</span>
      </div>`;
  }).join('');
}

// ---------- 二部图（知识点 → 行为 关联，认知图谱） ----------
// 自适应容器宽度渲染，节点标签过长时自动截断并附 hover 全名提示。
export function bipartite(el, { left = [], right = [], matrix = [], height = 240 } = {}) {
  if (!left.length || !right.length) { el.textContent = '图谱未就绪'; return; }
  const W = Math.min(420, Math.max(300, el.clientWidth || 340));
  const lx = 60, rx = W - 60;
  const yAt = (i, n) => (n === 1 ? height / 2 : 18 + (i * (height - 36)) / (n - 1));
  const lpos = left.map((_, i) => yAt(i, left.length));
  const rpos = right.map((_, j) => yAt(j, right.length));
  let maxV = 0;
  matrix.forEach((row) => row.forEach((v) => { if (v > maxV) maxV = v; }));
  maxV = maxV || 1;
  let edges = '';
  matrix.forEach((row, i) => row.forEach((v, j) => {
    if (v <= 0) return;
    const kw = 1 + (v / maxV) * 5;
    edges += `<line class="bip-edge" x1="${lx}" y1="${lpos[i]}" x2="${rx}" y2="${rpos[j]}"
      stroke-opacity="${(0.15 + 0.85 * (v / maxV)).toFixed(2)}" stroke-width="${kw.toFixed(1)}"/>`;
  }));
  const short = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const lnodes = left.map((l, i) =>
    `<circle cx="${lx}" cy="${lpos[i]}" r="6" fill="var(--accent)"/>
     <text class="g-node" x="${lx - 10}" y="${lpos[i] + 4}" text-anchor="end"><title>${l}</title>${short(l, 10)}</text>`).join('');
  const rnodes = right.map((r, j) =>
    `<circle cx="${rx}" cy="${rpos[j]}" r="6" fill="var(--accent)"/>
     <text class="g-node" x="${rx + 10}" y="${rpos[j] + 4}" text-anchor="start"><title>${r}</title>${short(r, 10)}</text>`).join('');
  el.innerHTML = `<svg viewBox="0 0 ${W} ${height}" width="${W}" height="${height}" class="chart-graph" role="img">${edges}${lnodes}${rnodes}</svg>`;
}

// ---------- 热力矩阵（认知图谱全量） ----------
export function heatmap(el, { left = [], right = [], matrix = [] } = {}) {
  if (!left.length || !right.length) { el.textContent = '图谱未就绪'; return; }
  let maxV = 0;
  matrix.forEach((row) => row.forEach((v) => { if (v > maxV) maxV = v; }));
  maxV = maxV || 1;
  // 动态尺寸：行高固定、列宽按列数自适应，使矩阵与左侧连线图（约 420px）等宽平衡；
  // 只在列内按原始比例居中显示（不横向拉伸），避免矩阵被拉高变形。
  const cellH = 30, lw = 110, pad = 26, headY = 13; // rows 从 pad 开始，为列头留出上方空间
  const cw = Math.max(...right.map((r) => r.length)) * 9 + 22; // 右列标题区留足空间，避免“卡壳”被截断
  const targetW = 420;
  const cellW = Math.max(46, Math.floor((targetW - lw - cw) / right.length));
  const W = lw + right.length * cellW + cw;
  const H = pad + left.length * cellH + 14;
  let head = right.map((r, j) =>
    `<text class="hm-head" x="${(lw + j * cellW + cellW / 2).toFixed(1)}" y="${headY}" text-anchor="middle">${r}</text>`).join('');
  let rows = '';
  left.forEach((l, i) => {
    const lbl = l.length > 9 ? l.slice(0, 9) + '…' : l;
    const cy = pad + i * cellH + cellH / 2 + 5;
    rows += `<text class="hm-row" x="${lw - 10}" y="${cy.toFixed(1)}" text-anchor="end"><title>${l}</title>${lbl}</text>`;
    right.forEach((_, j) => {
      const v = matrix[i]?.[j] || 0;
      const a = v > 0 ? 0.15 + 0.85 * (v / maxV) : 0.05;
      const cx = lw + j * cellW;
      const cy2 = pad + i * cellH;
      rows += `<rect class="hm-cell" x="${cx.toFixed(1)}" y="${cy2}" width="${cellW - 6}" height="${cellH - 4}"
        rx="5" fill="var(--accent)" fill-opacity="${a.toFixed(2)}"/>`;
      rows += `<text class="hm-celltext" x="${(cx + (cellW - 6) / 2).toFixed(1)}" y="${(cy2 + (cellH - 4) / 2 + 4).toFixed(1)}"
        text-anchor="middle" fill="var(--ink)" font-size="0.62rem">${(v * 100).toFixed(0)}</text>`;
    });
  });
  el.innerHTML = `<div class="hm-scroll"><svg viewBox="0 0 ${W} ${H}" width="${W}" class="chart-heatmap" role="img">${head}${rows}</svg></div>`;
}

// ---------- 关键词置信度条形图 ----------
export function kwBars(el, keywords) {
  if (!keywords || !keywords.length) { el.textContent = '暂无关键词'; return; }
  const max = Math.max(...keywords.map((k) => k.confidence), 0.01);
  el.innerHTML = keywords.map((k, i) => {
    const pct = Math.round((k.confidence / max) * 100);
    return `
      <div class="kwbar-row">
        <span class="kwbar-word">${k.word}</span>
        <div class="kwbar-track"><div class="kwbar-fill" style="width:${pct}%;background:${C[(i % 4) + 1]}"></div></div>
        <span class="kwbar-val">${Math.round(k.confidence * 100)}%</span>
      </div>`;
  }).join('');
}
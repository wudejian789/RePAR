import {mountKnowledge} from './knowledge.js';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value, digits = 2) => Number(value ?? 0).toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const integer = value => Number(value ?? 0).toLocaleString('en-US', {maximumFractionDigits: 0});
const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
const colors = {both:'#3c8a6b', y:'#5585cc', x:'#c29a53', neither:'#9ca6a2'};
const metrics = {service_seconds:['模型计算时间','秒'],wall_seconds:['在线实测时间','秒'],prompt_tokens:['累计输入','tokens'],completion_tokens:['累计输出','tokens'],score:['答题评分','分'],evidence_count:['最终证据片段','个']};
const state = {x:'direct', y:'repar', metric:'service_seconds', log:false, selected:null, visible:[], data:null, detail:null, evidenceMethod:'repar', evidenceID:null, evidenceQuery:'', allRetrieved:false, trace:null, traceMethod:'repar', round:0, traceEvidence:null};
const detailCache = new Map();
const traceCache = new Map();
const isMath = new URLSearchParams(location.search).get('dataset') === 'math';
const dataPath = isMath ? '/data/math' : '/data';
const outcomeLabels = isMath ? {both:'同分且得分',y:'Y 得分较高',x:'X 得分较高',neither:'双方零分'} :
  {both:'双方正确',y:'仅 Y 正确',x:'仅 X 正确',neither:'双方错误'};
let resizeTimer;

async function fetchJSON(url, cache) {
  if (cache?.has(url)) return cache.get(url);
  const promise = fetch(url).then(async response => {
    if (!response.ok) throw new Error(`读取失败 (${response.status}): ${url}`);
    return response.json();
  });
  cache?.set(url, promise);
  try { return await promise; } catch (error) { cache?.delete(url); throw error; }
}
function outcome(item) {
  const x = item.methods[state.x].score, y = item.methods[state.y].score;
  return y > x ? 'y' : x > y ? 'x' : x ? 'both' : 'neither';
}
function methodName(method) { return state.data.methods[method]; }
function axisName(method) {
  if (state.x === state.y) return 'X / Y';
  return method === state.x ? 'X' : 'Y';
}
function terminationLabel(reason) {
  return ({budget:'预算耗尽',model_sufficient:'模型主动结束',no_progress:'无新增信息，停止',no_available_actions:'没有新动作，停止',invalid_plan:'计划无效'})[reason] || reason;
}
function errorKind(error) {
  if (error?.includes('Repeated successful call')) return '重复调用被拦截';
  if (error?.includes('invalid literal') || error?.includes('Unknown entity ID')) return '无效节点 ID';
  if (error?.includes('disabled operator')) return '该接口已禁用';
  return '工具执行失败';
}
function points(value) { return Number.isInteger(value) ? String(value) : Number(value.toFixed(3)).toString(); }
function scoreDot(result) {
  return `<span class="score-dot ${result.fully_correct ? 'correct' : result.score ? 'partial' : ''}" title="${points(result.score)} / ${result.max_score} 分">${points(result.score)}</span>`;
}
function tag(text, type = '') { return `<span class="tag ${type}">${esc(text)}</span>`; }
function metricValue(value) { return state.metric === 'score' ? points(value) : state.metric.endsWith('_tokens') || state.metric === 'evidence_count' ? integer(value) : num(value); }
function saveLocation() {
  const params = new URLSearchParams({case:state.selected || '', x:state.x, y:state.y, metric:state.metric});
  if (state.log) params.set('log', '1');
  history.replaceState(null, '', `#${params}`);
}
function download(name, value, type = 'application/json') {
  const blob = new Blob([typeof value === 'string' ? value : JSON.stringify(value, null, 2)], {type});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function toast(message) {
  $('.success-toast')?.remove(); const box = document.createElement('div');
  box.className = 'success-toast'; box.textContent = message; box.setAttribute('role', 'status'); document.body.append(box);
  setTimeout(() => box.remove(), 3000);
}

async function init() {
  try {
    state.data = await fetchJSON(`${dataPath}/overview.json`);
    state.x = state.data.default_x || 'direct';
    state.y = state.data.default_y || 'repar';
    state.evidenceMethod = state.y;
    if (isMath) state.metric = 'score';
    const params = new URLSearchParams(location.hash.slice(1));
    if (params.has('x') && state.data.methods[params.get('x')]) state.x = params.get('x');
    if (params.has('y') && state.data.methods[params.get('y')]) state.y = params.get('y');
    if (metrics[params.get('metric')]) state.metric = params.get('metric');
    state.log = params.get('log') === '1';
    const options = Object.entries(state.data.methods).map(([id, label]) => `<option value="${id}">${esc(label)}</option>`).join('');
    $('#x-method').innerHTML = options; $('#y-method').innerHTML = options;
    $('#x-method').value = state.x; $('#y-method').value = state.y;
    $('#dataset-title').textContent = state.data.title;
    $('#dataset-summary').textContent = `${state.data.case_count} 道题 · ${Object.keys(state.data.methods).length} 个评测条件`;
    $(isMath ? '#math-link' : '#longbench-link').setAttribute('aria-current', 'page');
    $('#scoring-note').textContent = isMath ?
      '页面显示冻结评测的实验分。开放题使用匿名双评，有分歧时三评，按小问汇总。参考解法可用于人工核对，评分接受等价答案和其他正确解法。' :
      '四选一答案精确匹配。正确为 1 分，错误或无效为 0 分。';
    $('#outcome').innerHTML = '<option value="">全部评分组合</option>' +
      Object.entries(outcomeLabels).map(([id, label]) => `<option value="${id}">${label}</option>`).join('');
    if (isMath) {
      $('#length').hidden = true;
      $('#difficulty').hidden = true;
      $('#domain option').textContent = '全部题型';
      $('#domain').setAttribute('aria-label', '筛选题型');
      $('#metric option[value="score"]').textContent = '答题评分 · 实验分';
    }
    $('#metric').value = state.metric; $('#log-scale').checked = state.log;
    for (const value of [...new Set(state.data.cases.map(x => x.sub_domain))].sort()) {
      $('#domain').insertAdjacentHTML('beforeend', `<option value="${esc(value)}">${esc(value)}</option>`);
    }
    $('#global-status').hidden = true;
    if (state.data.indexing) {
      $('#indexing-note').textContent = `历史 v1 索引首次构建合计 ${num(state.data.indexing.build_seconds, 1)} 秒，覆盖 ${state.data.case_count} 个知识库。`;
    }
    bindEvents();
    state.selected = state.data.cases.some(x => x.id === params.get('case')) ? params.get('case') : state.data.cases[0].id;
    updateComparison();
    await selectCase(state.selected);
  } catch (error) {
    $('#global-status').className = 'status-message error';
    $('#global-status').textContent = `${error.message}。请通过 webserver/server.py 启动页面，确认数据已导出。`;
    $('#global-status').hidden = false;
  }
}

function bindEvents() {
  for (const id of ['x-method','y-method','metric','log-scale']) {
    $('#'+id).addEventListener('change', () => {
      state.x = $('#x-method').value; state.y = $('#y-method').value;
      state.metric = $('#metric').value; state.log = $('#log-scale').checked;
      updateComparison(); saveLocation();
    });
  }
  $('#swap-methods').addEventListener('click', () => {
    [state.x, state.y] = [state.y, state.x]; $('#x-method').value = state.x; $('#y-method').value = state.y;
    updateComparison(); saveLocation();
  });
  for (const id of ['domain','length','difficulty','outcome']) $('#'+id).addEventListener('change', updateComparison);
  $('#search').addEventListener('input', updateComparison);
  $('#reset-filters').addEventListener('click', () => { for (const id of ['search','domain','length','difficulty','outcome']) $('#'+id).value = ''; updateComparison(); });
  $('#legend').addEventListener('click', event => {
    const button = event.target.closest('[data-outcome]'); if (!button) return;
    $('#outcome').value = $('#outcome').value === button.dataset.outcome ? '' : button.dataset.outcome;
    updateComparison();
  });
  $('#case-list').addEventListener('click', event => { const node = event.target.closest('[data-case]'); if (node) selectCase(node.dataset.case); });
  $('#scatter').addEventListener('click', event => {
    const node = nearestPoint(event);
    if (node) selectCase(node.dataset.case);
  });
  $('#scatter').addEventListener('keydown', event => {
    const node = event.target.closest('.plot-point');
    if (node && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); selectCase(node.dataset.case); }
  });
  $('#scatter').addEventListener('pointermove', event => {
    const node = nearestPoint(event);
    if (!node) { $('#tooltip').hidden = true; return; }
    const item = state.data.cases.find(x => x.id === node.dataset.case);
    const t = $('#tooltip');
    t.innerHTML = `<strong>CASE ${String(item.number).padStart(3,'0')} · ${esc(item.sub_domain)}</strong><div class="tooltip-title">${esc(item.question.slice(0,155))}</div><div>X ${esc(methodName(state.x))}: ${metricValue(item.methods[state.x][state.metric])} · ${item.methods[state.x].score} 分</div><div>Y ${esc(methodName(state.y))}: ${metricValue(item.methods[state.y][state.metric])} · ${item.methods[state.y].score} 分</div><div class="tooltip-meta">点击查看题目、预测与检索证据 ↓</div>`;
    t.hidden = false;
    t.style.left = `${Math.max(8, Math.min(event.clientX+15, innerWidth-325))}px`;
    t.style.top = `${Math.max(8, Math.min(event.clientY+15, innerHeight-t.offsetHeight-12))}px`;
  });
  $('#scatter').addEventListener('pointerleave', () => { $('#tooltip').hidden = true; });
  $('#case-section').addEventListener('click', caseClick);
  $('#case-section').addEventListener('input', event => {
    if (event.target.id === 'evidence-search') { state.evidenceQuery = event.target.value; renderEvidenceBody(); }
  });
  $('#case-section').addEventListener('change', event => {
    if (event.target.id === 'all-retrieved') { state.allRetrieved = event.target.checked; state.evidenceID = null; renderEvidenceBody(); }
  });
  $('#export-csv').addEventListener('click', exportCSV);
  $('#jump-case').addEventListener('click', () => {
    $('#case-section').focus({preventScroll: true});
    $('#case-section').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
  });
  $('#close-trace').addEventListener('click', () => $('#trace-dialog').close());
  $('#trace-dialog').addEventListener('click', event => { if (event.target === $('#trace-dialog')) $('#trace-dialog').close(); });
  $('#download-trace').addEventListener('click', () => { if (state.trace) download(`${state.traceMethod}-${state.trace.id || state.trace._id}-trace.json`, state.trace); });
  $('#trace-content').addEventListener('click', event => {
    const round = event.target.closest('[data-round]');
    if (round) { state.round = Number(round.dataset.round); state.traceEvidence = null; renderTrace(); return; }
    const graphToggle = event.target.closest('[data-graph-view]');
    if (graphToggle) { renderEntityTraceGraph(graphToggle.dataset.graphView); return; }
    const entity = event.target.closest('[data-trace-entity]');
    if (entity) { showTraceEntity(entity.dataset.traceEntity); return; }
    const evidence = event.target.closest('[data-trace-evidence]');
    if (evidence) { state.traceEvidence = evidence.dataset.traceEvidence; renderTraceSource(); return; }
  });
  $('#trace-content').addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const entity = event.target.closest('[data-trace-entity]');
    if (entity) { event.preventDefault(); showTraceEntity(entity.dataset.traceEntity); return; }
    const node = event.target.closest('[data-trace-evidence]');
    if (node) { event.preventDefault(); state.traceEvidence = node.dataset.traceEvidence; renderTraceSource(); }
  });
  new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderScatter, 80); }).observe($('#scatter'));
}

function nearestPoint(event) {
  const target = event.target.closest('.plot-point');
  const chart = $('#scatter svg');
  const transform = chart?.getScreenCTM();
  if (!transform) return null;
  const {x, y} = new DOMPoint(event.clientX, event.clientY).matrixTransform(transform.inverse());
  let nearest = null, distance = 14;
  for (const point of $$('.plot-point', chart)) {
    const d = Math.hypot(x - point.cx.baseVal.value, y - point.cy.baseVal.value);
    if (d < distance || (d === distance && point === target)) {
      distance = d;
      nearest = point;
    }
  }
  return nearest;
}

function updateComparison() {
  if (!state.data) return;
  const query = $('#search').value.trim().toLowerCase();
  state.visible = state.data.cases.filter(item => {
    if ($('#domain').value && item.sub_domain !== $('#domain').value) return false;
    if ($('#length').value && item.length !== $('#length').value) return false;
    if ($('#difficulty').value && item.difficulty !== $('#difficulty').value) return false;
    if ($('#outcome').value && outcome(item) !== $('#outcome').value) return false;
    return !query || [item.id,item.question,...Object.values(item.choices),String(item.number)].join(' ').toLowerCase().includes(query);
  });
  $('#filtered-count').textContent = `${state.visible.length} / ${state.data.case_count} 题`;
  const noLog = state.metric === 'score' || state.metric === 'evidence_count';
  $('#log-scale').disabled = noLog;
  if (noLog) { $('#log-scale').checked = false; state.log = false; }
  renderSummary(); renderList(); renderScatter();
  $('#legend').innerHTML = Object.entries(outcomeLabels).map(([id, label]) => `<button class="legend-item ${$('#outcome').value === id ? 'active' : ''}" data-outcome="${id}" aria-pressed="${$('#outcome').value === id}"><span class="legend-dot"></span>${label}</button>`).join('');
  // Colors are applied through CSSOM to keep the static page compatible with CSP.
  $$('.legend-item').forEach(node => $('.legend-dot', node).style.backgroundColor = colors[node.dataset.outcome]);
  if (!state.visible.length) {
    $('#jump-case').disabled = true;
    state.detail = null; state.selected = null;
    $('#case-section').innerHTML = '<div class="empty-state">没有符合筛选条件的题目。调整筛选后继续比较。</div>';
  } else if (!state.visible.some(x => x.id === state.selected)) selectCase(state.visible[0].id);
  else {
    updateNavigation();
    if (state.detail) renderCaseComparison();
  }
}
function renderSummary() {
  const items = state.visible, n = items.length;
  const x = items.reduce((sum, i) => sum+i.methods[state.x].score, 0), y = items.reduce((sum, i) => sum+i.methods[state.y].score, 0);
  const wins = items.filter(i => outcome(i) === 'y').length, losses = items.filter(i => outcome(i) === 'x').length;
  const maximum = items.reduce((sum, item) => sum + (item.max_score ?? 1), 0);
  const delta = isMath ? y-x : n ? 100*(y-x)/n : 0;
  const xs = mean(items.map(i => i.methods[state.x].service_seconds)), ys = mean(items.map(i => i.methods[state.y].service_seconds));
  const summary = (method, axis, score) => {
    const full = items.filter(item => item.methods[method].fully_correct).length;
    const value = isMath ? `${points(score)}<small>/ ${maximum} 分</small>` : `${num(n ? 100*score/n : 0, 1)}<small>%</small>`;
    return `<article class="stat-card"><div class="stat-label">${esc(methodName(method))}<span class="axis-letter">${axis}</span></div>
      <div class="stat-value">${value}</div><div class="stat-foot">${full} / ${n} 题${isMath ? '满分' : '正确'} · 当前筛选</div></article>`;
  };
  $('#summary').innerHTML = summary(state.x, 'X', x) + summary(state.y, 'Y', y) +
    `<article class="stat-card"><div class="stat-label">${isMath ? '实验分差' : '准确率差'} · Y − X<span class="axis-letter">Δ</span></div>
      <div class="stat-value ${delta > 0 ? 'positive' : delta < 0 ? 'negative' : ''}">${delta > 0 ? '+' : ''}${isMath ? points(delta) : num(delta, 1)}<small>${isMath ? '分' : 'pp'}</small></div>
      <div class="stat-foot">Y 领先 ${wins} 题 / X 领先 ${losses} 题</div></article>
      <article class="stat-card"><div class="stat-label">平均模型计算 · X / Y<span class="axis-letter">s</span></div>
      <div class="stat-value">${num(xs)}<small>/ ${num(ys)} 秒</small></div><div class="stat-foot">预填充 + 解码 · 包括原始缓存成本</div></article>`;
}
function renderList() {
  $('#case-list').innerHTML = state.visible.length ? state.visible.map(item => `<button class="case-row ${item.id === state.selected ? 'selected' : ''}" data-case="${item.id}" aria-pressed="${item.id === state.selected}" title="${esc(item.question)}"><div><span class="case-number">CASE ${String(item.number).padStart(3,'0')}</span><div class="case-row-title">${esc(item.question)}</div><span class="case-row-metadata">${esc(item.sub_domain)}${isMath ? ` · ${item.max_score} 分` : ` · ${item.length}`}</span></div><div class="score-pair">${scoreDot(item.methods[state.x])}${scoreDot(item.methods[state.y])}</div></button>`).join('') : '<div class="empty-state">没有匹配的题目</div>';
}
function renderScatter() {
  if (!state.data) return;
  const items = state.visible;
  const size = $('#scatter').clientWidth;
  if (size === 0) return;
  const left = 56, top = 16, side = size - 72;
  const right = left + side, bottom = top + side;
  const isScore = state.metric === 'score';
  const pointsForMetric = items.map(item => ({
    item, x: item.methods[state.x][state.metric], y: item.methods[state.y][state.metric],
  }));
  const values = pointsForMetric.flatMap(({x, y}) => [x, y]);
  let lo = 0, hi, ticks;
  if (isScore && !isMath) {
    lo = -.16;
    hi = 1.16;
    ticks = [0, 1];
  } else if (state.log) {
    const positive = values.filter(value => value > 0);
    const firstPower = Math.floor(Math.log10(positive.length ? Math.min(...positive) : .1));
    const lastPower = Math.max(firstPower + 1, Math.ceil(Math.log10(Math.max(1, ...positive))));
    lo = 10 ** firstPower;
    hi = 10 ** lastPower;
    const skip = Math.ceil((lastPower - firstPower) / (size < 380 ? 4 : 6));
    ticks = [];
    for (let power = firstPower; power < lastPower; power += skip) ticks.push(10 ** power);
    ticks.push(hi);
  } else {
    const max = Math.max(1, ...values);
    const raw = max * 1.09 / (size < 380 ? 4 : 5);
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].find(n => n * power >= raw) * power;
    hi = Math.ceil(max * 1.09 / step) * step;
    ticks = Array.from({length: Math.round(hi / step) + 1}, (_, i) => i * step);
  }
  const scale = value => state.log
    ? (Math.log10(Math.max(value, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
    : (value - lo) / (hi - lo);
  const sx = value => left + scale(value) * side;
  const sy = value => bottom - scale(value) * side;
  const tf = v => v === 0 ? '0' : v >= 1000 ? `${num(v/1000,v%1000 ? 1 : 0)}k` : v >= 1 ? num(v, Number.isInteger(v) ? 0 : 1) : num(v, v < .01 ? 3 : 2);
  const [label, unit] = metrics[state.metric];
  let markup = `<svg viewBox="0 0 ${size} ${size}" role="group" aria-label="X ${esc(methodName(state.x))} 与 Y ${esc(methodName(state.y))} 的${label}散点图">
    <rect class="plot-frame" x="${left}" y="${top}" width="${side}" height="${side}"/>`;
  for (const tick of ticks) {
    markup += `<line class="plot-grid" x1="${sx(tick)}" y1="${top}" x2="${sx(tick)}" y2="${bottom}"/>
      <text class="plot-tick plot-tick-x" x="${sx(tick)}" y="${bottom+19}" text-anchor="middle">${tf(tick)}</text>
      <line class="plot-grid" x1="${left}" y1="${sy(tick)}" x2="${right}" y2="${sy(tick)}"/>
      <text class="plot-tick plot-tick-y" x="${left-9}" y="${sy(tick)+4}" text-anchor="end">${tf(tick)}</text>`;
  }
  markup += `<line class="plot-diagonal" x1="${left}" y1="${bottom}" x2="${right}" y2="${top}"/>
    <text class="plot-axis-label" x="${left+side/2}" y="${size-11}" text-anchor="middle">X · ${label} (${unit})</text>
    <text class="plot-axis-label" transform="translate(13 ${top+side/2}) rotate(-90)" text-anchor="middle">Y · ${label} (${unit})</text>`;
  const grouped = new Map();
  const positionKey = ({x, y}) => `${sx(x)}|${sy(y)}`;
  for (const entry of pointsForMetric) {
    const key = positionKey(entry);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(entry.item.id);
  }
  const positions = pointsForMetric.map(entry => {
    const {item, x, y} = entry;
    const group = grouped.get(positionKey(entry));
    if (group.length === 1) return {item, x: sx(x), y: sy(y)};
    const index = group.indexOf(item.id), cols = Math.ceil(Math.sqrt(group.length));
    const rows = Math.ceil(group.length/cols), dx = (index%cols-(cols-1)/2)*8, dy = (Math.floor(index/cols)-(rows-1)/2)*8;
    const cx = Math.min(right-6-(cols-1)*4, Math.max(left+6+(cols-1)*4,sx(x)));
    const cy = Math.min(bottom-6-(rows-1)*4, Math.max(top+6+(rows-1)*4,sy(y)));
    return {item, x:cx+dx, y:cy+dy};
  });
  for (const p of positions.filter(p => p.item.id !== state.selected)) markup += point(p);
  for (const p of positions.filter(p => p.item.id === state.selected)) markup += `<circle class="point-ring" cx="${p.x}" cy="${p.y}" r="11"/>`+point(p);
  if (!items.length) markup += `<text class="plot-tick" x="${left+side/2}" y="${top+side/2}" text-anchor="middle">没有符合条件的题目</text>`;
  markup += '</svg>'; $('#scatter').innerHTML = markup;
  const notes = ['虚线 X = Y'];
  if ([...grouped.values()].some(group => group.length > 1)) notes.push('重叠点已错开');
  if (state.log && values.includes(0)) notes.push('零值显示在轴下限');
  $('#chart-hint').textContent = notes.join(' · ');
  function point(p) {
    return `<circle class="plot-point ${p.item.id === state.selected ? 'selected' : ''}" cx="${p.x}" cy="${p.y}" r="4.6" fill="${colors[outcome(p.item)]}" data-case="${p.item.id}" tabindex="0" role="button" aria-label="Case ${p.item.number}: ${esc(p.item.question.slice(0,100))}; X ${metricValue(p.item.methods[state.x][state.metric])}; Y ${metricValue(p.item.methods[state.y][state.metric])}"><title>Case ${p.item.number} · ${p.item.id}</title></circle>`;
  }
}

async function selectCase(id) {
  if (!id) return;
  $('#jump-case').disabled = false;
  state.selected = id; state.detail = null; state.evidenceID = null; state.evidenceQuery = ''; state.allRetrieved = false;
  $('#tooltip').hidden = true; saveLocation(); renderList(); renderScatter();
  $('#case-section').innerHTML = '<div class="panel loading">正在读取题目、预测与检索证据…</div>';
  try {
    const item = await fetchJSON(`${dataPath}/cases/${id}.json`, detailCache);
    if (state.selected !== id) return;
    state.detail = item; renderCase();
  } catch (error) {
    if (state.selected === id) $('#case-section').innerHTML = `<div class="case-error">${esc(error.message)}<br><button class="button" data-retry-case="${id}">重试</button></div>`;
  }
}
function renderQuestion(item) {
  const correct = Array.isArray(item.answer) ? item.answer : String(item.answer || '').split('');
  const choices = Object.entries(item.choices).map(([letter, text]) => `<div class="choice ${correct.includes(letter) ? 'answer' : ''}">
    <span class="choice-letter">${esc(letter)}</span><div class="choice-text">${esc(text)}${correct.includes(letter) ? '<span class="choice-mark">✓ 标准答案</span>' : ''}</div></div>`).join('');
  const metadata = isMath ? tag(`满分 ${item.max_score}`) :
    tag(item.length.toUpperCase()) + tag(item.difficulty.toUpperCase());
  let extras = '';
  if (isMath) {
    const data = item.question_data;
    extras += data.tables.map(table => `<div class="math-table"><table><caption>${esc(table.title)}</caption>
      <thead><tr>${table.columns.map(cell => `<th>${esc(cell)}</th>`).join('')}</tr></thead>
      <tbody>${table.rows.map(row => `<tr>${row.map(cell => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`).join('');
    if (data.figures.length) extras += `<details class="raw-answer"><summary>图示文字与数据</summary><pre>${esc(JSON.stringify(data.figures, null, 2))}</pre></details>`;
    if (data.subquestions.length) extras += `<ol class="math-subquestions">${data.subquestions.map(part => `<li value="${esc(part.id.split('.')[0])}"><span>${esc(part.id)}.</span> ${esc(part.question)}</li>`).join('')}</ol>`;
    extras += `<details class="raw-answer" id="reference-solution"><summary>参考解答与评分说明</summary>
      <pre>${esc(item.reference_solution)}</pre><p>${esc(item.scoring_notes)}</p></details>`;
  }
  return `<article class="panel question-panel"><div class="case-tags">${tag(item.sub_domain)}${metadata}${tag(`${integer(item.context_chars)} 字符`)}</div>
    <p class="question-text">${esc(item.question)}</p><div class="choice-grid">${choices}</div>${extras}</article>`;
}

function renderGrade(grade) {
  const judges = (grade.grades || []).map((judge, i) => `<details class="raw-answer"><summary>评审 ${i + 1}</summary>
    ${(judge.units || []).map(unit => `<div class="judge-unit"><strong>小问 ${esc(unit.id)} · ${esc(unit.rating)} · 得分系数 ${esc(unit.credit)}</strong>
      <p>${esc(unit.reason)}</p><ul>${(unit.checks || []).map(check => `<li>${esc(check.claim)} · ${esc(check.verdict)}<br>${esc(check.verification)}</li>`).join('')}</ul></div>`).join('')}
    <p>${esc(judge.reference_check)}</p></details>`).join('');
  const credits = Object.entries(grade.unit_credits || {}).map(([id, credit]) => tag(`小问 ${id} · ${credit}`)).join('');
  return `<details class="raw-answer grade-details"><summary>评分记录${judges ? ` · ${grade.grades.length} 次评审` : ''}</summary>
    <p>${esc(grade.scoring)}</p>${credits ? `<div class="concepts">${credits}</div>` : ''}${judges}</details>`;
}

function renderMethod(method, result) {
  const label = result.fully_correct ? (isMath ? '满分' : '回答正确') : isMath ? (result.score ? '部分得分' : '零分') : '回答错误';
  const model = state.data.models?.[method];
  const prediction = result.pred || (isMath && result.grade?.grades?.length ? '见原始回答' : '—');
  return `<article data-method-card="${method}" class="method-card ${result.fully_correct ? 'correct' : result.score ? 'partial' : 'wrong'}">
    <div class="method-title"><span class="method-label"><span class="axis-letter">${axisName(method)}</span> ${esc(methodName(method))}</span>
      <span class="muted">${points(result.score)} / ${result.max_score}</span></div>
    ${model ? `<p class="method-model">${esc(model)}</p>` : ''}
    <div class="method-pred ${isMath ? 'math-pred' : ''}">${esc(prediction)}<small>${label}</small></div>
    <dl class="method-metrics">
      <div><dt>模型计算</dt><dd>${num(result.service_seconds)} s</dd></div>
      <div><dt>在线实测</dt><dd>${num(result.wall_seconds)} s</dd></div>
      <div><dt>累计输入</dt><dd>${integer(result.prompt_tokens)} tokens</dd></div>
      <div><dt>累计输出</dt><dd>${integer(result.completion_tokens)} tokens</dd></div>
      ${result.final_prompt_tokens === undefined ? '' : `<div><dt>最终输入</dt><dd>${integer(result.final_prompt_tokens)} tokens</dd></div>`}
      <div><dt>LLM 调用</dt><dd>${result.llm_calls} 次</dd></div>
      <div><dt>最终证据</dt><dd>${result.evidence_count} 段</dd></div>
    </dl>
    ${result.cached_calls ? tag(`${result.cached_calls} 次请求复用`,'warning') : ''}
    ${result.truncated ? tag('输入首尾截断','warning') : ''}
    ${result.finish_reason === 'length' ? tag('达到输出上限','warning') : ''}
    <details class="raw-answer final-input" data-final-input="${esc(method)}"><summary>最终模型输入</summary><div class="final-input-body"></div></details>
    <details class="raw-answer model-response"><summary>模型原始回答</summary><pre>${esc(result.response)}</pre></details>
    ${result.grade ? renderGrade(result.grade) : ''}
    ${(result.has_trace ?? method.startsWith('repar')) ? `<button class="button trace-open" data-open-trace="${method}">查看调用 Trace 与可视化</button>` : ''}
  </article>`;
}

async function loadFinalInput(panel, cid) {
  if (!panel.open || panel.dataset.status === 'loading' || panel.dataset.status === 'loaded') return;
  const body = $('.final-input-body', panel);
  panel.dataset.status = 'loading';
  body.textContent = '正在读取最终作答请求…';
  try {
    const method = panel.dataset.finalInput;
    const {messages} = await fetchJSON(`${dataPath}/final-inputs/${encodeURIComponent(method)}/${encodeURIComponent(cid)}.json`);
    if (!panel.isConnected) return;
    body.replaceChildren();
    for (const {role, content} of messages) {
      const label = document.createElement('div');
      label.className = 'input-role';
      label.textContent = role;
      const text = document.createElement('pre');
      text.textContent = content;
      body.append(label, text);
    }
    panel.dataset.status = 'loaded';
  } catch {
    panel.dataset.status = 'error';
    body.textContent = '最终作答请求读取失败，重新展开可重试。';
  }
}

function caseEvidence(method) {
  return state.detail.methods[method].evidence ?? state.detail.evidence;
}

function recordKey(record) { return `${record.id}:${record.start}:${record.end}`; }

function evidenceEntries(method, all) {
  const result = state.detail.methods[method], records = caseEvidence(method);
  if (!result.evidence_records) {
    return (all ? result.retrieved_ids : result.evidence_ids).map(id => [id, records[id], result.evidence_ids.includes(id)]);
  }
  const entries = new Map(result.evidence_records.map(record => [recordKey(record), [record, true]]));
  if (all) {
    for (const record of result.retrieved_records) {
      const key = recordKey(record);
      if (!entries.has(key)) entries.set(key, [record, false]);
    }
  }
  return [...entries].map(([key, [record, final]]) => [key, record, final]);
}

function renderCase() {
  const item = state.detail;
  $('#case-section').innerHTML = `<div class="case-topbar"><div class="case-id"><h2>第 ${item.number} 题</h2><code>${esc(item.id)}</code></div>
    <div class="case-nav"><button class="button" id="prev-case" aria-label="上一道题">←</button><span id="case-position"></span>
    <button class="button" id="next-case" aria-label="下一道题">→</button><button class="button" id="download-case">↓ Case JSON</button></div></div>
    ${renderQuestion(item)}<div id="case-comparison"></div>`;
  updateNavigation();
  renderCaseComparison();
}

function renderCaseComparison() {
  const host = $('#case-comparison');
  const pair = [...new Set([state.x, state.y])];
  const pairKey = JSON.stringify([state.x, state.y]);
  if (host.dataset.pair === pairKey) return;
  host.dataset.pair = pairKey;

  const item = state.detail;
  const evidenceMethods = pair.filter(method => item.structures?.[method] ||
    item.methods[method].retrieved_ids.length > 0 || item.methods[method].evidence_ids.length > 0);
  if (!evidenceMethods.includes(state.evidenceMethod)) {
    state.evidenceMethod = evidenceMethods.includes(state.y) ? state.y : evidenceMethods[0];
    state.evidenceID = null;
    state.evidenceQuery = '';
    state.allRetrieved = false;
  }
  const knowledgeMethods = pair.filter(method => item.structures?.[method]);
  const knowledgeOpen = $('#knowledge-panel')?.open;
  const lastKnowledgeMethod = $('#knowledge-method')?.value;
  const knowledgeMethod = knowledgeMethods.includes(lastKnowledgeMethod) ? lastKnowledgeMethod : state.evidenceMethod;
  const evidencePicker = evidenceMethods.length > 1 ?
    `<div class="tab-group">${evidenceMethods.map(method => `<button class="tab-button ${method === state.evidenceMethod ? 'active' : ''}"
      data-evidence-method="${method}" aria-pressed="${method === state.evidenceMethod}">${axisName(method)} · ${esc(methodName(method))}</button>`).join('')}</div>` :
    `<span class="evidence-method-label">${axisName(state.evidenceMethod)} · ${esc(methodName(state.evidenceMethod))}</span>`;
  host.innerHTML = `<div class="method-comparison">${pair.map(method => renderMethod(method, item.methods[method])).join('')}</div>
    ${evidenceMethods.length ? `
    <section class="panel evidence-panel"><div class="section-header"><h2>实际检索证据</h2><span class="muted" id="evidence-total"></span></div>
      <div class="evidence-toolbar">${evidencePicker}
      <div class="evidence-tools"><label class="evidence-scope"><input type="checkbox" id="all-retrieved"> 包括召回但未入最终上下文</label>
      <input type="search" id="evidence-search" aria-label="搜索检索证据" placeholder="搜索原文或片段 ID"></div></div>
      <div class="evidence-grid"><div class="evidence-index" id="evidence-index"></div><div class="evidence-reading" id="evidence-reading"></div></div></section>` : ''}
    ${knowledgeMethods.length ? `<details class="panel knowledge-panel" id="knowledge-panel"><summary>知识库检查 · 树、图与原文分块</summary>
      <div id="knowledge-content"></div></details>` : ''}`;
  for (const panel of $$('[data-final-input]', host)) {
    panel.addEventListener('toggle', () => loadFinalInput(panel, item.id));
  }
  if (evidenceMethods.length) {
    $('#evidence-search').value = state.evidenceQuery;
    $('#all-retrieved').checked = state.allRetrieved;
    renderEvidenceBody();
  }
  if (knowledgeMethods.length) {
    const panel = $('#knowledge-panel');
    const labels = Object.fromEntries(knowledgeMethods.map(method => [method, `${axisName(method)} · ${methodName(method)}`]));
    mountKnowledge(panel, item, labels, fetchJSON, knowledgeMethod);
    panel.open = Boolean(knowledgeOpen);
  }
}
function updateNavigation() {
  if (!$('#case-position')) return;
  const pos = state.visible.findIndex(x => x.id === state.selected);
  $('#case-position').textContent = `${pos+1} / ${state.visible.length}`;
  $('#prev-case').disabled = pos <= 0; $('#next-case').disabled = pos < 0 || pos >= state.visible.length-1;
}
function caseClick(event) {
  const trace = event.target.closest('[data-open-trace]'); if (trace) { openTrace(trace.dataset.openTrace); return; }
  const method = event.target.closest('[data-evidence-method]');
  if (method) {
    state.evidenceMethod = method.dataset.evidenceMethod;
    state.evidenceID = null;
    for (const button of $$('[data-evidence-method]')) {
      const active = button.dataset.evidenceMethod === state.evidenceMethod;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    renderEvidenceBody();
    return;
  }
  const eid = event.target.closest('[data-evidence-id]'); if (eid) { state.evidenceID = eid.dataset.evidenceId; renderEvidenceBody(); return; }
  if (event.target.closest('#prev-case') || event.target.closest('#next-case')) {
    const pos = state.visible.findIndex(x => x.id === state.selected), step = event.target.closest('#prev-case') ? -1 : 1;
    if (state.visible[pos+step]) selectCase(state.visible[pos+step].id); return;
  }
  if (event.target.closest('#download-case') && state.detail) download(`case-${state.selected}.json`, state.detail);
  const copy = event.target.closest('[data-copy-evidence]');
  if (copy) {
    const entry = evidenceEntries(state.evidenceMethod, state.allRetrieved).find(([key]) => key === copy.dataset.copyEvidence);
    if (entry) navigator.clipboard.writeText(entry[1].text).then(() => toast('已复制原文证据')).catch(() => toast('复制失败，请手动选择原文'));
  }
  const retry = event.target.closest('[data-retry-case]'); if (retry) selectCase(retry.dataset.retryCase);
}
function renderEvidenceBody() {
  if (!state.detail || !$('#evidence-index')) return;
  const all = evidenceEntries(state.evidenceMethod, true);
  const finalCount = all.filter(([, , final]) => final).length;
  const q = state.evidenceQuery.trim().toLowerCase();
  const shown = all.filter(([, e, final]) => e && (state.allRetrieved || final)
    && (!q || `${e.id} ${e.knowledge_id || ''} ${(e.entity_ids || []).join(" ")} ${(e.entities || []).map(x => x.name).join(" ")} ${e.text}`.toLowerCase().includes(q)));
  if (!shown.some(([key]) => key === state.evidenceID)) state.evidenceID = shown[0]?.[0] ?? null;
  $('#evidence-total').textContent = `显示 ${shown.length} 段 · 最终入选 ${finalCount} / 共展示 ${all.length} 段`;
  $('#evidence-total').style.fontSize = '10px';
  $('#evidence-index').innerHTML = shown.length ? shown.map(([key, e, final]) => `<button class="evidence-item ${key === state.evidenceID ? 'active' : ''}"
    data-evidence-id="${esc(key)}" aria-pressed="${key === state.evidenceID}"><strong>${esc(e.id)}</strong>
    <span class="evidence-item-meta">${final ? '已入选' : '仅召回'}</span><p>${esc(e.text.slice(0,150))}</p></button>`).join('') :
    '<div class="empty-state">没有匹配的证据</div>';
  const e = shown.find(([key]) => key === state.evidenceID)?.[1];
  $('#evidence-reading').innerHTML = e ? `<div class="evidence-head"><div><h3>${esc(e.id)} <span class="muted">${esc(e.knowledge_id)}</span></h3>
    <div class="evidence-location">字符区间 ${integer(e.start)} — ${integer(e.end)}${e.tree_path ? `<br>树路径 ${esc(e.tree_path.join(' / '))}` : ''}</div>
    </div><button class="button quiet" data-copy-evidence="${esc(state.evidenceID)}">复制原文</button></div>
    <div class="concepts">${(e.concepts || []).map(c => tag(c)).join('')}</div><pre class="evidence-text">${esc(e.text)}</pre>
    ${entityKnowledge(e)}
    ${e.claims?.length ? `<details class="knowledge-details"><summary>抽取命题与关联</summary>
    ${e.claims.map(c => `<div class="knowledge-claim">${esc(c.text)}<small>字符位置 ${c.start} — ${c.end}</small></div>`).join('')}
    <div class="concepts">${e.links.map(link => tag(`${e.id} → ${link.target} · ${link.relation}`)).join('')}</div></details>` : ''}` :
    '<div class="empty-state">此条件没有匹配的证据片段。</div>';
}

async function openTrace(method) {
  state.traceMethod = method; state.trace = null; state.round = 0; state.traceEvidence = null;
  const id = state.selected;
  $('#trace-title').textContent = `${methodName(method)} · Case ${String(state.detail.number).padStart(3,'0')} 调用轨迹`;
  $('#trace-content').innerHTML = '<div class="loading">正在读取完整调用轨迹…</div>';
  $('#download-trace').disabled = true; $('#trace-dialog').showModal();
  try {
    const trace = await fetchJSON(`${dataPath}/traces/${method}/${id}.json`, traceCache);
    if (state.selected !== id || state.traceMethod !== method) return;
    state.trace = trace; $('#download-trace').disabled = false; renderTrace();
  } catch (error) { $('#trace-content').innerHTML = `<div class="case-error">${esc(error.message)}</div>`; }
}
function stepCalls(step) {
  if (step?.fallback) return [{name:'tree_search', parameters:{query:state.detail.question, top_k:12}, observation:step.observation, new_evidence:step.observation?.results?.map(x => x.id) || []}];
  return step?.executed || [];
}
function traceRecords() {
  return state.trace.evidence_records ?? state.trace.evidence_ids.map(id => ({id}));
}
function traceRecordKey(record) {
  return state.trace.id && record.start !== undefined ? recordKey(record) : record.id;
}
function renderTrace() {
  if (!state.trace) return;
  const trace = state.trace, steps = trace.trace, final = state.round === steps.length;
  const step = steps[state.round], calls = stepCalls(step);
  const count = steps.reduce((n, s) => n+stepCalls(s).length, 0);
  const errors = steps.flatMap(stepCalls).filter(c => c.error).length;
  $('#trace-content').innerHTML = `<div class="trace-summary">${tag(`${steps.filter(x => x.round).length} 轮控制`)}${tag(`${count} 次调用尝试`)}${tag(`${errors} 次工具错误`, errors ? 'warning' : '')}${tag(`${trace.evidence_ids.length} 段最终证据`)}${tag(terminationLabel(trace.termination), terminationLabel(trace.termination))}${trace.fallback ? tag('发生确定性回退','warning') : ''}</div><p class="trace-policy-note">点击轮次查看真实的计划、反思、参数和执行结果；点击证据节点回读原文。错误调用和重复查询保留在轨迹中。</p><div class="round-strip">${steps.map((s,i) => {const c = stepCalls(s), e = c.filter(x => x.error).length; return `<button class="round-node ${i === state.round ? 'active' : ''} ${e ? 'has-error' : ''}" data-round="${i}" aria-pressed="${i === state.round}"><span class="round-label">${s.fallback ? 'FALLBACK' : `ROUND ${s.round}`}</span><strong>${s.fallback ? '树查询回退' : c.map(x => x.name || x.call?.name || 'error').join(' + ') || '停止 / 无调用'}</strong><small>${c.length} 次尝试 · ${e} 错误 · ${c.reduce((n,x) => n+(x.new_evidence?.length || 0),0)} 新证据</small></button>`;}).join('')}<button class="round-node final-node ${final ? 'active' : ''}" data-round="${steps.length}" aria-pressed="${final}"><span class="round-label">FINAL ANSWER</span><strong>答案 ${esc(state.detail.methods[state.traceMethod].pred || '—')} · ${state.detail.methods[state.traceMethod].score} / 1</strong><small>基于 ${trace.evidence_ids.length} 段原文作答</small></button></div><div class="trace-workspace"><section class="trace-visual"><h3>${final ? '最终作答证据' : '工具执行与证据流'}</h3><div class="trace-graph" id="trace-graph"></div><p class="trace-graph-caption">绿色：工具 / 蓝色：原文证据 / 橙色虚线：执行错误。图中连线表示真实调用返回。</p><div class="trace-source" id="trace-source"></div></section><section class="trace-step"><h3>${final ? '最终作答' : step.fallback ? '回退步骤' : `Round ${step.round} · 执行详情`}</h3><div class="trace-step-content">${final ? finalDetails() : roundDetails(step,calls)}</div></section></div>`;
  renderTraceGraph(step, calls, final); renderTraceSource();
  if (trace.state && Object.hasOwn(trace.state, 'direct_answer')) {
    const draft = trace.state.direct_answer || '模型未返回草稿';
    $('.trace-policy-note', $('#trace-content')).insertAdjacentHTML('afterend',
      `<details class="raw-answer trace-draft"><summary>先验草稿 · 检索前回答</summary><pre>${esc(draft)}</pre></details>`);
  }
  const result = state.detail.methods[state.traceMethod];
  $('.final-node strong').textContent = `评分 ${points(result.score)} / ${result.max_score}`;
  if (step?.protocol && traceGraphData().entities.size) {
    $('.trace-visual h3').insertAdjacentHTML('afterend', '<div class="trace-view-tabs"><button class="tab-button active" data-graph-view="flow">调用与原文流</button><button class="tab-button" data-graph-view="entities">实体关系图</button></div>');
  }
}
function roundDetails(step, calls) {
  const d=step.decision || {};
  const block=(label,value) => `<div class="reflection-block"><label>${label}</label><p>${esc(value || '未提供')}</p></div>`;
  const raw=(label,value) => `<details><summary>${label}</summary><pre class="json-block">${esc(typeof value==='string' ? value : JSON.stringify(value,null,2))}</pre></details>`;
  let html=step.error ? `<p class="tool-error-text">${esc(step.error)}</p>` : '';
  if (step.protocol) {
    html+='<div class="phase-sequence"><span>1 PLAN</span><b>→</b><span>2 ACTION</span><b>→</b><span>3 OBSERVATION</span><b>→</b><span>4 REFLECTION</span></div>';
    html+=block('PLAN / 执行前计划',step.plan?.objective)+block('QUERY / 当前信息缺口查询',step.plan?.query);
    html+='<p class="trace-policy-note">下面先展示实际执行，再展示收到结果后的反思。三个阶段由独立模型调用生成。</p>';
  } else if (step.fallback) html+=block('FALLBACK REASON',step.reason);
  else {
    html+=block('REFLECTION / v1 模型自述',d.reflection)+block('PLAN / v1 计划',d.plan);
    html+=block('POLICY UPDATE',Object.entries(d.policy_update || {}).map(([k,v]) => `${k}: ${v}`).join('\n'));
    html+=block('CONTROL',`finish: ${Boolean(d.finish)}；${step.termination==='model_sufficient' ? '主动结束' : '继续按预算执行'}${d.finish && d.tool_calls?.length ? '（工具调用优先于 finish）' : ''}`);
  }
  calls.forEach((c,i) => {
    const name=c.name || c.call?.name || 'unknown',parameters=c.parameters || c.call?.parameters || {};
    const o=c.observation || {},results=o.results || [];
    html+=`<div class="tool-call ${c.error ? 'error' : ''}"><div class="tool-call-title"><strong>${i+1}. ${esc(name)}</strong><span class="tool-call-status">${c.error ? errorKind(c.error) : `返回 ${results.length} 段 / 新增 ${c.new_evidence?.length || 0} 段`}</span></div><pre class="json-block">${esc(JSON.stringify(parameters,null,2))}</pre>`;
    if (c.error) html+=`<p class="tool-error-text">${esc(c.error)}</p>`;
    if (o.visited_tree_nodes) html+=`<p class="evidence-location">遍历 ${o.visited_tree_nodes} 个树节点 · scope ${esc(o.scope)}</p>`;
    if (o.seeds) html+=`<p class="evidence-location">图种子 ${esc(o.seeds.join(', '))}</p>`;
    if (o.entities) html+=`<p class="evidence-location">返回 ${o.entities.length} 个实体 · ${o.relations?.length || 0} 条关系</p>`;
    if (c.tool_seconds!==undefined) html+=`<p class="evidence-location">工具执行 ${(c.tool_seconds*1000).toFixed(2)} ms</p>`;
    html+=`<div class="evidence-chips">${results.map(r => `<button class="evidence-chip ${c.new_evidence?.includes(r.id) ? 'new' : ''}" data-trace-evidence="${esc(traceRecordKey(r))}">${esc(r.id)}${c.new_evidence?.includes(r.id) ? ' +' : ''}</button>`).join('')}</div>`;
    if (o.operator==='tree_browse') html+=raw('树目录返回结果',o);
    if (c.visible_observation) html+=raw('本次实际交给控制器的观察',c.visible_observation);
    if (o.relations?.length) html+=raw('实体关系与跨片段路径',{relations:o.relations,paths:o.paths,propositions:o.propositions});
    if (results.some(r => r.edge)) html+=raw('v1 片段桥接关系',results.filter(r => r.edge).map(r => ({id:r.id,edge:r.edge})));
    html+='</div>';
  });
  if (step.protocol) {
    html+=block('REFLECTION / 执行后证据评估',step.reflection?.assessment);
    html+=block('GAP / 尚缺的信息',step.reflection?.gap)+block('LESSON / 下轮策略',step.reflection?.lesson);
    html+=block('NEXT',step.reflection?.next==='answer' ? '基于证据回答' : step.decision?.finish ? '本轮选择直接回答' : '进入下一轮规划');
    if (step.plan_raw !== undefined) html+=raw('Plan 原始输出',step.plan_raw);
    if (step.action_raw !== undefined) html+=raw('Action 原始输出及候选参数',{output:step.action_raw,available_actions:step.available_actions});
    if (step.reflection_raw !== undefined) html+=raw('执行后 Reflection 原始输出',step.reflection_raw);
    if (step.progress) html+=raw('执行进展',step.progress);
    const attempts=(state.detail.methods[state.traceMethod].phase_attempts || []).filter(a =>
      new RegExp(`_${step.round}(?:_repair2?)?$`).test(a.tag));
    if (attempts.length) html+=raw('全部阶段生成尝试（含格式修复与截断）',attempts);
  } else html+=raw('模型原始输出',step.raw || '此步骤无模型输出');
  return html+raw('完整执行步骤 JSON',step)+raw('取证前的证据记忆',step.evidence_before || []);
}

function finalDetails() {
  const m = state.detail.methods[state.traceMethod];
  const descriptions={budget:'到达最大轮数，进入最终作答。该标记不表示证据已完整。',model_sufficient:'控制器根据观察选择进入回答。',no_progress:'连续动作未增加原文或实体，执行器停止继续取证；仍可能存在信息缺口。',no_available_actions:'没有尚未完成的可用动作，停止取证。'};
  const scoreDetail = isMath ? `实验分 ${points(m.score)} / ${m.max_score}` :
    `预测 ${esc(m.pred)} / 标准答案 ${esc(state.detail.answer)} → ${m.score} 分`;
  const sources = traceRecords().map(record =>
    `<button class="evidence-chip" data-trace-evidence="${esc(traceRecordKey(record))}">${esc(record.id)}${record.start === undefined ? '' : ` · ${record.start}–${record.end}`}</button>`).join('');
  return `<div class="reflection-block"><label>MODEL RESPONSE</label><p>${esc(m.response)}</p></div><div class="reflection-block"><label>评分</label><p>${scoreDetail}</p></div><div class="reflection-block"><label>TERMINATION · ${esc(terminationLabel(state.trace.termination))}</label><p>${esc(descriptions[state.trace.termination] || state.trace.termination)}</p></div><div class="evidence-chips">${sources}</div><details><summary>本方法全部 LLM 调用记录</summary><pre class="json-block">${esc(JSON.stringify(m.calls,null,2))}</pre></details>`;
}
function renderTraceGraph(step, calls, final) {
  const w = 590;
  const sources = final ? traceRecords() : calls.flatMap(c => c.observation?.results || []);
  const groups = final ? [{name:'最终作答', results:sources}] : calls.map(c => ({name:c.name || c.call?.name || 'unknown', results:c.observation?.results || [], error:c.error, observation:c.observation}));
  if (!groups.length) groups.push({name:step?.termination ? '主动结束' : '无工具调用',results:[],error:step?.error});
  let cursor = 30, nodes = '', edges = '';
  for (const group of groups) {
    const rows = Math.max(1,Math.ceil(group.results.length/2)), block = Math.max(85,rows*37);
    const cy = cursor+block/2;
    nodes += node(24,cy-20,145,40,group.name,group.error ? 'error' : final ? 'final' : 'tool');
    if (group.error) {
      nodes += node(340,cy-19,170,38,errorKind(group.error)+' → 反馈','error');
      edges += path(169,cy,340,cy,true);
    } else if (!group.results.length) {
      const text = group.observation?.operator === 'tree_browse' ? `${group.observation.children?.length || 0} 个目录节点` : '没有返回原文片段';
      nodes += node(340,cy-19,170,38,text,''); edges += path(169,cy,340,cy);
    }
    group.results.forEach((record,i) => {
      const x = 320+(i%2)*122, y = cursor+18+Math.floor(i/2)*37;
      nodes += node(x,y,102,27,record.knowledge_id ? `${record.id} · ${record.knowledge_id}` : record.id,'evidence',traceRecordKey(record));
      edges += path(169,cy,x,y+13.5);
    });
    cursor += block+27;
  }
  const height = Math.max(190,cursor+8);
  $('#trace-graph').innerHTML = `<svg viewBox="0 0 ${w} ${height}" aria-label="真实工具调用到原文证据的流转图"><defs><marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L6 3L0 6" fill="#a8bd9c"/></marker></defs>${edges}${nodes}</svg>`;
  function path(x1,y1,x2,y2,error=false) { return `<path class="flow-edge ${error ? 'error' : ''}" d="M${x1},${y1} C${x1+70},${y1} ${x2-45},${y2} ${x2},${y2}" marker-end="url(#arrow)"/>`; }
  function node(x,y,width,height,text,type,id) { return `<g class="flow-node ${type}" ${id ? `data-trace-evidence="${id}" tabindex="0" role="button" aria-label="读取证据 ${id}"` : ''}><rect x="${x}" y="${y}" width="${width}" height="${height}" rx="6"/><text x="${x+width/2}" y="${y+height/2+3}" text-anchor="middle">${esc(text)}</text></g>`; }
}
function renderTraceSource() {
  if (!$('#trace-source')) return;
  const id = state.traceEvidence;
  const step = state.trace.trace[state.round];
  const records = step ? stepCalls(step).flatMap(call => call.observation?.results || []) : traceRecords();
  const observed = records.find(record => record.text !== undefined && traceRecordKey(record) === id);
  const e = state.detail && (observed || caseEvidence(state.traceMethod)[id]);
  $$('.flow-node.evidence').forEach(n => n.classList.toggle('selected',n.dataset.traceEvidence === id));
  $('#trace-source').innerHTML = e ? `<h3>${esc(e.id)}</h3><p class="evidence-location">字符 ${integer(e.start)} — ${integer(e.end)}<br>${esc(e.tree_path?.join(' / '))}</p><pre class="evidence-text">${esc(e.text)}</pre>` : '<p>点击蓝色证据节点或右侧片段 ID，查看原文与树路径。</p>';
}

function entityKnowledge(e) {
  if (!e.entities?.length) return '';
  const entities = new Map(e.entities.map(entity => [entity.id,entity]));
  const name = id => entities.get(id)?.name || id;
  return `<details class="knowledge-details entity-knowledge"><summary>实体图：${e.entities.length} 个实体 · ${e.entity_relations?.length || 0} 条关系</summary><p class="evidence-location">C 是原文片段，E 是跨片段复用的实体，P 是原文陈述，R 是自动抽取的关系。关系尚未经人工验证。</p><div class="entity-chips">${e.entities.map(entity => `<div class="entity-pill"><strong>${esc(entity.id)} · ${esc(entity.name)}</strong><small>${esc(entity.type)} · ${entity.source_count || entity.passage_ids.length} 个原文片段</small><span>${esc(entity.passage_ids.slice(0,12).join(', '))}${entity.passage_ids.length>12 ? '…' : ''}</span></div>`).join('')}</div><div class="relation-list">${(e.entity_relations || []).slice(0,40).map(r => `<div class="relation-row"><span>${esc(name(r.subject))}</span><b>→ ${r.negated ? 'NOT ' : ''}${esc(r.modality?.join(' ') || '')} ${esc(r.predicate)} →</b><span>${esc(name(r.object))}</span><small>${esc(r.id)} · ${esc(r.kind)} · ${esc(r.time?.join(', ') || '')} · 来源 ${esc(r.proposition_ids.slice(0,4).join(', '))}</small></div>`).join('')}</div><details class="raw-answer"><summary>原文陈述与实体映射</summary><pre class="json-block">${esc(JSON.stringify(e.propositions,null,2))}</pre></details></details>`;
}

function traceGraphData() {
  const step = state.trace.trace[state.round];
  const observations = stepCalls(step).map(c => c.observation || {});
  const entities = new Map(), relations = new Map();
  for (const observation of observations) {
    for (const entity of observation.entities || []) entities.set(entity.id,entity);
    for (const relation of observation.relations || []) relations.set(relation.id,relation);
    for (const record of observation.results || []) {
      const e = state.trace.id ? record : caseEvidence(state.traceMethod)[record.id];
      for (const entity of e?.entities || []) entities.set(entity.id,entity);
      for (const relation of e?.entity_relations || []) relations.set(relation.id,relation);
    }
  }
  return {entities,relations};
}

function renderEntityTraceGraph(view) {
  $$('.trace-view-tabs .tab-button').forEach(node => node.classList.toggle('active',node.dataset.graphView===view));
  const step = state.trace.trace[state.round];
  if (view === 'flow') { renderTraceGraph(step,stepCalls(step),state.round===state.trace.trace.length); return; }
  const {entities,relations} = traceGraphData();
  if (!entities.size) { $('#trace-graph').innerHTML='<div class="empty-state">本轮没有返回实体或实体关系。</div>'; return; }
  const selected = [...relations.values()].filter(r => entities.has(r.subject) && entities.has(r.object)).slice(0,22);
  let ids = [...new Set(selected.flatMap(r => [r.subject,r.object]))];
  if (!ids.length) ids = [...entities.keys()].slice(0,16);
  const w=680,cols=Math.min(4,Math.ceil(Math.sqrt(ids.length))),h=Math.max(310,Math.ceil(ids.length/cols)*68+50);
  const positions=new Map(ids.map((id,i) => {
    return [id,{x:w/cols*(i%cols+.5),y:42+Math.floor(i/cols)*68}];
  }));
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="group" aria-label="跨片段实体关系图"><defs><marker id="entity-arrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L6 3L0 6" fill="#98b19c"/></marker></defs>`;
  selected.forEach((r,i) => {
    const a=positions.get(r.subject),b=positions.get(r.object),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy));
    const extent=Math.min(68/(Math.abs(dx/d)||1e-9),20/(Math.abs(dy/d)||1e-9));
    const x1=a.x+dx/d*extent,y1=a.y+dy/d*extent,x2=b.x-dx/d*extent,y2=b.y-dy/d*extent;
    svg+=`<g class="entity-edge ${r.kind === 'co_mention' ? 'weak' : ''}"><title>${esc(r.id)}: ${esc(entities.get(r.subject).name)} → ${r.negated ? 'NOT ' : ''}${esc(r.predicate)} → ${esc(entities.get(r.object).name)}; 来源 ${esc(r.proposition_ids.join(', '))}</title><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#entity-arrow)"/><text x="${(x1+x2)/2}" y="${(y1+y2)/2+(i%3-1)*10}" text-anchor="middle">${r.negated ? '¬ ' : ''}${esc(r.predicate.slice(0,20))}</text></g>`;
  });
  for (const id of ids) {
    const p=positions.get(id),e=entities.get(id);
    svg+=`<g class="flow-node entity" tabindex="0" role="button" data-trace-entity="${id}" aria-label="查看实体 ${esc(e.name)}"><title>${esc(e.name)} · ${id}</title><rect x="${p.x-66}" y="${p.y-18}" width="132" height="36" rx="8"/><text x="${p.x}" y="${p.y-2}" text-anchor="middle">${esc(e.name.length>20 ? e.name.slice(0,18)+'…' : e.name)}</text><text x="${p.x}" y="${p.y+11}" text-anchor="middle">${id} · ${e.source_count || e.passage_ids.length} sources</text></g>`;
  }
  svg+='</svg>';$('#trace-graph').innerHTML=svg;
  $('.trace-graph-caption').textContent=`本轮实体关系的局部视图：显示 ${ids.length} 个实体、${selected.length} 条关系。悬浮关系查看出处；虚线是共现，箭头是谓词方向，¬ 表示否定；点击实体查看来源。`;
}

function showTraceEntity(id) {
  const {entities,relations}=traceGraphData(),entity=entities.get(id);
  if (!entity) return;
  const sources=entity.passage_ids.filter(cid => state.detail.evidence[cid]);
  $('#trace-source').innerHTML=`<h3>${esc(entity.name)} <span class="muted">${id}</span></h3><p class="evidence-location">${esc(entity.type)} · ${entity.mention_count || '—'} 次提及 · ${entity.source_count || entity.passage_ids.length} 个原文片段<br>别名：${esc(entity.aliases?.join(' / '))}</p><div class="evidence-chips">${sources.map(cid => `<button class="evidence-chip" data-trace-evidence="${cid}">${cid} ↗ 原文</button>`).join('')}</div><div class="relation-list">${[...relations.values()].filter(r => r.subject===id || r.object===id).slice(0,16).map(r => `<div class="relation-row">${esc(entities.get(r.subject)?.name || r.subject)} <b>→ ${r.negated ? 'NOT ' : ''}${esc(r.predicate)} →</b> ${esc(entities.get(r.object)?.name || r.object)}<small>${r.id} · ${esc(r.proposition_ids.slice(0,5).join(', '))}</small></div>`).join('')}</div><p class="evidence-location">该局部图仅用于定位来源；关系由解析器自动抽取，尚未验证。</p>`;
}
function exportCSV() {
  const fields = ['case_number','id','sub_domain','length','difficulty','answer','x_method','y_method','metric','x_value','y_value','x_prediction','y_prediction','x_score','y_score','max_score'];
  const quote = value => `"${String(value ?? '').replace(/"/g,'""')}"`;
  const lines = [fields, ...state.visible.map(i => [i.number,i.id,i.sub_domain,i.length,i.difficulty,i.answer,state.x,state.y,state.metric,i.methods[state.x][state.metric],i.methods[state.y][state.metric],i.methods[state.x].pred,i.methods[state.y].pred,i.methods[state.x].score,i.methods[state.y].score,i.max_score ?? 1])];
  download(`repar-${state.x}-vs-${state.y}-${state.visible.length}-cases.csv`, '\uFEFF'+lines.map(row => row.map(quote).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
}

init();

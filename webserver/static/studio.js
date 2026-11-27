import {mountKnowledge} from './knowledge.js';
import {mountKnowledgeGraph} from './knowledge_graph.js';

function icon(name, size = 15) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  fetch(`/static/vendor/${name}.svg`).then(response => response.ok ? response.text() : '').then(markup => {
    const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml').documentElement;
    for (const element of parsed.children) svg.append(document.importNode(element, true));
  });
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size)); svg.setAttribute('height', String(size));
  svg.setAttribute('fill', 'none'); svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8'); svg.setAttribute('stroke-linecap', 'round'); svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true'); svg.classList.add('ui-icon');
  return svg;
}

const $ = id => document.getElementById(id);
const apiRoot = '/api/studio/';
const managing = location.pathname === '/studio/libraries';
const indexCache = new Map();
const collections = {math: 'MATH 2025', longbench: 'LongBench v2 · 多文档'};
const labels = {queued: '\u6392\u961f\u4e2d', running: '\u8fd0\u884c\u4e2d', ready: '\u5df2\u5b8c\u6210', error: '\u5931\u8d25'};
const stages = {
  source_scope: '指定原文块',
  embedding: '分块与向量树', graph: '抽取实体关系', saving: '保存索引',
  direct_answer: 'Direct answer', plan: 'Plan', action: 'Action', observation: 'Observation',
  reflection: 'Reflection', finalizing: '\u6700\u7ec8\u4f5c\u7b54', control_error: '\u63a7\u5236\u9519\u8bef',
  blocked_action: '\u52a8\u4f5c\u62e6\u622a', blocked_duplicate: '\u91cd\u590d\u52a8\u4f5c\u62e6\u622a', transition: '\u72b6\u6001\u53d8\u66f4',
};
const reasons = {model_sufficient: '\u6a21\u578b\u51b3\u5b9a\u4f5c\u7b54', budget: '\u52a8\u4f5c\u9884\u7b97\u8017\u5c3d', no_progress: '\u68c0\u7d22\u65e0\u8fdb\u5c55',
  invalid_control: '\u63a7\u5236\u8f93\u51fa\u65e0\u6548', no_available_actions: '\u6ca1\u6709\u53ef\u7528\u7684\u64cd\u4f5c'};
const operators = {tree_search: '\u8bed\u4e49\u641c\u7d22', tree_browse: '\u6d4f\u89c8\u6811', graph_entities: '\u67e5\u627e\u5b9e\u4f53',
  graph_neighbors: '\u63a2\u7d22\u90bb\u5c45', read: '\u8bfb\u53d6\u539f\u6587', answer: '\u7ed3\u675f\u68c0\u7d22'};
let libraries = [], sessions = [], runs = [], files = [];
let libraryId, sessionId, runId, latestTrace, selection = 0, traceSelection = 0, graphView, graphLoad;
let loadingHistory = false, submission, pollingLibraries = false;
let managerId, managerView = 'details';
let deleteTarget;
const libraryTabs = [...$('library-tabs').querySelectorAll('[role="tab"]')];
const runActivity = new Map();
const expandedReplies = new Set();
const openReplySources = new Map();
const pendingRun = run => run.status === 'queued' || run.status === 'running';
let activityClock, followTrace = true, rawSelection = 0, refreshStageCalls;
let traceEvents = [];
const modelStages = new Set(['direct_answer', 'plan', 'action', 'reflection', 'finalizing', 'control_error']);
const replySizes = new ResizeObserver(entries => {
  for (const {target} of entries) {
    const preview = target.parentElement;
    const limit = parseFloat(getComputedStyle(preview).getPropertyValue('--reply-preview-height'));
    preview.classList.toggle('overflows', target.offsetHeight > limit);
    preview.nextElementSibling.hidden = !preview.classList.contains('overflows');
  }
});

function node(tag, text = '', cls = '') {
  const element = document.createElement(tag);
  element.textContent = text;
  if (cls) element.className = cls;
  return element;
}

function button(label, onClick, cls = 'outline-control') {
  const element = node('button', label, cls);
  element.type = 'button';
  element.addEventListener('click', onClick);
  return element;
}

function renderMessage(body, text) {
  for (const [index, part] of text.split(/\*\*([^*\n]+)\*\*/g).entries()) {
    body.append(index % 2 ? node('strong', part) : document.createTextNode(part));
  }
  window.renderMathInElement?.(body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '\\[', right: '\\]', display: true},
      {left: '\\(', right: '\\)', display: false},
      {left: '$', right: '$', display: false},
    ],
    output: 'mathml', trust: false, throwOnError: false, maxExpand: 1000,
  });
  return body;
}

function appendAnswer(reply, run) {
  const body = renderMessage(node('div', '', 'reply-body'), run.response || describe(run));
  if (!run.response) {
    reply.append(body);
    return;
  }
  body.id = `answer-${run.id}`;
  const preview = node('div', '', 'reply-preview');
  const toggle = button('', () => {
    if (expandedReplies.has(run.id)) expandedReplies.delete(run.id);
    else expandedReplies.add(run.id);
    renderToggle();
    if (!expandedReplies.has(run.id)) toggle.scrollIntoView({block: 'nearest'});
  }, 'text-button reply-expand');
  toggle.hidden = true;
  toggle.setAttribute('aria-controls', body.id);
  function renderToggle() {
    const expanded = expandedReplies.has(run.id);
    preview.classList.toggle('collapsed', !expanded);
    toggle.textContent = expanded ? '收起回答' : '展开完整回答';
    toggle.setAttribute('aria-expanded', String(expanded));
  }
  renderToggle();
  preview.append(body);
  reply.append(preview, toggle);
  replySizes.observe(body);
}

async function request(path, data, signal, method = data === undefined ? 'GET' : 'POST') {
  const options = data === undefined ? {method} : {
    method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data),
  };
  const response = await fetch(apiRoot + path, {...options, signal});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function status(id, message = '', failed = false) {
  $(id).textContent = message;
  $(id).classList.toggle('error', failed);
}

function describe(job) {
  if (job.status === 'error') return job.error;
  return job.seconds === undefined ? labels[job.status] : `${labels[job.status]} · ${job.seconds.toFixed(1)} s`;
}

function library() { return libraries.find(item => item.id === libraryId); }
function activeSession() { return sessions.find(item => item.id === sessionId); }

function libraryName(item) {
  return item.collection === 'longbench' ? `LongBench v2 ${item.name}` : item.name;
}

function showLibraries() {
  const select = $('session-library');
  select.replaceChildren();
  for (const [key, label] of Object.entries({local: '已有知识库', ...collections})) {
    const group = node('optgroup');
    group.label = label;
    for (const item of libraries.filter(item => item.status === 'ready' && (item.collection || 'local') === key)) {
      group.append(new Option(libraryName(item), item.id));
    }
    if (group.children.length) select.append(group);
  }
  if (!select.options.length) select.append(new Option('请先创建知识库', ''));
  select.disabled = !select.value;
}

function showSessions() {
  const host = $('session-list');
  const items = sessions.map(item => {
    const row = node('div', '', 'session-row');
    const entry = button('', () => selectSession(item.id), 'session-option');
    entry.setAttribute('aria-pressed', String(item.id === sessionId));
    entry.append(node('strong', item.title), node('small', `${item.run_count || 0} 轮问答`));
    const remove = button('', () => {
      deleteTarget = item.id;
      $('delete-session-name').textContent = item.title;
      status('delete-session-status');
      $('delete-session-dialog').showModal();
    }, 'icon-control delete-session');
    remove.title = '删除会话';
    remove.setAttribute('aria-label', `删除会话 ${item.title}`);
    remove.append(icon('trash-2'));
    row.append(entry, remove);
    return row;
  });
  host.replaceChildren(...(items.length ? items : [node('p', '还没有会话', 'session-list-empty')]));
}

function updateComposer() {
  const ready = library()?.status === 'ready';
  const busy = submission || runs.some(pendingRun);
  $('question').disabled = !ready;
  $('ask').disabled = !ready || busy || loadingHistory;
  $('question').placeholder = ready ? '向选中的知识库提问…' : '选择可用的知识库';
  $('session-library').disabled = Boolean(submission) || !$('session-library').value;
  $('follow-up').parentElement.hidden = !runs.some(run => run.status === 'ready');
  $('follow-up').disabled = !ready || Boolean(submission);
  $('messages').setAttribute('aria-busy', String(loadingHistory));
}

function showChat() {
  const selected = library();
  $('chat-title').textContent = activeSession()?.title || '新会话';
  const container = $('messages');
  replySizes.disconnect();
  const sending = submission?.ticket === selection ? submission : null;
  if (!runs.length && !sending) {
    const hint = node('div', '', 'chat-empty');
    if (!sessionId) {
      hint.classList.add('chat-welcome');
      hint.append(node('h2', '今天想查什么？'),
        node('p', '从输入框选一个知识库，直接提问。点击回答下方的引用可查看原文。'));
      if (!selected) {
        const create = node('a', '创建知识库', 'outline-control');
        create.href = '/studio/libraries?create=1';
        hint.append(create);
      } else {
        const examples = node('div', '', 'starter-questions');
        for (const [label, question] of [
          ['总结资料要点', '请概括这些资料的主题和要点，并给出相关出处。'],
          ['解释资料术语', '请列出资料中的主要术语，解释它们各指什么并引用原文。'],
          ['整理操作步骤', '资料中有哪些流程或操作步骤？请按原文的顺序梳理。'],
        ]) {
          examples.append(button(label, () => {
            $('question').value = question;
            $('question').focus();
          }, 'starter-question'));
        }
        hint.append(examples);
      }
    } else {
      hint.append(node('strong', loadingHistory ? '正在读取会话…' : '从一条问题开始'));
    }
    container.replaceChildren(hint);
    updateActivityClock();
    return;
  }
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 70;
  const exchanges = [...runs].reverse().concat(sending || []).map(run => {
    const exchange = node('article', '', 'exchange');
    exchange.append(renderMessage(node('div', '', 'question-message'), run.task));
    const used = libraries.find(item => item.id === run.library_id);
    const name = used ? libraryName(used) : '历史知识库';
    exchange.append(node('small', run.block_ids?.length ? `${name} · ${run.block_ids.join('、')}` : name, 'message-scope'));
    const reply = node('div', '', `reply${run.id === runId ? ' active' : ''}`);
    reply.append(node('strong', 'RePAR', 'reply-label'));
    if (pendingRun(run) || run === sending) {
      const activity = node('div', '', 'run-activity');
      activity.dataset.runActivity = run.id || '';
      const dots = node('span', '', 'run-dots');
      dots.setAttribute('aria-hidden', 'true');
      dots.append(node('i'), node('i'), node('i'));
      const elapsed = node('time', '', 'run-elapsed');
      elapsed.dateTime = run.created || '';
      elapsed.setAttribute('aria-live', 'off');
      const heading = node('div', '', 'run-activity-heading');
      heading.append(dots, node('strong', '', 'run-stage'), elapsed);
      activity.append(heading, node('p', '', 'run-stage-detail'));
      updateRunActivity(activity, run);
      reply.append(activity);
    } else {
      appendAnswer(reply, run);
    }
    const controls = node('div', '', 'reply-actions');
    if (run.id) {
      const trace = button('查看轨迹', () => {
        if (runId !== run.id) selectRun(run.id);
        setTraceExpanded(true);
        $('trace-heading').focus({preventScroll: true});
      });
      trace.setAttribute('aria-controls', 'trace-panel');
      controls.append(trace);
    }
    const sourceMappings = [];
    const sourceIds = (run.evidence_ids || []).map(id => {
      const sourceId = originalBlock(id, run);
      if (sourceId !== id) sourceMappings.push(`${id} → ${sourceId}`);
      return sourceId;
    });
    if (sourceMappings.length) reply.append(node('p', `本次引用 → 库内编号 · ${sourceMappings.join('、')}`, 'citation-map'));
    reply.append(controls);
    if (sourceIds.length) mountReplySources(reply, controls, run, sourceIds);
    exchange.append(reply);
    return exchange;
  });
  container.replaceChildren(...exchanges);
  updateActivityClock();
  if (nearBottom) requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
}

function activityText(run, event) {
  if (run.status === 'submitting') return ['正在提交问题', ''];
  if (run.status === 'queued') return ['已排队', '等待执行'];
  if (!event) return ['正在启动问答', '准备索引与初步回答'];
  const round = event.round ? `第 ${event.round} 轮 · ` : '';
  switch (event.stage) {
    case 'direct_answer': return ['已生成初步回答', ''];
    case 'plan': return [round + '已制定检索计划', event.plan.objective];
    case 'action': {
      const tool = event.decision.tool;
      return tool === 'answer' ? [round + '已决定结束检索', ''] : [round + '正在查询知识库', operators[tool] || tool];
    }
    case 'observation': return [round + '已收到查询结果', observationSummary(event)];
    case 'reflection': return [round + '已评估检索结果', event.reflection.assessment];
    case 'finalizing': return ['正在生成最终回答', reasons[event.termination] || ''];
    default: return [round + (stages[event.stage] || event.stage), eventSummary(event)];
  }
}

function updateRunActivity(host, run) {
  const state = runActivity.get(run.id);
  const [title, detail] = activityText(run, state?.event);
  const description = [detail, state?.offline ? '进度连接中断，正在重连' : ''].filter(Boolean).join(' · ');
  for (const [selector, value] of [['.run-stage', title], ['.run-stage-detail', description]]) {
    const field = host.querySelector(selector);
    if (field.textContent !== value) field.textContent = value;
  }
}

function updateActivityClock() {
  clearInterval(activityClock);
  const timers = [...$('messages').querySelectorAll('.run-elapsed')];
  if (!timers.length) return;
  function tick() {
    for (const timer of timers) {
      const elapsed = Math.max(0, Math.floor((Date.now() - Date.parse(timer.dateTime)) / 1000));
      if (!Number.isFinite(elapsed)) continue;
      const units = elapsed < 60 ? `${elapsed} 秒` : `${Math.floor(elapsed / 60)} 分 ${elapsed % 60} 秒`;
      const label = `已用 ${units}`;
      if (timer.textContent !== label) timer.textContent = label;
    }
  }
  tick();
  activityClock = setInterval(tick, 1000);
}

function observationSummary(event) {
  return (event.executed || []).map(({name, observation}) => {
    const tool = operators[name] || name;
    if (observation.status === 'no_match') return `${tool} · 无匹配`;
    if (observation.status !== 'ok') return `${tool} · ${observation.status}`;
    const count = observation.results?.length;
    return count ? `${tool} · ${count} 段原文` : `${tool} · 查询完成`;
  }).join(' / ');
}

function rawSection(label, value, open = false) {
  const details = node('details');
  details.open = open;
  const payload = node('pre');
  function render() {
    payload.textContent = value === undefined ? '此记录未保存该字段' : JSON.stringify(value, null, 2);
  }
  if (open) render();
  else details.addEventListener('toggle', () => {
    if (details.open && !payload.hasChildNodes()) render();
  });
  details.append(node('summary', label), payload);
  return details;
}

function eventSummary(event) {
  if (event.stage === 'source_scope') return event.block_ids.join('、');
  if (event.stage === 'direct_answer') return event.answer || '\u6a21\u578b\u672a\u8fd4\u56de\u8349\u7a3f';
  if (event.stage === 'plan') return event.plan.objective;
  if (event.stage === 'action') return event.decision.tool;
  if (event.stage === 'observation') return observationSummary(event);
  if (event.stage === 'reflection') return event.reflection.assessment;
  if (event.stage === 'finalizing') return reasons[event.termination] || event.termination;
  if (event.stage === 'embedding') return `${event.characters.toLocaleString()} 字符`;
  if (event.stage === 'graph') return `${event.leaves} \u539f\u6587\u5757 \u00b7 ${event.tree_nodes} \u6811\u8282\u70b9`;
  if (event.stage === 'saving') return `${event.entities} 实体`;
  return event.error || event.reason || '';
}

function eventDetail(event) {
  const panel = $('event-detail');
  refreshStageCalls = undefined;
  panel.scrollTop = 0;
  panel.replaceChildren(node('strong', stages[event.stage] || event.stage));
  if (modelStages.has(event.stage)) {
    if (event.stage === 'control_error') panel.append(node('p', event.error, 'panel-status error'));
    panel.append(modelCallDetails(event));
    return;
  }
  if (event.stage === 'observation') {
    const run = runs.find(item => item.id === runId);
    const executions = event.executed || [];
    if (!executions.length) panel.append(node('p', '此步骤没有工具调用记录', 'panel-status'));
    for (const execution of executions) {
      const visible = execution.visible_observation;
      const tool = node('details', '', 'observation-tool');
      const heading = node('summary', '', 'observation-tool-title');
      const elapsed = Number.isFinite(execution.tool_seconds) ? ` · ${execution.tool_seconds.toFixed(3)} s` : '';
      heading.append(node('code', execution.name), node('span', `${execution.observation.status}${elapsed}`));
      const body = node('div', '', 'observation-tool-content');
      body.append(rawSection('工具原始输入', execution.parameters),
        rawSection('工具完整输出', execution.observation));
      if (visible) body.append(rawSection('提交给 Reflection 的观察', visible));
      const sources = visible?.source_views || [];
      if (sources.length && run) {
        const references = node('details', '', 'observation-sources');
        references.append(node('summary', `原文出处 · ${sources.length} 段`));
        const fragments = node('div', '', 'observation-fragments');
        for (const source of sources) {
          const id = originalBlock(source.id, run);
          const fragment = node('details', '', 'observation-fragment');
          const label = id === source.id ? id : `${source.id} → ${id}`;
          fragment.append(node('summary', [label, sourceRange(source)].filter(Boolean).join(' · ')));
          fragment.append(node('pre', source.text, 'source-quotation'));
          fragments.append(fragment);
        }
        references.append(fragments);
        body.append(references);
      }
      tool.append(heading, body);
      panel.append(tool);
    }
    return;
  }
  panel.append(rawSection('步骤记录', event));
}

function appendMessages(host, messages, emptyLabel) {
  if (!messages?.length) {
    host.append(node('p', emptyLabel, 'panel-status'));
    return;
  }
  for (const message of messages) {
    const block = node('section', '', 'call-message');
    block.append(node('h3', message.role || 'assistant'));
    const texts = typeof message.content === 'string' ? [message.content] :
      Array.isArray(message.content) ? message.content.map(part => part?.text).filter(text => typeof text === 'string') : [];
    if (texts.some(text => text.length)) {
      for (const text of texts) block.append(node('pre', text));
    } else block.append(node('p', '此消息没有文本正文', 'panel-status'));
    host.append(block);
  }
}

function tokenCounts(usage) {
  const input = usage?.prompt_tokens;
  const output = usage?.completion_tokens;
  if (![input, output].every(value => Number.isSafeInteger(value) && value >= 0)) return 'Token 用量未记录';
  return `输入 ${input.toLocaleString()} · 输出 ${output.toLocaleString()} tokens`;
}

function showTraceUsage(usage) {
  const summary = $('trace-usage');
  const detail = usage?.calls ? [
    tokenCounts(usage.measured ? usage : null),
    usage.measured && usage.measured < usage.calls
      ? `已统计 ${usage.measured}/${usage.calls} 次调用` : `${usage.calls} 次调用`,
  ].join(' · ') : '';
  summary.hidden = !detail;
  if (summary.textContent !== detail) summary.textContent = detail;
}

function showModelCalls(body, calls) {
  const controls = node('div', '', 'call-controls');
  const content = node('div', '', 'call-content');
  const usage = node('p', '', 'token-usage call-usage');
  let attempt = 0, view = 'input';
  const views = [['input', '输入消息'], ['output', '输出正文'], ['json', 'JSON']];
  const modes = node('div', '', 'call-modes');
  modes.setAttribute('role', 'group');
  modes.setAttribute('aria-label', '模型调用内容');
  const buttons = views.map(([key, label]) => button(label, () => {
    view = key;
    render();
  }, 'call-mode'));
  modes.append(...buttons);
  if (calls.length > 1) {
    const select = node('select');
    select.setAttribute('aria-label', '调用次数');
    calls.forEach((_, i) => select.add(new Option(`第 ${i + 1} 次调用`, String(i))));
    select.addEventListener('change', () => {
      attempt = Number(select.value);
      render();
    });
    controls.append(select);
  }
  controls.append(modes);
  body.replaceChildren(controls, usage, content);
  function render() {
    buttons.forEach((control, i) => control.setAttribute('aria-pressed', String(views[i][0] === view)));
    const call = calls[attempt];
    usage.textContent = [
      tokenCounts(call.response?.usage),
      call.cached ? '缓存复用' : Number.isFinite(call.seconds) ? `${call.seconds.toFixed(2)} s` : '',
    ].filter(Boolean).join(' · ');
    content.replaceChildren();
    if (view === 'input') {
      appendMessages(content, call.request?.messages, '原始输入消息不可用');
    } else if (view === 'output') {
      const messages = call.response?.choices?.map(choice => choice.message).filter(Boolean);
      appendMessages(content, messages, '原始输出正文不可用');
    } else {
      if (call.request) content.append(rawSection('请求 JSON', call.request, true));
      if (call.response) content.append(rawSection('响应 JSON', call.response));
    }
  }
  render();
}

function modelCallDetails(event) {
  const id = runId;
  const section = node('details', '', 'model-call-details');
  const body = node('div', '', 'model-call-records');
  section.append(node('summary', '模型原始输入与输出'), body);
  let loaded = false, loading = false;
  async function load() {
    if (!section.open || loading || loaded) return;
    loading = true;
    body.textContent = '读取调用记录…';
    try {
      const {calls} = await request(`runs/${id}/calls/${event.stage}/${event.round || 0}`);
      loaded = calls.length > 0;
      if (loaded) showModelCalls(body, calls);
      else body.textContent = '此步骤暂无已保存的模型调用';
    } catch (error) {
      body.textContent = error.message;
    } finally {
      loading = false;
    }
  }
  section.addEventListener('toggle', load);
  refreshStageCalls = load;
  return section;
}

function originalBlock(id, run) {
  return /^C\d+$/.test(id) ? run?.block_ids?.[Number(id.slice(1))] || id : id;
}

function loadIndex(id) {
  if (!indexCache.has(id)) {
    indexCache.set(id, request(`libraries/${id}/index`).catch(error => {
      indexCache.delete(id);
      throw error;
    }));
    if (indexCache.size > 2) indexCache.delete(indexCache.keys().next().value);
  }
  return indexCache.get(id);
}

function sourceRange({start, end}) {
  return Number.isInteger(start) && Number.isInteger(end) ? `字符 ${start.toLocaleString()}–${end.toLocaleString()}` : '';
}

function sourceReading(passage, library) {
  const heading = node('div', '', 'source-heading');
  const caption = [library?.name && libraryName(library), sourceRange(passage)].filter(Boolean).join(' · ');
  heading.append(node('strong', passage.id, 'evidence-id'), node('span', caption, 'source-offset'));
  const reader = node('article', '', 'source-detail');
  reader.append(heading, node('pre', passage.text, 'source-quotation'));
  if (library) {
    const link = node('a', '在知识库中查看', 'outline-control');
    link.href = `/studio/libraries?library=${encodeURIComponent(library.id)}&block=${encodeURIComponent(passage.id)}`;
    reader.append(link);
  }
  return reader;
}

function mountReplySources(reply, controls, run, ids) {
  const host = node('div', '', 'reply-source');
  host.id = `reply-source-${run.id}`;
  host.hidden = true;
  host.setAttribute('aria-live', 'polite');
  const triggers = new Map();
  let ticket = 0;
  for (const id of new Set(ids)) {
    const control = button(id, () => openSource(openReplySources.get(run.id) === id ? undefined : id),
      'outline-control source-pill');
    control.setAttribute('aria-controls', host.id);
    control.setAttribute('aria-expanded', 'false');
    triggers.set(id, control);
    controls.append(control);
  }
  reply.append(host);
  const selected = openReplySources.get(run.id);
  if (triggers.has(selected)) openSource(selected);

  async function openSource(id) {
    const current = ++ticket;
    if (id) openReplySources.set(run.id, id);
    else openReplySources.delete(run.id);
    for (const [candidate, control] of triggers) control.setAttribute('aria-expanded', String(candidate === id));
    host.hidden = !id;
    if (!id) {
      host.replaceChildren();
      return;
    }
    host.replaceChildren(node('p', '读取原文…', 'panel-status'));
    try {
      const index = await loadIndex(run.library_id);
      if (current !== ticket || !host.isConnected) return;
      const passage = index.chunks.find(chunk => chunk.id === id);
      if (!passage) throw new Error('该知识库中没有这段原文。');
      host.replaceChildren(sourceReading(passage, libraries.find(item => item.id === run.library_id)));
    } catch (error) {
      if (current === ticket && host.isConnected) host.replaceChildren(node('p', error.message, 'panel-status error'));
    }
  }
}

function saveLibraryLocation(block) {
  const params = new URLSearchParams();
  if (managerId) params.set('library', managerId);
  if (managerView !== 'details') params.set('view', managerView);
  if (block) params.set('block', block);
  history.replaceState(null, '', params.size ? `/studio/libraries?${params}` : '/studio/libraries');
}

function clearGraph() {
  graphView?.stop();
  graphView = graphLoad = undefined;
  $('graph-view').replaceChildren();
}

async function loadGraph(selected, block) {
  const pending = {};
  graphLoad = pending;
  const host = $('graph-view');
  const loading = node('p', '加载知识图谱…', 'graph-loading');
  loading.setAttribute('role', 'status');
  host.replaceChildren(loading);
  try {
    const data = await loadIndex(selected.id);
    if (graphLoad !== pending) return;
    graphView = mountKnowledgeGraph(host, selected, data, sourceReading);
    if (block) graphView.selectBlock(block);
  } catch (error) {
    if (graphLoad === pending) host.replaceChildren(node('p', error.message, 'graph-error'));
  }
}

function showLibraryView(view, block) {
  const selected = libraries.find(item => item.id === managerId);
  managerView = selected?.status === 'ready' ? view : 'details';
  clearGraph();
  $('knowledge-content').replaceWith(Object.assign(node('div'), {id: 'knowledge-content'}));
  for (const tab of libraryTabs) {
    const active = tab.dataset.libraryView === managerView;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    $(tab.getAttribute('aria-controls')).hidden = !active;
  }
  $('library-content').scrollTop = 0;
  saveLibraryLocation(managerView === 'graph' ? block : undefined);
  if (managerView === 'graph') loadGraph(selected, block);
  if (managerView === 'index') openIndex(selected);
}

function addEvents(events) {
  traceEvents.push(...events);
  const timeline = $('timeline');
  for (const event of events) {
    const item = node('li');
    const choose = button('', () => {
      followTrace = choose === timeline.lastElementChild.firstElementChild;
      selectEvent(choose, event);
    }, 'timeline-event');
    const label = [stages[event.stage] || event.stage, event.round && `\u7b2c ${event.round} \u8f6e`].filter(Boolean).join(' \u00b7 ');
    choose.setAttribute('aria-pressed', 'false');
    choose.dataset.stage = event.stage;
    choose.append(node('strong', label), node('span', eventSummary(event)));
    item.append(choose);
    timeline.append(item);
  }
  if (events.length && followTrace) {
    timeline.lastElementChild.firstElementChild.click();
    timeline.scrollTop = timeline.scrollHeight;
  }
  $('trace-empty').hidden = Boolean(timeline.childElementCount);
  $('trace-count').textContent = timeline.childElementCount ? `${timeline.childElementCount} 事件` : '';
}

function selectEvent(control, event) {
  $('timeline').querySelector('[aria-pressed="true"]')?.setAttribute('aria-pressed', 'false');
  control.setAttribute('aria-pressed', 'true');
  $('follow-trace').hidden = followTrace;
  showRawTrace(false);
  eventDetail(event);
}

function setTraceExpanded(expanded) {
  const toggle = $('trace-toggle');
  toggle.setAttribute('aria-expanded', String(expanded));
  const label = expanded ? '收起执行轨迹' : '展开执行轨迹';
  toggle.setAttribute('aria-label', label);
  toggle.title = label;
  $('trace-content').hidden = !expanded;
  if (expanded && followTrace) {
    const timeline = $('timeline');
    timeline.scrollTop = timeline.scrollHeight;
  }
}

function showRawTrace(open) {
  rawSelection++;
  $('raw-content').hidden = !open;
  $('event-detail').hidden = open;
  $('download').hidden = !open;
  $('raw-trace').textContent = open ? '返回步骤' : '完整 Trace';
}

function resetTrace() {
  traceEvents = [];
  followTrace = true;
  latestTrace = undefined;
  refreshStageCalls = undefined;
  showRawTrace(false);
  $('timeline').replaceChildren();
  $('event-detail').replaceChildren();
  $('raw-content').replaceChildren();
  $('trace-count').textContent = '';
  showTraceUsage();
  $('follow-trace').hidden = true;
  $('raw-trace').disabled = true;
  const run = runId ? runs.find(item => item.id === runId) : submission?.ticket === selection ? submission : null;
  $('trace-panel').hidden = !run;
  $('trace-question').textContent = run?.task || '';
  $('trace-question').title = run?.task || '';
  $('trace-link').hidden = !runId;
  if (runId) $('trace-link').href = `/studio?session=${sessionId}&run=${runId}`;
  $('trace-empty').hidden = false;
  $('trace-empty').textContent = run?.status === 'ready' ? '正在读取轨迹…' : '等待执行事件';
  status('trace-status', runId ? describe(run) : run ? '正在提交问题' : '');
}

const pause = delay => new Promise(resolve => setTimeout(resolve, delay));

async function watchRuns(id, ticket) {
  const offsets = new Map();
  let traceComplete = false;
  while (ticket === traceSelection) {
    let finished = false;
    const watching = runs.filter(run => run.id === id ? !traceComplete : pendingRun(run));
    await Promise.all(watching.map(async run => {
      const state = runActivity.get(run.id) || {};
      try {
        const result = await request(`runs/${run.id}?offset=${offsets.get(run.id) ?? 0}`, undefined, AbortSignal.timeout(8000));
        if (ticket !== traceSelection) return;
        const {job, events, offset, usage} = result;
        offsets.set(run.id, offset);
        if (pendingRun(job)) runActivity.set(run.id, {event: events.at(-1) || state.event});
        else runActivity.delete(run.id);
        const index = runs.findIndex(item => item.id === run.id);
        if (index >= 0) runs[index] = job;
        finished ||= pendingRun(run) && !pendingRun(job);
        if (run.id === id) {
          addEvents(events);
          showTraceUsage(usage);
          const termination = reasons[job.termination] || job.termination;
          const progress = pendingRun(job) ? activityText(job, runActivity.get(run.id)?.event)[0] : describe(job);
          status('trace-status', [progress, termination].filter(Boolean).join(' · '), job.status === 'error');
          traceComplete = !pendingRun(job);
          if (traceComplete) $('trace-empty').textContent = '没有可显示的执行事件';
          $('raw-trace').disabled = !traceComplete;
          if (traceComplete) refreshStageCalls?.();
        }
      } catch (error) {
        if (ticket !== traceSelection) return;
        if (pendingRun(run)) runActivity.set(run.id, {...state, offline: true});
        if (run.id === id) status('trace-status', '轨迹连接中断，正在重连', true);
      }
      const activity = $('messages').querySelector(`[data-run-activity="${CSS.escape(run.id)}"]`);
      if (activity) updateRunActivity(activity, runs.find(item => item.id === run.id) || run);
    }));
    if (ticket !== traceSelection) return;
    if (finished) {
      showChat();
      updateComposer();
    }
    if (traceComplete && !runs.some(pendingRun)) return;
    await pause(1100);
  }
}

function saveLocation() {
  const params = new URLSearchParams();
  if (sessionId) params.set('session', sessionId);
  else {
    params.set('new', '1');
    if (libraryId) params.set('library', libraryId);
  }
  if (runId) params.set('run', runId);
  history.replaceState(null, '', params.size ? `/studio?${params}` : '/studio');
}

function chooseLibrary(...ids) {
  const preferred = [...ids, localStorage.getItem('repar-studio-default-library')]
    .map(id => libraries.find(item => item.id === id && item.status === 'ready')).find(Boolean);
  libraryId = preferred?.id || libraries.find(item => item.status === 'ready')?.id;
  $('session-library').value = libraryId || '';
}

async function selectSession(id, restoreRun) {
  const selected = sessions.find(item => item.id === id);
  if (!selected) return;
  if (managing) {
    location.href = `/studio?session=${encodeURIComponent(id)}`;
    return;
  }
  const ticket = ++selection;
  traceSelection++;
  runActivity.clear();
  expandedReplies.clear();
  openReplySources.clear();
  sessionId = selected.id;
  localStorage.setItem('repar-studio-default-session', sessionId);
  runId = undefined;
  runs = [];
  loadingHistory = true;
  chooseLibrary(selected.last_library_id, selected.library_id);
  $('question').value = '';
  $('follow-up').checked = false;
  status('chat-status');
  resetTrace();
  showSessions();
  showChat();
  updateComposer();
  saveLocation();
  try {
    const result = await request(`sessions/${id}/runs`);
    if (ticket !== selection) return;
    loadingHistory = false;
    runs = result.runs;
    selected.run_count = runs.length;
    if (selected.auto_title && runs.length) selected.title = runs.at(-1).task.replace(/\s+/g, ' ').slice(0, 48);
    showSessions();
    const run = runs.find(item => item.id === restoreRun) || runs[0];
    if (run) selectRun(run.id);
    else {
      showChat();
      updateComposer();
    }
  } catch (error) {
    if (ticket === selection) {
      loadingHistory = false;
      status('chat-status', error.message, true);
      updateComposer();
    }
  }
}

function selectRun(id) {
  runId = id;
  resetTrace();
  setTraceExpanded(true);
  showChat();
  updateComposer();
  saveLocation();
  watchRuns(id, ++traceSelection);
}

function openIndex(selected) {
  mountKnowledge($('index-host'), {structures: {v5: selected.id}, methods: {v5: {evidence_ids: []}}},
    {v5: 'RePAR v5'}, () => loadIndex(selected.id), 'v5', {eager: true, singleIndex: true});
}

function openNewSession(preferredId = libraryId, focus = true) {
  if (managing) {
    location.href = '/studio?new=1';
    return;
  }
  selection++;
  traceSelection++;
  runActivity.clear();
  expandedReplies.clear();
  openReplySources.clear();
  sessionId = runId = undefined;
  runs = [];
  loadingHistory = false;
  chooseLibrary(preferredId);
  $('question').value = '';
  $('follow-up').checked = false;
  status('chat-status');
  resetTrace();
  showSessions();
  showChat();
  updateComposer();
  saveLocation();
  if (focus) $('question').focus();
}

function listFiles(selected) {
  files = Array.from(selected);
  $('files-list').replaceChildren(...files.map(file => node('li', file.name)));
  if (!$('library-name').value && files.length) $('library-name').value = files[0].name.replace(/\.[^.]+$/, '');
  $('files-list').hidden = !files.length;
}

for (const [id, name] of [['new-session', 'plus'], ['trace-toggle', 'chevron-down']]) $(id).prepend(icon(name));

$('new-session').addEventListener('click', () => openNewSession());
$('cancel-session-delete').addEventListener('click', () => $('delete-session-dialog').close());
$('delete-session-dialog').addEventListener('cancel', event => {
  if ($('confirm-session-delete').disabled) event.preventDefault();
});
$('delete-session-form').addEventListener('submit', async event => {
  event.preventDefault();
  const confirm = $('confirm-session-delete');
  const cancel = $('cancel-session-delete');
  if (!deleteTarget || confirm.disabled) return;
  const id = deleteTarget;
  confirm.disabled = cancel.disabled = true;
  status('delete-session-status', '正在删除…');
  try {
    await request(`sessions/${encodeURIComponent(id)}`, undefined, undefined, 'DELETE');
    sessions = sessions.filter(item => item.id !== id);
    if (localStorage.getItem('repar-studio-default-session') === id) localStorage.removeItem('repar-studio-default-session');
    $('delete-session-dialog').close();
    if (sessionId === id) openNewSession();
    else {
      showSessions();
      const current = $('session-list').querySelector('[aria-pressed="true"]');
      (current || $('new-session')).focus();
    }
  } catch (error) {
    status('delete-session-status', error.message, true);
  } finally {
    confirm.disabled = cancel.disabled = false;
  }
});
$('session-library').addEventListener('change', event => {
  libraryId = event.target.value;
  if (libraryId) localStorage.setItem('repar-studio-default-library', libraryId);
  saveLocation();
  updateComposer();
});
$('files').addEventListener('change', event => listFiles(event.target.files));
$('dropbox').addEventListener('dragover', event => {
  event.preventDefault();
  $('dropbox').classList.add('dragging');
});
$('dropbox').addEventListener('dragleave', () => $('dropbox').classList.remove('dragging'));
$('dropbox').addEventListener('drop', event => {
  event.preventDefault();
  $('dropbox').classList.remove('dragging');
  listFiles(event.dataTransfer.files);
});

function renderDirectory() {
  const query = $('library-search').value.trim().toLowerCase();
  const filtered = libraries.filter(item => !query || [libraryName(item), ...item.sources.map(source => source.name)]
    .some(text => text.toLowerCase().includes(query)));
  const directory = $('library-directory');
  directory.replaceChildren();
  for (const [key, label] of Object.entries({local: '上传资料', ...collections})) {
    const entries = filtered.filter(item => (item.collection || 'local') === key);
    if (!entries.length) continue;
    const group = node('section', '', 'directory-group');
    group.append(node('h3', `${label} · ${entries.length}`));
    for (const item of entries) {
      const choice = button('', () => {
        if (managerId === item.id) return;
        managerId = item.id;
        renderDirectory();
        renderLibraryDetails();
        showLibraryView(managerView);
      }, 'directory-option');
      choice.setAttribute('aria-pressed', String(item.id === managerId));
      choice.append(node('strong', libraryName(item)), node('small', item.status === 'ready' ? '可查询' : labels[item.status]));
      group.append(choice);
    }
    directory.append(group);
  }
  if (!filtered.length) directory.append(node('p', '没有匹配的知识库。', 'panel-status'));
}

function renderLibraryDetails() {
  const host = $('library-summary');
  const selected = libraries.find(item => item.id === managerId);
  $('library-heading').hidden = !selected;
  const ready = selected?.status === 'ready';
  for (const tab of libraryTabs.slice(1)) tab.disabled = !ready;
  const ask = $('ask-with-library');
  ask.hidden = !ready;
  host.replaceChildren();
  if (!selected) {
    host.append(node('p', '创建知识库后可在这里查看索引和来源文件。', 'panel-status'));
    return;
  }
  $('current-library-name').textContent = libraryName(selected);
  status('current-library-status', ready ? '可查询' : describe(selected), selected.status === 'error');
  if (ready) ask.href = `/studio?new=1&library=${encodeURIComponent(selected.id)}`;
  if (selected.stats) {
    const stats = node('dl', '', 'library-metrics');
    for (const [label, value] of [
      ['原文块', selected.stats.leaves], ['实体', selected.stats.entities], ['关系', selected.stats.relations],
    ]) {
      const metric = node('div');
      metric.append(node('dt', label), node('dd', value.toLocaleString()));
      stats.append(metric);
    }
    host.append(stats);
  }
  const sources = node('ul', '', 'manager-sources');
  sources.append(...selected.sources.map(source => {
    const row = node('li');
    row.append(node('strong', source.name), node('small', `${(source.end - source.start).toLocaleString()} 字符`));
    return row;
  }));
  host.append(node('h3', '来源文件'), sources);
}

function showBuildForm(visible) {
  showLibraryView('details');
  $('upload-form').hidden = !visible;
  document.querySelector('.manager-grid').hidden = visible;
  if (visible) $('library-name').focus();
}

async function pollLibraries() {
  if (pollingLibraries) return;
  pollingLibraries = true;
  status('manager-status');
  while (libraries.some(item => item.status === 'queued' || item.status === 'running')) {
    await pause(1100);
    try {
      const selected = libraries.find(item => item.id === managerId);
      libraries = (await request('libraries')).libraries;
      renderDirectory();
      if (selected?.id !== managerId || selected?.status !== 'ready') renderLibraryDetails();
    } catch (error) {
      status('manager-status', error.message, true);
      break;
    }
  }
  pollingLibraries = false;
}

$('library-search').addEventListener('input', renderDirectory);
for (const tab of libraryTabs) tab.addEventListener('click', () => {
  if (tab.dataset.libraryView !== managerView) showLibraryView(tab.dataset.libraryView);
});
$('library-tabs').addEventListener('keydown', event => {
  const enabled = libraryTabs.filter(tab => !tab.disabled);
  const current = enabled.indexOf(event.target);
  if (current < 0) return;
  const destinations = {
    ArrowLeft: (current - 1 + enabled.length) % enabled.length,
    ArrowRight: (current + 1) % enabled.length,
    Home: 0, End: enabled.length - 1,
  };
  if (!Object.hasOwn(destinations, event.key)) return;
  event.preventDefault();
  const next = enabled[destinations[event.key]];
  next.click();
  next.focus();
});
$('new-library').addEventListener('click', () => showBuildForm(true));
$('cancel-build').addEventListener('click', () => showBuildForm(false));

$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('build').disabled = true;
  status('upload-status');
  try {
    if (!files.length) throw new Error('请选择要构建知识库的文本文件');
    if (files.reduce((sum, file) => sum + file.size, 0) > 16 * 1024 * 1024) throw new Error('文档超过 16 MiB');
    const decoder = new TextDecoder('utf-8', {fatal: true});
    const documents = await Promise.all(files.map(async file => ({name: file.name, text: decoder.decode(await file.arrayBuffer())})));
    const job = await request('libraries', {name: $('library-name').value, documents});
    libraries.unshift(job);
    managerId = job.id;
    $('library-search').value = '';
    $('upload-form').reset();
    listFiles([]);
    showBuildForm(false);
    renderDirectory();
    renderLibraryDetails();
    pollLibraries();
  } catch (error) {
    status('upload-status', error.message, true);
  } finally {
    $('build').disabled = false;
  }
});

$('ask-form').addEventListener('submit', async event => {
  event.preventDefault();
  const id = libraryId;
  let currentSessionId = sessionId;
  const ticket = selection;
  const task = $('question').value.trim();
  const previous = $('follow-up').checked ? runs.find(run => run.status === 'ready')?.id : undefined;
  if (!task || $('ask').disabled) return;
  submission = {ticket, task, library_id: id, status: 'submitting', created: new Date().toISOString()};
  updateComposer();
  status('chat-status');
  ++traceSelection;
  runId = undefined;
  resetTrace();
  setTraceExpanded(true);
  showChat();
  $('messages').scrollTop = $('messages').scrollHeight;
  try {
    if (!currentSessionId) {
      const session = await request('sessions', {});
      currentSessionId = session.id;
      sessions.unshift(session);
      showSessions();
      if (ticket === selection) {
        sessionId = currentSessionId;
        localStorage.setItem('repar-studio-default-session', sessionId);
        localStorage.setItem('repar-studio-default-library', id);
        saveLocation();
      }
    }
    const job = await request('runs', {session_id: currentSessionId, library_id: id, task, previous_run: previous});
    const active = sessions.find(item => item.id === currentSessionId);
    active.run_count = (active.run_count || 0) + 1;
    active.last_library_id = id;
    if (active.auto_title && active.run_count === 1) active.title = task.replace(/\s+/g, ' ').slice(0, 48);
    showSessions();
    submission = undefined;
    if (currentSessionId !== sessionId) return;
    if (ticket !== selection) {
      await selectSession(currentSessionId, job.id);
      return;
    }
    if ($('question').value.trim() === task) $('question').value = '';
    runs.unshift(job);
    selectRun(job.id);
    $('messages').scrollTop = $('messages').scrollHeight;
  } catch (error) {
    submission = undefined;
    if (currentSessionId === sessionId) {
      if (runs.length) selectRun(runs[0].id);
      else {
        resetTrace();
        showChat();
      }
      status('chat-status', error.message, true);
    }
  } finally {
    updateComposer();
  }
});

$('trace-toggle').addEventListener('click', () => setTraceExpanded($('trace-content').hidden));
$('follow-trace').addEventListener('click', () => {
  const timeline = $('timeline');
  const latest = timeline.lastElementChild?.firstElementChild;
  latest?.click();
  if (latest) {
    timeline.scrollTop = timeline.scrollHeight;
    latest.focus({preventScroll: true});
  }
});

$('raw-trace').addEventListener('click', async () => {
  const open = $('raw-content').hidden;
  showRawTrace(open);
  if (!open) return;
  const id = runId;
  const ticket = rawSelection;
  $('download').disabled = true;
  latestTrace = undefined;
  $('raw-content').replaceChildren(node('p', '\u8bfb\u53d6\u5b8c\u6574\u8f68\u8ff9\u2026'));
  try {
    const result = await request(`runs/${id}/trace`);
    if (ticket !== rawSelection) return;
    latestTrace = {id, result};
    const calls = result.calls.map(call => rawSection(`${call.tag} · ${call.seconds.toFixed(2)} s`, call));
    $('raw-content').replaceChildren(node('p', `${result.calls.length} \u6b21\u6a21\u578b\u8c03\u7528 \u00b7 ${result.trace.length} \u8f6e\u63a7\u5236`),
      rawSection('作答结果', {response: result.response, evidence: result.evidence, error: result.error}, true),
      rawSection('\u7ec8\u6b62\u72b6\u6001', {termination: result.termination, state: result.state}),
      rawSection('事件 JSON · 原始调试记录', traceEvents), ...calls);
    if (result.source_map) $('raw-content').children[1].after(rawSection('引用编号映射', result.source_map));
    $('download').disabled = false;
  } catch (error) {
    if (ticket === rawSelection) $('raw-content').replaceChildren(node('p', error.message, 'panel-status error'));
  }
});

$('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(latestTrace.result, null, 2)], {type: 'application/json'}));
  const link = node('a');
  link.href = url;
  link.download = `repar-v5-${latestTrace.id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function init() {
  if (managing) {
    document.body.classList.add('studio-management');
    document.querySelector('.session-panel').hidden = true;
    document.querySelector('.workspace-main').hidden = true;
    $('library-manager').hidden = false;
    $('libraries-link').setAttribute('aria-current', 'page');
    document.title = 'RePAR · 知识库管理';
    status('manager-status', '加载知识库…');
  } else {
    $('chat-link').setAttribute('aria-current', 'page');
  }
  try {
    const query = new URLSearchParams(location.search);
    const [libraryList, sessionList] = await Promise.all([request('libraries'), managing ? null : request('sessions')]);
    libraries = libraryList.libraries;
    if (managing) {
      status('manager-status');
      const linked = libraries.find(item => item.id === query.get('library'));
      managerId = linked?.id || libraries[0]?.id;
      renderDirectory();
      renderLibraryDetails();
      if (query.has('create')) showBuildForm(true);
      else if (linked && (query.get('view') === 'graph' || query.has('graph') || query.has('block'))) {
        showLibraryView('graph', query.get('block'));
      } else showLibraryView(linked && query.get('view') === 'index' ? 'index' : 'details');
      pollLibraries();
      return;
    }
    sessions = sessionList.sessions;
    showLibraries();
    if (query.has('new')) {
      openNewSession(query.get('library'), false);
      return;
    }
    let selected = sessions.find(item => item.id === query.get('session'));
    if (!selected && query.get('library')) {
      const legacyId = `legacy-${query.get('library')}`;
      selected = sessions.find(item => item.id === legacyId) || sessions.find(item => item.library_id === query.get('library'));
    }
    if (!selected) {
      const preferredId = localStorage.getItem('repar-studio-default-session');
      selected = sessions.find(item => item.id === preferredId);
    }
    if (selected) await selectSession(selected.id, query.get('run'));
    else openNewSession(undefined, false);
  } catch (error) {
    if (managing) {
      status('manager-status', error.message, true);
      return;
    }
    status('chat-status', error.message, true);
    showSessions();
    showChat();
  }
}

init();


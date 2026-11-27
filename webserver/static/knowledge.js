const cache = new Map();
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pageSize = 12;
const button = (kind, id, label, active = false) =>
  `<button type="button" class="knowledge-item ${active ? 'active' : ''}" data-knowledge-${kind}="${esc(id)}"${active ? ' aria-current="true"' : ''}>${esc(label)}</button>`;

export function mountKnowledge(panel, item, methods, fetchJSON, preferredMethod, {eager = false, chunk: initialChunk, singleIndex = false} = {}) {
  const host = panel.querySelector('#knowledge-content');
  const available = Object.keys(methods);
  const defaultMethod = available.includes(preferredMethod) ? preferredMethod : available[0];
  host.innerHTML = `<div class="knowledge-controls">
    <label ${singleIndex ? 'hidden' : ''}>检索方法 <select id="knowledge-method">${available.map(key => `<option value="${key}">${esc(methods[key])}</option>`).join('')}</select></label>
    <label>查找 <input id="knowledge-search" type="search" placeholder="节点、关系、关键词或原文"></label>
    <button class="button" id="knowledge-final" ${singleIndex && !item.methods[defaultMethod].evidence_ids.length ? 'hidden' : ''}>定位到首段作答证据</button>
    </div><p id="knowledge-status" class="evidence-location" role="status"></p>
    <div id="knowledge-search-results"></div><div id="knowledge-workspace"></div>`;
  const select = host.querySelector('#knowledge-method');
  const search = host.querySelector('#knowledge-search');
  const status = host.querySelector('#knowledge-status');
  const finalButton = host.querySelector('#knowledge-final');
  const workspace = host.querySelector('#knowledge-workspace');
  const results = host.querySelector('#knowledge-search-results');
  let index, method, treeId, chunkId, entityId, edgeId, chunkPage, edgePage, resultPage;
  let request = 0;
  select.value = defaultMethod;

  function prepare(data) {
    const byId = records => new Map((records || []).map(record => [record.id, record]));
    const chunks = byId(data.chunks);
    const nodes = byId(data.nodes);
    const entities = byId(data.entities);
    const edges = byId(data.relations);
    const propositions = byId(data.propositions);
    const edgesByChunk = new Map(), edgesByEntity = new Map();
    function add(map, key, edge) {
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(edge);
    }
    for (const edge of edges.values()) {
      for (const id of edge.passage_ids) add(edgesByChunk, id, edge);
      for (const id of [edge.subject, edge.object]) add(edgesByEntity, id, edge);
    }
    return {data, chunks, nodes, entities, edges, propositions, edgesByChunk, edgesByEntity};
  }

  async function load() {
    const ticket = ++request;
    method = select.value;
    const key = item.structures[method];
    index = null;
    search.value = '';
    workspace.replaceChildren();
    results.replaceChildren();
    status.textContent = '正在读取索引…';
    finalButton.disabled = true;
    try {
      const data = await fetchJSON(`/data/knowledge/${encodeURIComponent(key)}.json`, cache);
      if (cache.size > 3) cache.delete(cache.keys().next().value);
      if (ticket !== request || !host.isConnected) return;
      index = prepare(data);
      treeId = data.nodes?.[0]?.id;
      chunkId = entityId = edgeId = null;
      chunkPage = edgePage = resultPage = 0;
      const extractors = data.extractor?.models ? Object.entries(data.extractor.models).map(entry => entry.join(' ')) :
        [data.extractor?.model, data.extractor?.version].filter(Boolean);
      const graphStatus = graphEnabled() ? (data.kind === 'entity' ?
        `${index.entities.size} 个实体 · ${index.edges.size} 条关系 · 抽取器 ${extractors.join(' / ')}` :
        `${index.edges.size} 条片段关联`) : data.kind === 'bm25' ? `BM25 · 目标块长 ${data.chunk_words} 词` : '图查询已禁用';
      status.textContent = `${index.chunks.size} 个原文块${index.nodes.size ? ` · ${index.nodes.size} 个树节点` : ''} · ${graphStatus}`;
      finalButton.disabled = !item.methods[method].evidence_ids.some(id => index.chunks.has(id));
      if (initialChunk) showChunk(initialChunk);
      render();
    } catch (error) {
      if (ticket === request) status.textContent = error.message;
    }
  }

  function graphEnabled() {
    return index.data.kind !== 'bm25' && !method.startsWith('repar_tree');
  }

  function pagination(kind, page, total) {
    const pages = Math.ceil(total / pageSize);
    if (pages <= 1) return '';
    return `<div class="knowledge-pages">
      <button class="button" data-knowledge-page="${kind}" data-direction="-1" ${page === 0 ? 'disabled' : ''}>上一页</button>
      <span>${page + 1} / ${pages}</span>
      <button class="button" data-knowledge-page="${kind}" data-direction="1" ${page + 1 >= pages ? 'disabled' : ''}>下一页</button>
    </div>`;
  }

  function label(id) {
    return index.entities.get(id)?.name ?? id;
  }

  function find() {
    const query = search.value.trim().toLowerCase();
    if (!query || !index) {
      results.replaceChildren();
      return;
    }
    const matches = [];
    const has = (...parts) => parts.join(' ').toLowerCase().includes(query);
    for (const node of index.nodes.values()) {
      if (has(node.id, ...node.keywords)) matches.push(['tree', node.id, `${node.id} · ${node.keywords.slice(0, 4).join(' / ')}`]);
    }
    for (const chunk of index.chunks.values()) {
      if (has(chunk.id, chunk.text)) matches.push(['chunk', chunk.id, `${chunk.id} · ${chunk.text.trim().slice(0, 55)}`]);
    }
    if (graphEnabled()) {
      for (const entity of index.entities.values()) {
        if (has(entity.id, entity.name, ...entity.aliases)) matches.push(['entity', entity.id, `${entity.id} · ${entity.name}`]);
      }
      for (const edge of index.edges.values()) {
        if (has(edge.id, edge.kind, edge.predicate, label(edge.subject), label(edge.object))) {
          matches.push(['edge', edge.id, `${edge.id} · ${label(edge.subject)} → ${edge.predicate} → ${label(edge.object)}`]);
        }
      }
    }
    results.innerHTML = `<h3>${matches.length} 个匹配项</h3><div class="knowledge-items">
      ${matches.slice(resultPage * pageSize, (resultPage + 1) * pageSize).map(match => button(...match)).join('')}
      </div>${pagination('results', resultPage, matches.length)}`;
  }

  function render() {
    const node = index.nodes.get(treeId);
    const chunk = index.chunks.get(chunkId);
    const chunks = index.data.chunks.slice(node?.lo ?? 0, node?.hi ?? index.chunks.size);
    const breadcrumbs = [];
    for (let current = node; current; current = index.data.nodes[current.parent]) breadcrumbs.unshift(current.id);
    const tree = node ? `<nav class="knowledge-breadcrumbs" aria-label="树路径">${breadcrumbs.map(id => button('tree', id, id, id === treeId)).join('')}</nav>
      <div class="tree-map"><div class="tree-root">${button('tree', node.id, `${node.id} · ${node.keywords.slice(0, 3).join(' / ') || '叶节点'}`, true)}</div>
      <div class="tree-children">${node.children.map(i => {
        const child = index.data.nodes[i];
        return button('tree', child.id, `${child.id} · ${child.hi - child.lo} 块 · ${child.keywords.slice(0, 3).join(' / ')}`);
      }).join('')}</div></div>` : '';
    const finalIds = new Set(item.methods[method].evidence_ids);
    const entity = index.entities.get(entityId);
    const selectedEdge = index.edges.get(edgeId);
    const edges = new Map();
    if (graphEnabled()) {
      const sources = chunk ? [chunk] : chunks;
      if (entityId) {
        for (const edge of index.edgesByEntity.get(entityId) || []) edges.set(edge.id, edge);
      } else {
        for (const source of sources) {
          for (const edge of index.edgesByChunk.get(source.id) || []) edges.set(edge.id, edge);
        }
      }
    }
    const visibleEdges = Array.from(edges.values()).slice(edgePage * pageSize, (edgePage + 1) * pageSize);
    const sourceText = chunk ? `<section class="knowledge-detail"><h3>${esc(chunk.id)} · 字符 ${chunk.start}–${chunk.end}</h3>
      ${chunk.concepts?.length ? `<p>${esc(chunk.concepts.join(' · '))}</p>` : ''}
      <pre class="evidence-text">${esc(chunk.text)}</pre>
      ${chunk.entity_ids?.length && graphEnabled() ? `<details><summary>本块实体（${chunk.entity_ids.length}）</summary><div class="knowledge-items">
        ${chunk.entity_ids.map(id => button('entity', id, `${id} · ${label(id)}`, id === entityId)).join('')}</div></details>` : ''}
      ${chunk.proposition_ids?.length && graphEnabled() ? `<details><summary>本块陈述（${chunk.proposition_ids.length}）</summary>
        ${chunk.proposition_ids.map(id => { const p = index.propositions.get(id); return `<p>${esc(id)} · ${p.start}–${p.end} · ${esc(p.text)}</p>`; }).join('')}</details>` : ''}
      </section>` : '';
    workspace.innerHTML = `<div class="knowledge-grid ${graphEnabled() ? '' : 'knowledge-flat'}">
      <section class="knowledge-pane"><h3>${node ? '有序原文树' : 'BM25 原文块'}</h3>${tree}
      <p class="evidence-location">${chunks.length} 个原文块 · 字符 ${chunks[0]?.start ?? 0}–${chunks.at(-1)?.end ?? 0}</p>
      <div class="knowledge-items">${chunks.slice(chunkPage * pageSize, (chunkPage + 1) * pageSize).map(record =>
        button('chunk', record.id, `${record.id}${finalIds.has(record.id) ? ' · 作答证据' : ''} · ${record.text.trim().slice(0, 58)}`, record.id === chunkId)).join('')}
      </div>${pagination('chunks', chunkPage, chunks.length)}${sourceText}</section>
      ${graphEnabled() ? `<section class="knowledge-pane"><h3>${index.data.kind === 'entity' ? '实体关系图' : '片段关联图'}</h3>
        <p class="evidence-location">${entityId ? `${esc(entityId)} 的相邻关系` : `${esc(chunkId || treeId)} 的出处关系`} · ${edges.size} 条</p>
        <p class="evidence-location">关系由索引自动抽取。共现用虚线表示。</p>
        ${drawGraph(selectedEdge ? [selectedEdge] : visibleEdges.slice(0, 5))}
        ${entity || selectedEdge ? '<button class="button" data-knowledge-reset>回到当前原文范围的关系</button>' : ''}
        <div class="knowledge-items">${visibleEdges.map(edge => button('edge', edge.id,
          `${edge.id} · ${label(edge.subject)} → ${edge.predicate} → ${label(edge.object)}`, edge.id === edgeId)).join('')}</div>
        ${pagination('edges', edgePage, edges.size)}${entity ? entityDetails(entity) : ''}${selectedEdge ? edgeDetails(selectedEdge) : ''}
      </section>` : ''}</div>`;
  }

  function drawGraph(edges) {
    if (!edges.length) return '<p class="evidence-location">这个范围内没有关系。</p>';
    const ids = [...new Set(edges.flatMap(edge => [edge.subject, edge.object]))];
    const focus = entityId && ids.includes(entityId) ? entityId : edges[0].subject;
    const neighbors = ids.filter(id => id !== focus);
    const positions = new Map([[focus, {x:310, y:172}]]);
    neighbors.forEach((id, i) => {
      const angle = Math.PI * (2 * i / neighbors.length - .5);
      positions.set(id, {x:310 + 228 * Math.cos(angle), y:172 + 132 * Math.sin(angle)});
    });
    const node = id => {
      const {x, y} = positions.get(id);
      const kind = index.entities.has(id) ? 'entity' : 'chunk';
      return `<g class="knowledge-graph-node ${id === focus ? 'focus' : ''}" data-knowledge-${kind}="${esc(id)}" role="button" tabindex="0" aria-label="${esc(id + ' ' + label(id))}">
        <title>${esc(id + ' ? ' + label(id))}</title><rect x="${x - 66}" y="${y - 15}" width="132" height="30" rx="5"/>
        <text x="${x}" y="${y + 4}">${esc(label(id).length > 17 ? label(id).slice(0, 16) + '…' : label(id))}</text></g>`;
    };
    function drawEdge(edge, i) {
      const from = positions.get(edge.subject), to = positions.get(edge.object);
      const dx = to.x - from.x, dy = to.y - from.y;
      const inset = Math.min(70 / (Math.abs(dx) || 1), 18 / (Math.abs(dy) || 1));
      const bend = (i % 3 - 1) * 17;
      const midX = (from.x + to.x) / 2 + bend, midY = (from.y + to.y) / 2 - bend;
      const path = `M${from.x + dx * inset},${from.y + dy * inset} Q${midX},${midY} ${to.x - dx * inset},${to.y - dy * inset}`;
      return `<g class="knowledge-graph-edge ${edge.kind === 'co_mention' ? 'weak' : ''}" data-knowledge-edge="${esc(edge.id)}"
        role="button" tabindex="0" aria-label="${esc(edge.id + ' ' + edge.predicate)}">
        <title>${esc(edge.id + ' ? ' + edge.predicate)}</title><path class="edge-hit" d="${path}"/>
        <path class="edge-line" d="${path}" ${edge.kind === 'predicate' ? 'marker-end="url(#knowledge-arrow)"' : ''}/>
        <text x="${midX}" y="${midY - 6}">${esc((edge.negated ? '否定 · ' : '') + edge.predicate.slice(0, 23))}</text></g>`;
    }
    return `<div class="knowledge-graph"><svg viewBox="0 0 620 345" role="group" aria-label="当前原文范围内的局部关系图">
      <defs><marker id="knowledge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>
      ${edges.map(drawEdge).join('')}${ids.map(node).join('')}</svg></div>`;
  }

  function entityDetails(entity) {
    const mentions = entity.mentions.filter(mention => !chunkId || mention.passage_id === chunkId);
    return `<section class="knowledge-detail"><h3>${esc(entity.id)} · ${esc(entity.name)}</h3>
      <p>${esc(entity.type)} · ${entity.passage_ids.length} 个出处 · ${entity.mentions.length} 次提及</p>
      <p>${esc(entity.aliases.join(' / '))}</p><div class="knowledge-items">
      ${entity.passage_ids.map(id => button('chunk', id, id, id === chunkId)).join('')}</div>
      <details><summary>${chunkId ? '当前原文块中的提及' : '全部原文提及'}（${mentions.length}）</summary>
      ${mentions.map(mention => `<p>${esc(mention.passage_id)} · ${mention.start}–${mention.end} · ${esc(mention.text)}</p>`).join('')}</details></section>`;
  }

  function edgeDetails(edge) {
    const endpoints = index.data.kind === 'entity' ? 'entity' : 'chunk';
    return `<section class="knowledge-detail"><h3>${esc(edge.id)} · ${esc(edge.predicate)}</h3>
      <div class="knowledge-items">${button(endpoints, edge.subject, label(edge.subject))} → ${button(endpoints, edge.object, label(edge.object))}</div>
      <p>${esc(edge.kind)}${edge.negated ? ' · 否定' : ''}${edge.modality?.length ? ` · 情态 ${esc(edge.modality.join(' / '))}` : ''}
      ${edge.time?.length ? ` · 时间 ${esc(edge.time.join(' / '))}` : ''}</p>
      <div class="knowledge-items">${edge.passage_ids.map(id => button('chunk', id, `回读 ${id}`, id === chunkId)).join('')}</div>
      ${(edge.proposition_ids || []).map(id => {
        const p = index.propositions.get(id);
        return `<p class="knowledge-proposition">${esc(p.passage_id)} · ${esc(id)} · ${p.start}–${p.end}<br>${esc(p.text)}</p>`;
      }).join('')}</section>`;
  }

  function showChunk(id) {
    const chunk = index.chunks.get(id);
    if (!chunk) return;
    chunkId = id;
    treeId = chunk.tree;
    edgePage = 0;
    chunkPage = treeId ? 0 : Math.floor(Number(id.slice(1)) / pageSize);
  }

  function selectItem(target) {
    const data = target.dataset;
    if (data.knowledgePage) {
      const change = Number(data.direction);
      if (data.knowledgePage === 'results') { resultPage += change; find(); return; }
      if (data.knowledgePage === 'chunks') chunkPage += change;
      if (data.knowledgePage === 'edges') edgePage += change;
    } else if (data.knowledgeTree) {
      treeId = data.knowledgeTree;
      chunkId = entityId = edgeId = null;
      chunkPage = edgePage = 0;
      const node = index.nodes.get(treeId);
      if (!node.children.length) showChunk(index.data.chunks[node.lo].id);
    } else if (data.knowledgeChunk) {
      showChunk(data.knowledgeChunk);
    } else if (data.knowledgeEntity) {
      entityId = data.knowledgeEntity;
      edgeId = null;
      edgePage = 0;
      const entity = index.entities.get(entityId);
      if (!entity.passage_ids.includes(chunkId)) showChunk(entity.passage_ids[0]);
    } else if (data.knowledgeEdge) {
      edgeId = data.knowledgeEdge;
      const edge = index.edges.get(edgeId);
      if (!edge.passage_ids.includes(chunkId)) showChunk(edge.passage_ids[0]);
    } else {
      entityId = edgeId = null;
      edgePage = 0;
    }
    render();
  }

  const targets = '[data-knowledge-page],[data-knowledge-tree],[data-knowledge-chunk],[data-knowledge-entity],[data-knowledge-edge],[data-knowledge-reset]';
  host.addEventListener('click', event => {
    const target = event.target.closest(targets);
    if (target && index) selectItem(target);
  });
  host.addEventListener('keydown', event => {
    const target = event.target.closest('svg [role="button"]');
    if (target && ['Enter', ' '].includes(event.key) && index) {
      event.preventDefault();
      selectItem(target);
    }
  });
  panel.addEventListener('toggle', () => { if (panel.open && !index) load(); });
  select.addEventListener('change', load);
  search.addEventListener('input', () => { resultPage = 0; find(); });
  finalButton.addEventListener('click', () => {
    const id = item.methods[method].evidence_ids.find(id => index.chunks.has(id));
    entityId = edgeId = null;
    showChunk(id);
    render();
  });
  if (eager) load();
}

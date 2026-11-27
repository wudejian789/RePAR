const COLORS = {
  mint: '#078576', cyan: '#347fa5', lilac: '#5968b8',
  amber: '#c27a1f', coral: '#d65d69', ink: '#263331',
};
const NODE_LIMIT = 64;
const RELATION_LIMIT = 88;
const STATEMENT_LIMIT = 38;
const SOURCE_LIMIT = 28;

function shortLabel(text, limit = 32) {
  const words = String(text || '').replace(/\s+/g, ' ').trim();
  return words.length > limit ? words.slice(0, limit - 1) + '…' : words;
}

function searchTerms(value) {
  const query = String(value || '').toLocaleLowerCase();
  const terms = new Set(query.match(/[\p{L}\p{N}_-]{3,}/gu) || []);
  for (const id of query.match(/\b[ceprt]\d+\b/g) || []) terms.add(id);
  for (const run of query.match(/[\p{Script=Han}]{2,}/gu) || []) {
    if (run.length <= 4) terms.add(run);
    else {
      const step = Math.max(1, Math.floor((run.length - 2) / 4));
      for (let start = 0; start < run.length - 1; start += step) {
        terms.add(run.slice(start, start + 2));
      }
    }
  }
  return [...terms].slice(0, 12);
}

function matchesTerms(record, terms) {
  const fields = [record.name, record.aliases, record.text, record.preview, record.propositionText,
    record.time, record.modality].filter(Boolean).join(' ').toLocaleLowerCase();
  return terms.some(term => /^[ceprt]\d+$/.test(term)
    ? record.id.toLowerCase() === term
    : fields.includes(term));
}

function connectionStatus() {
  return '自动抽取 · 待原文核验';
}

export function mountKnowledgeGraph(host, library, data, renderSource) {
  const degree = new Map();
  for (const {subject, object} of data.relations) {
    degree.set(subject, (degree.get(subject) || 0) + 1);
    if (subject !== object) degree.set(object, (degree.get(object) || 0) + 1);
  }
  const entityRecords = [...data.entities].sort((a, b) =>
    (degree.get(b.id) || 0) - (degree.get(a.id) || 0) ||
    b.mentions.length - a.mentions.length || a.id.localeCompare(b.id));
  const entities = entityRecords.slice(0, NODE_LIMIT);
  const entityIds = new Set(entities.map(item => item.id));
  const relations = data.relations.filter(item =>
    entityIds.has(item.subject) && entityIds.has(item.object));
  const visibleRelations = relations.slice(0, RELATION_LIMIT);
  const provenanceChunkIds = new Set();
  const propositionIds = new Set();
  for (const entity of entities) {
    for (const id of entity.passage_ids) provenanceChunkIds.add(id);
  }
  for (const relation of visibleRelations) {
    for (const id of relation.passage_ids) provenanceChunkIds.add(id);
    for (const id of relation.proposition_ids) propositionIds.add(id);
  }
  const passages = data.chunks.filter(item => provenanceChunkIds.has(item.id)).slice(0, SOURCE_LIMIT);
  const passageIds = new Set(passages.map(item => item.id));
  const statements = data.propositions.filter(item =>
    propositionIds.has(item.id) && passageIds.has(item.passage_id)).slice(0, STATEMENT_LIMIT);

  const treeFallback = entities.length === 0;
  const graph = {entities, relations: visibleRelations, passages, propositions: statements,
    treeNodes: data.nodes || [], treeChunks: data.chunks || []};
  const truncated = entityRecords.length > NODE_LIMIT || relations.length > RELATION_LIMIT ||
    data.propositions.length > STATEMENT_LIMIT || data.chunks.length > SOURCE_LIMIT;
  const hostElement = document.createElement('div');
  hostElement.className = 'knowledge-graph-shell';
  hostElement.innerHTML = `<header class="graph-toolbar">
    <div class="graph-identity"><span class="graph-index-mark" aria-hidden="true"><span></span><span></span><span></span></span><strong>${treeFallback ? '原文树' : '实体关系图'}</strong></div>
    <div class="graph-tools"><label class="graph-search"><span class="graph-search-icon"></span><input id="kg-search" type="search" aria-label="搜索图谱" placeholder="搜索实体、关系或出处" autocomplete="off"></label>
    <div class="graph-modes" role="group" aria-label="图谱视图"><button data-view="2d" class="active" aria-pressed="true"><i class="mode-2d-icon"></i><span>2D</span></button><button data-view="3d" aria-pressed="false"><i class="mode-3d-icon"></i><span>3D</span></button></div>
      <button id="kg-fit" class="graph-icon-button" aria-label="适应图谱" title="适应图谱"></button>
    </div>
  </header>
  <div class="graph-insight-strip">
    <div class="graph-stat"><span class="stat-mark entity-mark"></span><span class="stat-value" data-count="entities"></span><span class="stat-name">实体</span></div>
    <span class="stat-divider"></span>
    <div class="graph-stat"><span class="stat-mark relation-mark"></span><span class="stat-value" data-count="relations"></span><span class="stat-name">关系线索</span></div>
    <span class="stat-divider"></span>
    <div class="graph-stat"><span class="stat-mark evidence-mark"></span><span class="stat-value" data-count="passages"></span><span class="stat-name">出处片段</span></div>
  </div>
  <div class="graph-stage">
  <div class="graph-stage-inner">
    <div class="graph-plot">
    <div class="knowledge-map" id="knowledge-graph" role="img" aria-label="实体、关系线索与原文出处图"></div>
    <canvas class="knowledge-map-3d" id="knowledge-graph-3d" aria-label="三维实体关系图，拖动旋转，滚轮缩放" hidden></canvas>
    <div class="map-legend"><span><i class="legend-entity"></i>实体 / 概念</span><span><i class="legend-statement"></i>有出处的陈述</span><span><i class="legend-source"></i>原文片段</span><span><i class="legend-relationship"></i>抽取关系</span></div>
    <div class="map-zoom"><button id="kg-zoom-in" aria-label="放大图谱" title="放大图谱"></button><button id="kg-zoom-out" aria-label="缩小图谱" title="缩小图谱"></button><button id="kg-center" aria-label="居中所选项" title="居中所选项"></button></div>
    <div class="map-counter" role="status"></div>
  </div>
  <aside class="graph-detail-host" id="knowledge-detail-host">
    <section class="graph-blocks" aria-label="知识库原文块">
      <header class="graph-blocks-heading">
        <h3>原文块 <span>${data.chunks.length.toLocaleString()}</span></h3>
        <button id="read-graph-block" type="button" class="outline-control" aria-controls="graph-source" aria-expanded="false" disabled>查看原文</button>
      </header>
      <div id="graph-block-list"></div>
    </section>
    <div class="graph-inspector" hidden>
      <section id="graph-record" aria-label="图谱记录详情" hidden></section>
      <section id="graph-source" class="graph-source" aria-label="原文出处" aria-live="polite" hidden></section>
    </div>
  </aside>
  </div>
  <footer class="graph-caption"><div class="graph-citation"><span class="citation-mark" aria-hidden="true">i</span><span>${truncated ? '画布展示部分索引记录，可在「完整索引」中检索全库。' : '点击节点或连线查看原文出处。'}</span></div><button id="graph-reset" class="graph-text-button" type="button">重置视图</button></footer>
  </div>`;
  host.replaceChildren(hostElement);

  hostElement.querySelector('[data-count="entities"]').textContent = data.entities.length.toLocaleString();
  hostElement.querySelector('[data-count="relations"]').textContent = data.relations.length.toLocaleString();
  hostElement.querySelector('[data-count="passages"]').textContent = data.chunks.length.toLocaleString();

  const layoutOptions = treeFallback ? {name: 'breadthfirst', directed: true, animate: false,
    spacingFactor: 1.22, padding: 60, nodeDimensionsIncludeLabels: true,
    roots: `#${data.nodes?.[0]?.id || 'T0'}`} : {name: 'cose', animate: false,
    animationDuration: 1200, fit: true, padding: window.innerWidth < 580 ? 28 : 48, nodeRepulsion: 9400,
    idealEdgeLength: edge => edge.data('provenance') ? 82 : 142,
    edgeElasticity: 100, gravity: 0.28, numIter: 420, componentSpacing: 108,
    nodeOverlap: 14, tile: true, randomize: false};
  const cy = window.cytoscape({
    container: hostElement.querySelector('#knowledge-graph'),
    elements: treeFallback ? buildTreeElements(graph) : buildElements(graph),
    minZoom: 0.16,
    maxZoom: 4,
    wheelSensitivity: 0.18,
    boxSelectionEnabled: false,
    motionBlur: false,
    pixelRatio: Math.min(window.devicePixelRatio || 1, 2),
    style: cytoscapeStyle(),
    layout: layoutOptions,
  });
  let disposed = false, renderer3d = null, filterText = '';
  const detail = hostElement.querySelector('#graph-record');
  const inspector = hostElement.querySelector('.graph-inspector');
  const readBlock = hostElement.querySelector('#read-graph-block');
  const sourceHost = hostElement.querySelector('#graph-source');
  let activeSource, activeSourceId;
  const selectedBlocks = new Set();
  const chunks = new Map(data.chunks.map(chunk => [chunk.id, chunk]));
  const overview = new Set(cy.elements().map(item => item.id()));
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const canvas = hostElement.querySelector('#knowledge-graph-3d');
  const counter = hostElement.querySelector('.map-counter');
  const graphCount = treeFallback
    ? `显示 ${Math.min(graph.treeNodes.length, NODE_LIMIT)} / ${graph.treeNodes.length} 个树节点`
    : `显示 ${entities.length} / ${data.entities.length} 个实体`;
  counter.textContent = graphCount;
  const isThreeD = () => !canvas.hidden;
  const padding = window.innerWidth < 580 ? 28 : 48;
  const fit = () => {
    if (isThreeD()) renderer3d?.fit();
    else cy.stop().fit(cy.elements(), padding);
  };
  let sourceIds = new Set();
  let localBlock;

  const blockList = mountBlockList(hostElement.querySelector('#graph-block-list'), data.chunks, {
    selected: selectedBlocks, onPick: chunk => selectBlock(chunk.id),
  });
  readBlock.addEventListener('click', () => showSource(selectedBlocks.values().next().value, readBlock));

  function closeSource() {
    activeSource?.setAttribute('aria-expanded', 'false');
    activeSource = activeSourceId = undefined;
    sourceHost.hidden = true;
    sourceHost.replaceChildren();
    inspector.hidden = detail.hidden;
  }

  function showSource(id, trigger) {
    const wasOpen = activeSourceId === id && activeSource === trigger;
    closeSource();
    if (wasOpen) return;
    const passage = chunks.get(id);
    if (!passage) return;
    activeSource = trigger;
    activeSourceId = id;
    trigger.setAttribute('aria-expanded', 'true');
    const collapse = document.createElement('button');
    collapse.type = 'button';
    collapse.className = 'text-button';
    collapse.textContent = '收起原文';
    collapse.addEventListener('click', () => {
      closeSource();
      trigger.focus({preventScroll: true});
      if (detail.contains(trigger)) {
        inspector.scrollTop += trigger.getBoundingClientRect().top - inspector.getBoundingClientRect().top;
      }
    });
    sourceHost.replaceChildren(collapse, renderSource(passage));
    sourceHost.hidden = false;
    inspector.hidden = false;
    inspector.scrollTop += sourceHost.getBoundingClientRect().top - inspector.getBoundingClientRect().top;
  }

  function blockLinks(record) {
    if (record.kind === 'passage') return [record.id];
    if (record.passageId) return [record.passageId];
    if (record.passageIds?.length) return record.passageIds;
    if (chunks.has(record.target)) return [record.target];
    if (record.kind === 'tree') return data.chunks.slice(record.lo, record.hi).map(chunk => chunk.id);
    const target = cy.getElementById(record.target || '').data();
    return target?.passageId ? [target.passageId] : [];
  }

  function highlightBlocks(ids, active) {
    selectedBlocks.clear();
    for (const id of ids) if (chunks.has(id)) selectedBlocks.add(id);
    const selectedBlock = selectedBlocks.values().next().value;
    readBlock.disabled = !selectedBlock;
    readBlock.textContent = selectedBlock ? `查看 ${selectedBlock} 原文` : '查看原文';
    if (selectedBlock) blockList.reveal(selectedBlock);
    else blockList.refreshSelection();

    const hits = cy.elements().filter(item => item.id() === active ||
      (item.data('kind') !== 'tree' && blockLinks(item.data()).some(id => selectedBlocks.has(id))));
    const endpoints = hits.edges().connectedNodes().union(hits.nodes());
    cy.batch(() => {
      cy.elements().removeClass('block-match block-dim');
      if (hits.length) {
        hits.union(endpoints).addClass('block-match');
        cy.elements().difference(hits.union(endpoints)).addClass('block-dim');
      }
    });
    renderer3d?.filter(new Set(endpoints.map(item => item.id())), new Set(hits.nodes().map(item => item.id())),
      new Set(hits.edges().map(item => item.id())), Boolean(hits.length));
    return hits.union(endpoints);
  }

  function select(id) {
    const element = cy.getElementById(id);
    if (!element.length) return;
    closeSource();
    cy.elements().unselect().removeClass('selected neighbor');
    element.select().addClass('selected');
    element.neighborhood().addClass('neighbor');
    const isPassage = element.data('kind') === 'passage';
    detail.replaceChildren(...(isPassage ? [] : [renderDetails(element.data(), data, showSource)]));
    detail.hidden = isPassage;
    inspector.hidden = isPassage;
    inspector.scrollTop = 0;
    const related = blockLinks(element.data());
    const current = [...selectedBlocks].find(block => related.includes(block)) || related[0];
    const focusElements = highlightBlocks(current ? [current] : [], id);
    renderer3d?.select(id);
    if (isThreeD()) {
      renderer3d?.center();
    } else {
      focus2d(element, focusElements);
    }
  }

  function focus2d(element, related) {
    cy.stop();
    const gutter = Math.max(27, Math.min(cy.width(), cy.height()) * .08);
    const fitted = cy.getFitViewport(related.length ? related : element, gutter);
    const zoom = Math.min(cy.maxZoom(), Math.max(1.3, fitted?.zoom ?? 1.3));
    const bounds = (fitted?.zoom >= 1.3 && related.length ? related : element).boundingBox({includeOverlays: false});
    const pan = {
      x: (cy.width() - (bounds.x1 + bounds.x2) * zoom) / 2,
      y: (cy.height() - (bounds.y1 + bounds.y2) * zoom) / 2,
    };
    if (motion.matches) cy.viewport({zoom, pan});
    else cy.animate({zoom, pan}, {duration: 420, easing: 'ease-out-cubic'});
  }

  function selectBlock(id) {
    const chunk = chunks.get(id);
    if (!chunk) return;
    const rebuild = localBlock !== id;
    const view = rebuild ? renderer3d?.view() : undefined;
    if (rebuild) {
      localBlock = id;
      cy.elements().filter(item => !overview.has(item.id())).remove();
      const visible = new Set((chunk.entity_ids || []).slice(0, 30));
      for (const entity of entities) if (entity.passage_ids.includes(id)) visible.add(entity.id);
      const local = buildElements({
        entities: data.entities.filter(item => visible.has(item.id)),
        relations: data.relations.filter(item => visible.has(item.subject) && visible.has(item.object) &&
          item.passage_ids.includes(id)).slice(0, 48),
        passages: [chunk],
        propositions: data.propositions.filter(item => item.passage_id === id).slice(0, 20),
      });
      if (treeFallback && cy.getElementById(chunk.tree || '').length) {
        local.push({data: {id: `${chunk.tree}:${id}`, source: chunk.tree, target: id, kind: 'provenance', caption: ''}});
      }
      cy.add(local.filter(item => !cy.getElementById(item.data.id).length));
      cy.layout({...layoutOptions, fit: false}).run();
      renderer3d?.stop();
      renderer3d = null;
    }
    select(id);
    if (isThreeD() && rebuild) open3d(view);
  }

  function visibleFilter(query) {
    cy.elements().removeClass('block-match block-dim');
    filterText = query.trim().toLocaleLowerCase();
    const terms = searchTerms(filterText);
    const matchingNodes = new Set(cy.nodes().filter(item => matchesTerms(item.data(), terms)).map(item => item.id()));
    const matchingEdges = new Set(cy.edges().filter(item => matchesTerms(item.data(), terms)).map(item => item.id()));
    const focusNodes = new Set(matchingNodes);
    for (const id of matchingEdges) {
      for (const node of cy.getElementById(id).connectedNodes()) focusNodes.add(node.id());
    }
    for (const id of [...focusNodes]) {
      for (const node of cy.getElementById(id).neighborhood('node')) focusNodes.add(node.id());
    }
    const hasMatches = terms.length > 0 && (matchingNodes.size > 0 || matchingEdges.size > 0);

    cy.nodes().forEach(item => {
      const matched = matchingNodes.has(item.id());
      const nearby = focusNodes.has(item.id());
      item.toggleClass('focus-match', matched);
      item.toggleClass('context-dim', hasMatches && !nearby);
    });
    cy.edges().forEach(item => {
      const matched = matchingEdges.has(item.id());
      const nearby = focusNodes.has(item.data('source')) || focusNodes.has(item.data('target'));
      item.toggleClass('focus-match', matched);
      item.toggleClass('context-dim', hasMatches && !nearby && !matched);
    });
    counter.textContent = filterText
      ? `${graphCount} · ${hasMatches ? `匹配 ${matchingNodes.size + matchingEdges.size} 项` : '无匹配'}`
      : graphCount;
    renderer3d?.filter(focusNodes, matchingNodes, matchingEdges, hasMatches);
  }

  cy.on('tap', 'node', event => select(event.target.id()));
  cy.on('tap', 'edge', event => select(event.target.id()));
  cy.on('mouseover', 'edge', event => event.target.addClass('hovered'));
  cy.on('mouseout', 'edge', event => event.target.removeClass('hovered'));
  hostElement.querySelector('#kg-search').addEventListener('input', event => visibleFilter(event.target.value));
  hostElement.querySelector('#kg-fit').append(svgIcon('focus', 15));
  hostElement.querySelector('#kg-fit').addEventListener('click', fit);
  function zoom(factor) {
    if (isThreeD()) renderer3d?.zoom(factor);
    else cy.zoom({level: cy.zoom() * factor, renderedPosition: {x: cy.width() / 2, y: cy.height() / 2}});
  }
  hostElement.querySelector('#kg-zoom-in').append(svgIcon('zoom-in', 15));
  hostElement.querySelector('#kg-zoom-in').addEventListener('click', () => zoom(1.24));
  hostElement.querySelector('#kg-zoom-out').append(svgIcon('zoom-out', 15));
  hostElement.querySelector('#kg-zoom-out').addEventListener('click', () => zoom(1 / 1.24));
  hostElement.querySelector('#kg-center').append(svgIcon('locate-fixed', 15));
  hostElement.querySelector('#kg-center').addEventListener('click', () => {
    if (isThreeD()) {
      renderer3d?.center();
      return;
    }
    const selected = cy.$(':selected');
    if (selected.length) cy.center(selected);
    else fit();
  });
  hostElement.querySelector('#graph-reset').addEventListener('click', () => {
    closeSource();
    cy.stop();
    hostElement.querySelector('#kg-search').value = '';
    visibleFilter('');
    cy.elements().unselect().removeClass('selected neighbor');
    const changed = cy.elements().some(item => !overview.has(item.id()));
    localBlock = undefined;
    cy.elements().filter(item => !overview.has(item.id())).remove();
    if (changed) {
      renderer3d?.stop();
      renderer3d = null;
      if (isThreeD()) open3d();
    } else renderer3d?.reset();
    detail.replaceChildren();
    detail.hidden = true;
    inspector.hidden = true;
    selectedBlocks.clear();
    readBlock.disabled = true;
    readBlock.textContent = '查看原文';
    blockList.refreshSelection();
    cy.layout(layoutOptions).run();
  });
  for (const tab of hostElement.querySelectorAll('.graph-modes button')) {
    tab.addEventListener('click', async () => {
      const threeD = tab.dataset.view === '3d';
      for (const control of hostElement.querySelectorAll('.graph-modes button')) {
        const active = control === tab;
        control.classList.toggle('active', active);
        control.setAttribute('aria-pressed', String(active));
      }
      hostElement.querySelector('#knowledge-graph').hidden = threeD;
      canvas.hidden = !threeD;
      const selected = cy.$(':selected').first();
      if (threeD) {
        if (!renderer3d) await open3d();
        else if (selected.length) renderer3d.center();
        renderer3d?.resume();
      } else {
        renderer3d?.pause();
        cy.resize();
        if (selected.length) focus2d(selected, cy.elements('.block-match'));
      }
    });
  }
  async function open3d(view) {
    const tab = hostElement.querySelector('[data-view="3d"]');
    tab.disabled = true;
    try {
      const THREE = await import('./vendor/three.module.js');
      if (disposed || !isThreeD() || renderer3d) return;
      renderer3d = make3d(THREE, canvas, cy, select, view);
      const id = cy.$(':selected').first().id();
      renderer3d.markSources(sourceIds);
      if (id) {
        highlightBlocks([...selectedBlocks], id);
        renderer3d.select(id);
        renderer3d.center();
      } else visibleFilter(filterText);
      renderer3d.resume();
    } catch (error) {
      console.warn('Unable to initialize the 3D knowledge graph', error);
      hostElement.querySelector('[data-view="2d"]').click();
      counter.textContent = `${graphCount} · 3D 需要 WebGL 2`;
    } finally {
      tab.disabled = false;
    }
  }
  const observer = new ResizeObserver(() => {
    if (disposed) return;
    cy.resize();
  });
  observer.observe(hostElement.querySelector('.graph-stage'));

  return {
    libraryId: library.id,
    stop() {
      if (disposed) return;
      disposed = true;
      observer.disconnect();
      renderer3d?.stop();
      cy.stop();
      cy.destroy();
    },
    select,
    selectBlock,
    setVisible(visible) {
      if (!visible) {
        renderer3d?.pause();
        return;
      }
      requestAnimationFrame(() => {
        if (disposed) return;
        cy.resize();
        fit();
        renderer3d?.resize();
        if (isThreeD()) renderer3d?.resume();
      });
    },
    focus(query) {
      hostElement.querySelector('#kg-search').value = query;
      visibleFilter(query);
    },
    markSources(ids) {
      sourceIds = new Set(ids);
      cy.elements().removeClass('trace-source');
      for (const id of sourceIds) {
        const element = cy.getElementById(id);
        if (element.length) element.addClass('trace-source');
      }
      renderer3d?.markSources(sourceIds);
    },
    fit,
  };
}

function buildElements({entities, relations, passages, propositions}) {
  const items = [];
  for (const entity of entities) {
    const type = entity.type === 'PERSON' ? 'person' : entity.type === 'ORG' ? 'organization' : 'concept';
    items.push({data: {id: entity.id, name: entity.name, caption: entity.name,
      type, kind: 'entity', degree: entity.relation_count || entity.relation_ids?.length || 1,
      aliases: (entity.aliases || []).join(' · '), passageIds: entity.passage_ids, mentionCount: entity.mentions.length}});
  }
  for (const passage of passages) {
    items.push({data: {id: passage.id, name: passage.id, caption: passage.id,
      type: 'passage', kind: 'passage', start: passage.start, end: passage.end,
      preview: passage.text}});
  }
  for (const statement of propositions) {
    items.push({data: {id: statement.id, name: statement.id, caption: shortLabel(statement.text, 42),
      type: 'statement', kind: 'statement', passageId: statement.passage_id,
      start: statement.start, end: statement.end, text: statement.text}});
  }
  for (const entity of entities) {
    for (const propositionId of entity.proposition_ids || []) {
      const statement = propositions.find(item => item.id === propositionId);
      if (!statement) continue;
      items.push({data: {id: `${entity.id}:${propositionId}`, source: entity.id,
        target: propositionId, kind: 'mention', caption: '', provenance: true}});
    }
    for (const passageId of entity.passage_ids || []) {
      if (!passages.some(item => item.id === passageId) ||
          (entity.proposition_ids || []).some(id => propositions.some(item => item.id === id && item.passage_id === passageId))) continue;
      items.push({data: {id: `${entity.id}:${passageId}`, source: entity.id,
        target: passageId, kind: 'mention', caption: '', provenance: true}});
    }
  }
  for (const statement of propositions) {
    items.push({data: {id: `${statement.id}:${statement.passage_id}`, source: statement.id,
      target: statement.passage_id, kind: 'provenance', caption: '', provenance: true}});
  }
  for (const relation of relations) {
    const proposition = propositions.find(item => relation.proposition_ids?.includes(item.id));
    items.push({data: {id: relation.id, source: relation.subject, target: relation.object,
      name: relation.predicate, caption: relation.predicate, kind: relation.kind, recordType: 'relation',
      relationId: relation.id, negated: relation.negated, time: (relation.time || []).join(' · '),
      modality: (relation.modality || []).join(' · '), propositionIds: relation.proposition_ids,
      passageIds: relation.passage_ids, propositionText: proposition?.text,
      status: relation.status || 'automatically_extracted_not_verified'}});
  }
  return items;
}

function buildTreeElements({treeNodes, treeChunks}) {
  const nodes = treeNodes.slice(0, NODE_LIMIT);
  const allowed = new Set(nodes.map(item => item.id));
  const elements = nodes.map(item => ({data: {id: item.id, type: 'tree-node', kind: 'tree', caption: item.id,
    lo: item.lo, hi: item.hi, parent: item.parent, children: item.children}}));
  for (const item of nodes) {
    const parent = typeof item.parent === 'number' ? `T${item.parent}` : item.parent;
    if (parent && allowed.has(parent)) {
      elements.push({data: {id: `${parent}->${item.id}`, source: parent, target: item.id,
        kind: 'tree_branch', caption: ''}});
    }
  }
  for (const chunk of treeChunks.slice(0, SOURCE_LIMIT)) {
    elements.push({data: {id: chunk.id, type: 'passage', kind: 'passage', caption: chunk.id,
      start: chunk.start, end: chunk.end, preview: chunk.text}});
    if (chunk.tree && allowed.has(chunk.tree)) {
      elements.push({data: {id: `${chunk.tree}:${chunk.id}`, source: chunk.tree, target: chunk.id,
        kind: 'provenance', caption: ''}});
    }
  }
  return elements;
}

function cytoscapeStyle() {
  return [
    {selector: 'node', style: {label: 'data(caption)', 'font-family': 'DM Sans, sans-serif',
      'font-size': 12, 'min-zoomed-font-size': 9, color: COLORS.ink, 'text-wrap': 'wrap', 'text-max-width': 110,
      'text-valign': 'center', 'text-halign': 'center', 'border-width': 1,
      'border-color': '#ffffff', 'background-color': '#e3efed',
      width: 48, height: 40,
      'overlay-opacity': 0, 'text-outline-color': '#ffffff', 'text-outline-width': 2}},
    {selector: 'node[type = "person"]', style: {'background-color': '#fde4e7', 'border-color': COLORS.coral,
      width: 58, height: 58, shape: 'ellipse'}},
    {selector: 'node[type = "organization"]', style: {'background-color': '#e7e9fb', 'border-color': COLORS.lilac,
      shape: 'round-rectangle', width: 64, height: 44}},
    {selector: 'node[type = "concept"]', style: {'background-color': '#d9f1eb', 'border-color': COLORS.mint,
      shape: 'round-rectangle', width: 57, height: 43}},
    {selector: 'node[type = "statement"]', style: {'background-color': '#fff0d8', 'border-color': COLORS.amber,
      shape: 'round-rectangle', width: 104, height: 42, 'font-size': 11, 'text-max-width': 92}},
    {selector: 'node[type = "passage"]', style: {'background-color': '#e1eff7', 'border-color': COLORS.cyan,
      shape: 'round-rectangle', width: 54, height: 34, 'font-family': 'IBM Plex Mono, monospace'}},
    {selector: 'edge', style: {width: 1.5, 'line-color': '#7c918e',
      'target-arrow-color': '#7c918e', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier',
      'line-cap': 'round', 'arrow-scale': .76, label: 'data(caption)', 'font-family': 'IBM Plex Mono, monospace',
      'font-size': 10, 'min-zoomed-font-size': 11, color: '#475a57', 'text-background-color': '#ffffff',
      'text-background-opacity': .92, 'text-background-padding': 3,
      'text-margin-y': -9, 'overlay-opacity': 0}},
    {selector: 'edge[kind = "predicate"]', style: {'line-color': '#6572bf', 'target-arrow-color': '#6572bf', 'line-style': 'dashed'}},
    {selector: 'edge[kind = "co_mention"]', style: {'line-color': '#a5b0ae', 'target-arrow-shape': 'none', 'line-style': 'dotted', label: ''}},
    {selector: 'edge[kind = "mention"], edge[kind = "provenance"]', style: {'line-color': '#55a99b',
      'target-arrow-color': '#55a99b', width: 1.2, 'line-style': 'dotted', 'target-arrow-shape': 'chevron', label: ''}},
    {selector: 'edge[?negated]', style: {'line-color': COLORS.coral, 'target-arrow-color': COLORS.coral, 'line-style': 'dashed'}},
    {selector: 'node.context-dim', style: {opacity: .16, 'text-opacity': .26}},
    {selector: 'edge.context-dim', style: {opacity: .1, 'text-opacity': 0}},
    {selector: 'node.block-dim', style: {opacity: .24, 'text-opacity': 0}},
    {selector: 'edge.block-dim', style: {opacity: .17, 'text-opacity': 0}},
    {selector: 'node.block-match', style: {opacity: 1, 'text-opacity': 1, 'border-width': 2.5, 'border-color': '#087f73'}},
    {selector: 'edge.block-match', style: {opacity: 1, 'text-opacity': 1, width: 2.5}},
    {selector: 'node.focus-match', style: {'border-width': 3, 'border-color': '#173f3b', 'overlay-color': '#62cabc', 'overlay-opacity': .17}},
    {selector: 'edge.focus-match', style: {width: 3, 'line-color': '#008f80', 'target-arrow-color': '#008f80', 'z-index': 8}},
    {selector: 'node.neighbor', style: {'border-width': 3, 'border-color': '#375f59'}},
    {selector: 'node.selected', style: {opacity: 1, 'text-opacity': 1, 'font-weight': 'bold', 'border-width': 4, 'border-color': '#087f73', 'underlay-color': '#81c7b9', 'underlay-opacity': .3, 'underlay-padding': 8}},
    {selector: 'node.trace-source', style: {opacity: 1, 'text-opacity': 1, 'border-width': 4, 'border-color': '#d28b28'}},
    {selector: 'edge.selected, edge.hovered', style: {width: 3, 'line-color': '#d28b28', 'target-arrow-color': '#d28b28', 'z-index': 8}},
    {selector: 'edge.trace-source', style: {width: 2.7, 'line-color': '#d28b28', 'target-arrow-color': '#d28b28'}},
  ];
}

function renderDetails(record, data, onSource) {
  const card = document.createElement('article');
  card.className = 'entity-detail';
  const header = document.createElement('header');
  const nodeTypes = {person: '人物', organization: '组织', concept: '实体 / 概念',
    passage: '原文片段', statement: '有出处的陈述', 'tree-node': '树节点'};
  const linkTypes = {mention: '实体提及', provenance: '出处映射', tree_branch: '树的父子关系'};
  const title = record.recordType === 'relation' ? '抽取关系' : nodeTypes[record.type] || linkTypes[record.kind];
  header.innerHTML = `<span class="detail-token">${escapeHTML(title)}</span><span class="detail-record-id">${escapeHTML(record.id)}</span>`;
  card.append(header);
  if (record.type === 'tree-node') {
    card.append(textField('原文块范围', `${record.lo}–${record.hi} · ${Math.max(0, record.hi - record.lo)} 个叶节点`));
    for (const chunk of data.chunks.slice(record.lo, record.hi).slice(0, 8)) card.append(sourceButton(chunk.id, onSource));
    if (record.hi - record.lo > 8) card.append(textField('其他原文块', `${record.hi - record.lo - 8} 个，按需浏览完整索引`));
  } else if (record.recordType === 'relation') {
    const directed = document.createElement('div');
    directed.className = 'relation-direction';
    const subject = data.entities.find(item => item.id === record.source);
    const object = data.entities.find(item => item.id === record.target);
    directed.innerHTML = `<strong>${escapeHTML(subject?.name || record.source)}</strong><span>${record.negated ? '≠' : '→'}</span><strong>${escapeHTML(object?.name || record.target)}</strong>`;
    card.append(directed);
    card.append(textField('谓词', record.name || record.predicate));
    card.append(statusPill(connectionStatus()));
    if (record.time) card.append(textField('时间限定', record.time));
    if (record.modality) card.append(textField('情态限定', record.modality));
    for (const id of record.propositionIds || []) {
      const item = data.propositions.find(value => value.id === id);
      if (!item) continue;
      card.append(quoteCard(item.text, `${item.id} · 字符 ${item.start}–${item.end}`));
      card.append(sourceButton(item.passage_id, onSource));
    }
    if (!record.propositionIds?.length) {
      for (const id of record.passageIds || []) card.append(sourceButton(id, onSource));
    }
  } else if (record.type === 'statement') {
    card.append(textField('原文跨度', `${record.start}–${record.end}`));
    card.append(quoteCard(record.text, record.passageId));
    card.append(sourceButton(record.passageId, onSource));
  } else if (record.source && record.target) {
    card.append(textField('连接记录', `${record.source} → ${record.target}`));
    const passageId = data.propositions.find(item => item.id === record.target)?.passage_id || record.target;
    if (data.chunks.some(item => item.id === passageId)) card.append(sourceButton(passageId, onSource));
  } else {
    card.append(textField('实体名称', record.name));
    if (record.aliases) card.append(textField('别名', record.aliases));
    if (record.mentionCount) card.append(textField('原文提及', `${record.mentionCount} 次`));
    const pids = [...new Set(data.relations.filter(item => item.subject === record.id || item.object === record.id)
      .flatMap(item => item.proposition_ids || []))];
    const evidences = new Set([...record.passageIds || [], ...pids.map(id => data.propositions.find(p => p.id === id)?.passage_id).filter(Boolean)]);
    for (const id of evidences) card.append(sourceButton(id, onSource));
  }
  return card;
}

function textField(label, value) {
  const field = document.createElement('div');
  field.className = 'detail-field';
  field.append(document.createElement('span'), document.createElement('strong'));
  field.firstElementChild.textContent = label.toUpperCase();
  field.lastElementChild.textContent = value || '—';
  return field;
}

function quoteCard(text, caption) {
  const quote = document.createElement('blockquote');
  quote.className = 'detail-quote';
  quote.textContent = text;
  if (caption) quote.setAttribute('cite', caption);
  return quote;
}

function statusPill(text) {
  const item = document.createElement('span');
  item.className = 'relation-status';
  item.append(svgIcon('circle-help', 13), document.createTextNode(text));
  return item;
}

function sourceButton(id, onSource) {
  const wrap = document.createElement('div');
  wrap.className = 'detail-source-link';
  const item = document.createElement('span');
  item.className = 'detail-source-id';
  item.append(svgIcon('file-text', 13), document.createTextNode(id));
  const link = document.createElement('button');
  link.type = 'button';
  link.setAttribute('aria-controls', 'graph-source');
  link.setAttribute('aria-expanded', 'false');
  link.append(document.createTextNode('查看原文'), svgIcon('chevron-down', 13));
  link.addEventListener('click', () => onSource(id, link));
  wrap.append(item, link);
  return wrap;
}

function make3d(THREE, canvas, cy, onSelect, view) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true, preserveDrawingBuffer: true});
  } catch (error) {
    throw new Error(`WebGL is unavailable: ${error.message}`);
  }
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 1, 5000);
  const target = new THREE.Vector3();
  const center = view?.center.clone() || new THREE.Vector3();
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const interaction = new AbortController();
  let distance = view?.distance ?? 800;
  let transition, framedIds;
  scene.add(new THREE.AmbientLight(0xffffff, 1.9));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
  keyLight.position.set(-140, 220, 280);
  scene.add(keyLight);

  const graphRoot = new THREE.Group();
  if (view) graphRoot.rotation.copy(view.rotation);
  else graphRoot.rotation.set(-.12, .28, 0);
  scene.add(graphRoot);
  const nodeObjects = new Map();
  const labelObjects = new Map();
  const edgeObjects = [];
  let selected = '';
  let active = true;
  let pointer = null;
  let raf = 0;
  let hasMatches = false;
  let focusNodes = new Set();
  let matchingNodes = new Set();
  let matchingEdges = new Set();
  let tracedSources = new Set();

  const nodeRecords = cy.nodes().map(item => ({...item.data()}));
  const layers = new Map();
  for (const record of nodeRecords) {
    if (!layers.has(record.kind)) layers.set(record.kind, []);
    layers.get(record.kind).push(record);
  }
  for (const records of layers.values()) records.sort((a, b) => a.id.localeCompare(b.id));
  const layerPhases = {entity: 0, statement: 1.3, passage: 2.6, tree: .7};
  const nodes = nodeRecords.map(record => {
    const peers = layers.get(record.kind);
    const index = peers.findIndex(item => item.id === record.id);
    const y = 1 - 2 * (index + .5) / peers.length;
    const angle = index * 2.399963 + (layerPhases[record.kind] || 0);
    const spread = Math.sqrt(Math.max(0, 1 - y * y));
    const hash = [...record.id].reduce((total, char) => (total * 31 + char.charCodeAt(0)) >>> 0, 17);
    const baseRadius = record.kind === 'entity' ? 122 : record.kind === 'statement' ? 174 : 228;
    const radius = baseRadius + hash % 13 - 6;
    record.position3d = [
      radius * spread * Math.cos(angle),
      radius * y,
      radius * spread * Math.sin(angle),
    ];
    return record;
  });
  const edges = cy.edges().map(item => ({...item.data()}));
  const entityLabels = new Set(nodes.filter(item => item.kind === 'entity')
    .sort((a, b) => (b.degree || 0) - (a.degree || 0)).slice(0, 4).map(item => item.id));

  function nodeColor(record) {
    if (tracedSources.has(record.id)) return COLORS.amber;
    if (record.type === 'person') return COLORS.coral;
    if (record.type === 'organization') return COLORS.lilac;
    if (record.type === 'passage') return COLORS.cyan;
    if (record.type === 'statement') return COLORS.amber;
    return COLORS.mint;
  }

  function makeLabel(record) {
    const text = shortLabel(record.caption || record.id, 38);
    const labelCanvas = document.createElement('canvas');
    const context = labelCanvas.getContext('2d');
    const font = '600 30px "DM Sans", "Segoe UI", sans-serif';
    context.font = font;
    labelCanvas.width = Math.ceil(context.measureText(text).width) + 28;
    labelCanvas.height = 52;
    context.font = font;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = '#ffffff';
    context.strokeStyle = '#c9d5d2';
    context.lineWidth = 2;
    context.beginPath();
    context.roundRect(2, 2, labelCanvas.width - 4, 48, 12);
    context.fill();
    context.stroke();
    context.fillStyle = '#263331';
    context.fillText(text, labelCanvas.width / 2, 26);
    const texture = new THREE.CanvasTexture(labelCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.SpriteMaterial({map: texture, transparent: true, depthWrite: false});
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(labelCanvas.width * .4, labelCanvas.height * .4, 1);
    return sprite;
  }

  for (const record of nodes) {
    const radius = record.kind === 'entity' ? 7.2 : record.kind === 'statement' ? 5.8 : 4.8;
    const geometry = new THREE.SphereGeometry(radius, 20, 14);
    const material = new THREE.MeshStandardMaterial({
      color: nodeColor(record), roughness: .38, metalness: .04,
      emissive: nodeColor(record), emissiveIntensity: .07, transparent: true, depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.fromArray(record.position3d);
    mesh.userData.record = record;
    graphRoot.add(mesh);
    nodeObjects.set(record.id, mesh);

    const halo = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.75, 18, 12),
      new THREE.MeshBasicMaterial({color: '#d28b28', transparent: true, opacity: .13, wireframe: true}));
    halo.visible = false;
    halo.position.copy(mesh.position);
    graphRoot.add(halo);
    mesh.userData.halo = halo;

    const label = makeLabel(record);
    label.position.copy(mesh.position).add(new THREE.Vector3(0, radius + 14, 0));
    label.renderOrder = 2;
    label.visible = entityLabels.has(record.id) || record.kind === 'tree';
    graphRoot.add(label);
    labelObjects.set(record.id, label);
  }

  for (const record of edges) {
    const source = nodeObjects.get(record.source);
    const target = nodeObjects.get(record.target);
    if (!source || !target) continue;
    const geometry = new THREE.BufferGeometry().setFromPoints([source.position, target.position]);
    const material = record.kind === 'co_mention'
      ? new THREE.LineDashedMaterial({color: '#8e9d99', transparent: true, opacity: .3, dashSize: 5, gapSize: 5})
      : new THREE.LineBasicMaterial({color: record.provenance ? '#55a99b' : '#6d8580', transparent: true, opacity: .34});
    const line = new THREE.Line(geometry, material);
    if (record.kind === 'co_mention') line.computeLineDistances();
    line.userData.record = record;
    graphRoot.add(line);
    edgeObjects.push(line);
  }

  function render() {
    if (!active || !canvas.clientWidth || canvas.hidden) return;
    graphRoot.localToWorld(target.copy(center));
    camera.position.copy(target);
    camera.position.z += distance;
    camera.lookAt(target);
    renderer.render(scene, camera);
  }

  function updateStyles() {
    const labels = hasMatches
      ? new Set(nodes.filter(record => focusNodes.has(record.id) && (record.kind === 'entity' || record.kind === 'tree'))
        .sort((a, b) => b.degree - a.degree).slice(0, 12).map(record => record.id))
      : entityLabels;
    for (const [id, mesh] of nodeObjects) {
      const focused = focusNodes.has(id);
      const matched = matchingNodes.has(id);
      const traced = tracedSources.has(id);
      const isSelected = selected === id;
      const dim = hasMatches && !focused && !traced && !isSelected;
      mesh.material.color.set(nodeColor(mesh.userData.record));
      mesh.material.emissive.set(nodeColor(mesh.userData.record));
      mesh.material.opacity = dim ? .14 : 1;
      mesh.scale.setScalar(isSelected || matched ? 1.22 : 1);
      mesh.userData.halo.visible = isSelected || traced || matched;
      mesh.userData.halo.material.color.set(traced ? COLORS.amber : matched || isSelected ? COLORS.mint : COLORS.cyan);
      mesh.userData.halo.material.opacity = dim ? .035 : .15;

      const label = labelObjects.get(id);
      const showLabel = isSelected || labels.has(id) || (matched && focusNodes.size <= 12) ||
        (!hasMatches && mesh.userData.record.kind === 'tree');
      label.visible = showLabel;
      label.material.opacity = dim ? .22 : 1;
    }
    for (const line of edgeObjects) {
      const record = line.userData.record;
      const focused = focusNodes.has(record.source) || focusNodes.has(record.target);
      const matched = matchingEdges.has(record.id);
      const selectedEdge = selected === record.id;
      line.material.opacity = hasMatches && !focused && !matched ? .045 : selectedEdge ? .94 : matched ? .86 : focused ? .56 : .34;
      if (selectedEdge || matched) line.material.color.set(COLORS.amber);
      else line.material.color.set(record.provenance ? '#55a99b' : record.kind === 'co_mention' ? '#8e9d99' : '#6d8580');
    }
    render();
  }

  let viewportWidth, viewportHeight;
  function resize() {
    const bounds = canvas.getBoundingClientRect();
    const {width, height} = bounds;
    if (!width || !height || (width === viewportWidth && height === viewportHeight)) return;
    const first = viewportWidth === undefined;
    viewportWidth = width;
    viewportHeight = height;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (framedIds) frame(framedIds);
    else if (first && !view) fit();
    else render();
  }

  function overviewDistance() {
    const angle = THREE.MathUtils.degToRad(camera.fov / 2);
    const halfFov = Math.min(angle, Math.atan(Math.tan(angle) * camera.aspect));
    const extent = Math.max(30, ...nodes.map(record => Math.hypot(...record.position3d)));
    return (extent + 28) / Math.sin(halfFov);
  }

  function fit() {
    transition = framedIds = undefined;
    center.set(0, 0, 0);
    distance = overviewDistance();
    render();
  }

  function frame(ids) {
    const points = [...ids].map(id => nodeObjects.get(id)?.position).filter(Boolean);
    if (!points.length) return;
    framedIds = ids;
    const nextCenter = new THREE.Box3().setFromPoints(points).getCenter(new THREE.Vector3());
    const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const offset = new THREE.Vector3();
    let nextDistance = 0;
    for (const point of points) {
      offset.copy(point).sub(nextCenter).applyEuler(graphRoot.rotation);
      const horizontal = (Math.abs(offset.x) + 36) / (tangent * camera.aspect);
      const vertical = (Math.abs(offset.y) + 32) / tangent;
      nextDistance = Math.max(nextDistance, offset.z + Math.max(horizontal, vertical));
    }
    nextDistance = Math.max(100, Math.min(nextDistance, overviewDistance() * .8));
    if (motion.matches) {
      transition = undefined;
      center.copy(nextCenter);
      distance = nextDistance;
      render();
    } else {
      transition = {start: performance.now(), center: center.clone(), distance, nextCenter, nextDistance};
      resume();
    }
  }

  function zoom(factor) {
    transition = undefined;
    distance = Math.max(85, Math.min(3900, distance / factor));
    render();
  }

  function animate(now) {
    if (!active || canvas.hidden || motion.matches) { raf = 0; return; }
    if (transition) {
      const progress = Math.min(1, (now - transition.start) / 480);
      const eased = 1 - (1 - progress) ** 3;
      center.lerpVectors(transition.center, transition.nextCenter, eased);
      distance = THREE.MathUtils.lerp(transition.distance, transition.nextDistance, eased);
      if (progress === 1) transition = undefined;
    } else if (!pointer && !framedIds) graphRoot.rotation.y += .0008;
    render();
    raf = transition || !framedIds ? requestAnimationFrame(animate) : 0;
  }

  function resume() {
    if (active && !canvas.hidden && !raf && !motion.matches) raf = requestAnimationFrame(animate);
  }

  function pause() {
    cancelAnimationFrame(raf);
    raf = 0;
  }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 8;
  const pointerPosition = new THREE.Vector2();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  resize();

  canvas.addEventListener('pointerdown', event => {
    pointer = {x: event.clientX, y: event.clientY, moved: false};
    canvas.setPointerCapture(event.pointerId);
  }, {signal: interaction.signal});
  canvas.addEventListener('pointermove', event => {
    if (!pointer) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) pointer.moved = true;
    graphRoot.rotation.y += dx * .006;
    graphRoot.rotation.x = Math.max(-1.35, Math.min(1.35, graphRoot.rotation.x + dy * .004));
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    render();
  }, {signal: interaction.signal});
  canvas.addEventListener('pointerup', event => {
    if (pointer && !pointer.moved) {
      const bounds = canvas.getBoundingClientRect();
      pointerPosition.set(
        ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
        -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointerPosition, camera);
      const hit = raycaster.intersectObjects([...nodeObjects.values()], false)[0]?.object ??
        raycaster.intersectObjects(edgeObjects, false)[0]?.object;
      const record = hit?.userData.record;
      if (record) onSelect(record.id);
    }
    pointer = null;
    resume();
  }, {signal: interaction.signal});
  canvas.addEventListener('pointercancel', () => { pointer = null; resume(); }, {signal: interaction.signal});
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoom(Math.exp(-event.deltaY * .001));
  }, {passive: false, signal: interaction.signal});

  return {
    fit,
    zoom,
    view() { return {center: center.clone(), distance, rotation: graphRoot.rotation.clone()}; },
    center() {
      if (focusNodes.size) frame(focusNodes);
      else if (nodeObjects.has(selected)) frame(new Set([selected]));
      else {
        const edge = edges.find(record => record.id === selected);
        if (edge) frame(new Set([edge.source, edge.target]));
        else fit();
      }
    },
    reset() {
      selected = '';
      graphRoot.rotation.set(-.12, .28, 0);
      fit();
      updateStyles();
      resume();
    },
    resume,
    pause,
    select(id) {
      selected = id;
      updateStyles();
    },
    filter(focusedIds, matchedNodeIds, matchedEdgeIds, found) {
      focusNodes = new Set(focusedIds);
      matchingNodes = new Set(matchedNodeIds);
      matchingEdges = new Set(matchedEdgeIds);
      hasMatches = found;
      updateStyles();
    },
    markSources(sourceIds) {
      tracedSources = new Set(sourceIds);
      updateStyles();
    },
    stop() {
      active = false;
      pointer = null;
      pause();
      interaction.abort();
      observer.disconnect();
      for (const object of graphRoot.children) {
        object.geometry?.dispose();
        object.material.map?.dispose();
        object.material.dispose();
      }
      renderer.dispose();
    },
  };
}

function svgIcon(name, size) {
  const icon = document.createElement('img');
  icon.src = `/static/vendor/${name}.svg`;
  icon.alt = '';
  icon.width = size;
  icon.height = size;
  return icon;
}

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
}
import {mountBlockList} from './studio_blocks.js';


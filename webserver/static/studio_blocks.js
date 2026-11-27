export function mountBlockList(host, chunks, {onPick, selected, pageSize = 12} = {}) {
  const make = (tag, text = '', cls = '') => {
    const item = document.createElement(tag);
    item.textContent = text;
    item.className = cls;
    return item;
  };
  const search = make('input');
  search.type = 'search';
  search.placeholder = '搜索编号或原文';
  search.setAttribute('aria-label', '搜索原文块');
  const list = make('div', '', 'block-results');
  const pages = make('div', '', 'block-pages');
  const previous = make('button', '上一页', 'outline-control');
  const next = make('button', '下一页', 'outline-control');
  previous.type = next.type = 'button';
  const label = make('span');
  label.setAttribute('role', 'status');
  pages.append(previous, label, next);
  host.replaceChildren(search, list, pages);
  host.classList.add('block-list');

  let page = 0;
  let matches = chunks;
  function render() {
    const start = page * pageSize;
    list.replaceChildren(...matches.slice(start, start + pageSize).map(chunk => {
      const row = make('button', '', 'block-option');
      row.type = 'button';
      row.dataset.block = chunk.id;
      row.setAttribute('aria-pressed', String(selected?.has(chunk.id) || false));
      const preview = chunk.text.trim().replace(/\s+/g, ' ').slice(0, 146);
      row.append(make('strong', chunk.id), make('span', preview));
      row.addEventListener('click', () => onPick(chunk));
      return row;
    }));
    list.scrollTop = 0;
    if (!matches.length) list.append(make('p', '没有匹配的原文块', 'panel-status'));
    label.textContent = matches.length ? `${start + 1}–${Math.min(start + pageSize, matches.length)} / ${matches.length}` : '0 项';
    previous.disabled = page === 0;
    next.disabled = start + pageSize >= matches.length;
  }
  function refreshSelection() {
    for (const row of list.querySelectorAll('[data-block]')) {
      row.setAttribute('aria-pressed', String(selected?.has(row.dataset.block) || false));
    }
  }
  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    matches = query ? chunks.filter(chunk => chunk.id.toLowerCase() === query || chunk.text.toLowerCase().includes(query)) : chunks;
    page = 0;
    render();
  });
  previous.addEventListener('click', () => { page--; render(); });
  next.addEventListener('click', () => { page++; render(); });
  render();
  return {
    refreshSelection,
    reveal(id) {
      const index = matches.findIndex(chunk => chunk.id === id);
      if (index < 0) {
        search.value = '';
        matches = chunks;
      }
      const position = index < 0 ? chunks.findIndex(chunk => chunk.id === id) : index;
      if (position < 0) return;
      const targetPage = Math.floor(position / pageSize);
      if (targetPage !== page || index < 0) {
        page = targetPage;
        render();
      } else refreshSelection();
      const row = list.children[position % pageSize];
      const rowBounds = row.getBoundingClientRect();
      const listBounds = list.getBoundingClientRect();
      if (rowBounds.top < listBounds.top) list.scrollTop -= listBounds.top - rowBounds.top;
      else if (rowBounds.bottom > listBounds.bottom) list.scrollTop += rowBounds.bottom - listBounds.bottom;
    },
  };
}

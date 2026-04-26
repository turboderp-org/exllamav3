// Branch Explorer — frontend logic
const $ = (id) => document.getElementById(id);
const blocksEl = $('blocks-section');
const branchesEl = $('branches-section');

function params() {
  return {
    top_p: parseFloat($('top_p').value),
    top_k: parseInt($('top_k').value, 10),
    max_span: parseInt($('max_span').value, 10),
  };
}

async function api(path, body) {
  document.body.classList.add('loading');
  try {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err);
    }
    return await r.json();
  } catch (e) {
    console.error(e);
    alert('Error: ' + e.message);
    return null;
  } finally {
    document.body.classList.remove('loading');
  }
}

let prevBlockCount = 0;

function render(state) {
  if (!state) return;

  blocksEl.style.display = '';
  branchesEl.style.display = '';

  blocksEl.innerHTML = '';
  const lastIdx = state.blocks.length - 1;
  const grew = state.blocks.length > prevBlockCount;
  state.blocks.forEach((b, i) => {
    const span = document.createElement('span');
    let cls = 'block ' + (b.is_prompt ? 'prompt' : 'model');
    if (i === lastIdx) cls += ' latest';
    if (grew && i === lastIdx) cls += ' flash';
    span.className = cls;
    span.textContent = b.text;
    span.title = i === lastIdx ? 'Current position' : 'Click to rewind here';
    span.addEventListener('click', () => rewind(i));
    blocksEl.appendChild(span);
  });
  prevBlockCount = state.blocks.length;

  branchesEl.innerHTML = '';
  const label = document.createElement('div');
  label.className = 'section-label';
  label.textContent = state.branches.length ? 'Continue with…' : 'No continuations';
  branchesEl.appendChild(label);

  if (state.branches.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Reached end of sequence (EOS or max span).';
    branchesEl.appendChild(empty);
    return;
  }

  state.branches.forEach((br, i) => {
    const card = document.createElement('div');
    card.className = 'branch';
    card.style.setProperty('--i', i);
    const pctNum = (br.prob * 100).toFixed(1);

    const idx = document.createElement('div');
    idx.className = 'idx';
    idx.textContent = (i + 1);

    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = br.text || '∅';

    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = pctNum + '%';
    bar.appendChild(fill);

    const prob = document.createElement('div');
    prob.className = 'prob';
    prob.textContent = pctNum + '%';

    card.append(idx, text, bar, prob);
    card.addEventListener('click', () => commit(i));
    branchesEl.appendChild(card);
  });
}

async function start() {
  const prompt = $('prompt').value;
  if (!prompt) return;
  const s = await api('/api/init', { prompt, ...params() });
  render(s);
}

async function commit(i) {
  const s = await api('/api/commit', { branch_index: i, ...params() });
  render(s);
}

async function rewind(i) {
  const s = await api('/api/rewind', { block_index: i, ...params() });
  render(s);
}

$('start-btn').addEventListener('click', start);
$('reset-btn').addEventListener('click', () => {
  $('prompt').value = '';
  blocksEl.style.display = 'none';
  branchesEl.style.display = 'none';
  prevBlockCount = 0;
});

$('default-btn').addEventListener('click', () => {
  (async () => {
    try {
      const r = await fetch('/api/default_prompt');
      if (!r.ok) return;
      const { prompt } = await r.json();
      $('prompt').value = prompt;
      blocksEl.style.display = 'none';
      branchesEl.style.display = 'none';
      prevBlockCount = 0;
    } catch (e) { /* ignore */ }
  })();

});

// Cmd/Ctrl+Enter in the textarea starts a session.
$('prompt').addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    start();
  }
});

// Number keys 1-9 to pick a branch (when not typing in an input).
document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'TEXTAREA' || tag === 'INPUT') return;
  const n = parseInt(e.key, 10);
  if (!Number.isNaN(n) && n >= 1 && n <= 9) {
    const cards = branchesEl.querySelectorAll('.branch');
    if (cards[n - 1]) cards[n - 1].click();
  }
});

// Load current prompt from the server on page load.
(async () => {
  try {
    const r = await fetch('/api/current_prompt');
    if (!r.ok) return;
    const { prompt } = await r.json();
    if (prompt && !$('prompt').value) $('prompt').value = prompt;
  } catch (e) { /* ignore */ }
})();
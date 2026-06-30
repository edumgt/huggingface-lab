const SEND_ICON = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;
const SPIN_ICON = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2a10 10 0 0 1 10 10"/></svg>`;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `요청 실패: ${url}`);
  }
  return response.json();
}

let currentTab = 'ai';

function toggleModeUI(outputType) {
  const isImage = outputType === 'image';
  document.querySelectorAll('.image-only').forEach((el) => el.classList.toggle('hidden', !isImage));
  document.querySelectorAll('.video-only').forEach((el) => el.classList.toggle('hidden', isImage));
}

function setTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tab));
  document.querySelectorAll('.ai-only').forEach((el) => el.classList.toggle('hidden', tab !== 'ai'));
  document.querySelectorAll('.stock-only').forEach((el) => el.classList.toggle('hidden', tab !== 'stock'));
  if (tab === 'ai') toggleModeUI(document.getElementById('output-type').value);
  resetToWelcome();
}

async function initializeForm() {
  const data = await fetchJson('/api/options');
  const modelSelect = document.getElementById('model-id');
  modelSelect.innerHTML = '';
  data.models.forEach((id) => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    modelSelect.appendChild(opt);
  });
}

function readAiPayload() {
  return {
    model_id: document.getElementById('model-id').value,
    output_type: document.getElementById('output-type').value,
    width: Number(document.getElementById('width').value),
    height: Number(document.getElementById('height').value),
    video_size: document.getElementById('video-size').value,
    style_preset: document.getElementById('style-preset').value,
    prompt: document.getElementById('prompt').value.trim(),
  };
}

function readStockPayload() {
  const indicators = Array.from(document.querySelectorAll('.indicator-item input:checked')).map((el) => el.value);
  return {
    ticker: document.getElementById('ticker-input').value.trim(),
    period: document.getElementById('period').value,
    interval: document.getElementById('interval').value,
    chart_type: document.getElementById('chart-type').value,
    indicators,
  };
}

function showResult(bubbleText) {
  document.getElementById('welcome-screen').classList.add('hidden');
  const screen = document.getElementById('result-screen');
  screen.classList.remove('hidden');
  document.getElementById('result-prompt').textContent = bubbleText;
  document.getElementById('summary-grid').classList.add('hidden');
  document.getElementById('summary-grid').innerHTML = '';
  screen.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showSummary(summary) {
  const labels = {
    last_close: '종가',
    change_pct: '변동률',
    high: '최고가',
    low: '최저가',
    avg_volume: '평균 거래량',
  };
  const grid = document.getElementById('summary-grid');
  grid.innerHTML = Object.entries(summary)
    .map(([key, value]) => {
      const label = labels[key] || key;
      const display = key === 'change_pct' ? `${value > 0 ? '+' : ''}${value}%` : Number(value).toLocaleString();
      return `<div class="summary-item"><span class="summary-label">${label}</span><span class="summary-value">${display}</span></div>`;
    })
    .join('');
  grid.classList.remove('hidden');
}

function resetToWelcome() {
  document.getElementById('welcome-screen').classList.remove('hidden');
  document.getElementById('result-screen').classList.add('hidden');
  document.getElementById('result-image').classList.add('hidden');
  document.getElementById('result-image').src = '';
  document.getElementById('result-prompt').textContent = '';
  document.getElementById('status').textContent = '';
  document.getElementById('summary-grid').classList.add('hidden');
  document.getElementById('summary-grid').innerHTML = '';
  const promptEl = document.getElementById('prompt');
  promptEl.value = '';
  promptEl.style.height = 'auto';
  document.getElementById('ticker-input').value = '';
  updateSendBtn();
}

function updateSendBtn() {
  const btn = document.getElementById('submit-button');
  const value = currentTab === 'ai'
    ? document.getElementById('prompt').value.trim()
    : document.getElementById('ticker-input').value.trim();
  btn.disabled = !value;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

async function generateAi() {
  const statusEl = document.getElementById('status');
  const resultImg = document.getElementById('result-image');
  const payload = readAiPayload();
  if (!payload.prompt) return;

  showResult(payload.prompt);
  statusEl.textContent = '생성 중...';

  const data = await fetchJson('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  resultImg.src = data.data_url || `${data.file_url}?t=${Date.now()}`;
  resultImg.classList.remove('hidden');
  statusEl.textContent = `${data.model_id} · ${data.width}×${data.height}`;
}

async function generateStock() {
  const statusEl = document.getElementById('status');
  const resultImg = document.getElementById('result-image');
  const payload = readStockPayload();
  if (!payload.ticker) return;

  showResult(payload.ticker.toUpperCase());
  statusEl.textContent = '시세 조회 및 차트 생성 중...';

  const data = await fetchJson('/api/stock-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  resultImg.src = data.data_url || `${data.file_url}?t=${Date.now()}`;
  resultImg.classList.remove('hidden');
  statusEl.textContent = `${data.ticker} · ${data.period}/${data.interval} · ${data.chart_type}`;
  showSummary(data.summary);
}

async function generate(event) {
  event.preventDefault();

  const statusEl = document.getElementById('status');
  const btn = document.getElementById('submit-button');

  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = SPIN_ICON;
  document.getElementById('result-image').classList.add('hidden');

  try {
    if (currentTab === 'ai') {
      await generateAi();
    } else {
      await generateStock();
    }
  } catch (err) {
    statusEl.textContent = `오류: ${err.message}`;
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = SEND_ICON;
    updateSendBtn();
  }
}

async function boot() {
  const sidebar = document.getElementById('sidebar');
  document.getElementById('sidebar-toggle').addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  document.getElementById('new-gen-btn').addEventListener('click', resetToWelcome);

  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });

  const outputTypeEl = document.getElementById('output-type');
  outputTypeEl.addEventListener('change', (e) => toggleModeUI(e.target.value));
  toggleModeUI(outputTypeEl.value);

  const promptEl = document.getElementById('prompt');
  promptEl.addEventListener('input', () => {
    autoResize(promptEl);
    updateSendBtn();
  });

  promptEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!document.getElementById('submit-button').disabled) {
        document.getElementById('generate-form').requestSubmit();
      }
    }
  });

  const tickerEl = document.getElementById('ticker-input');
  tickerEl.addEventListener('input', updateSendBtn);
  tickerEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!document.getElementById('submit-button').disabled) {
        document.getElementById('generate-form').requestSubmit();
      }
    }
  });

  try {
    await initializeForm();
  } catch {
    document.getElementById('status').textContent = '초기 데이터 로드 실패';
  }

  document.getElementById('generate-form').addEventListener('submit', generate);
}

boot();

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

function toggleModeUI(outputType) {
  const isImage = outputType === 'image';
  document.querySelectorAll('.image-only').forEach((el) => el.classList.toggle('hidden', !isImage));
  document.querySelectorAll('.video-only').forEach((el) => el.classList.toggle('hidden', isImage));
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

function readPayload() {
  const outputType = document.getElementById('output-type').value;
  return {
    model_id: document.getElementById('model-id').value,
    output_type: outputType,
    width: Number(document.getElementById('width').value),
    height: Number(document.getElementById('height').value),
    video_size: document.getElementById('video-size').value,
    prompt: document.getElementById('prompt').value.trim(),
  };
}

function showResult(promptText) {
  document.getElementById('welcome-screen').classList.add('hidden');
  const screen = document.getElementById('result-screen');
  screen.classList.remove('hidden');
  document.getElementById('result-prompt').textContent = promptText;
  screen.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetToWelcome() {
  document.getElementById('welcome-screen').classList.remove('hidden');
  document.getElementById('result-screen').classList.add('hidden');
  document.getElementById('result-image').classList.add('hidden');
  document.getElementById('result-image').src = '';
  document.getElementById('result-prompt').textContent = '';
  document.getElementById('status').textContent = '';
  const promptEl = document.getElementById('prompt');
  promptEl.value = '';
  promptEl.style.height = 'auto';
  updateSendBtn();
}

function updateSendBtn() {
  const btn = document.getElementById('submit-button');
  btn.disabled = !document.getElementById('prompt').value.trim();
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

async function generate(event) {
  event.preventDefault();

  const statusEl = document.getElementById('status');
  const resultImg = document.getElementById('result-image');
  const btn = document.getElementById('submit-button');
  const payload = readPayload();

  if (!payload.prompt) return;

  showResult(payload.prompt);

  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = SPIN_ICON;
  resultImg.classList.add('hidden');
  statusEl.textContent = '생성 중...';

  try {
    const data = await fetchJson('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    resultImg.src = data.data_url || `${data.file_url}?t=${Date.now()}`;
    resultImg.classList.remove('hidden');
    statusEl.textContent = `${data.model_id} · ${data.width}×${data.height}`;
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

  try {
    await initializeForm();
  } catch {
    document.getElementById('status').textContent = '초기 데이터 로드 실패';
  }

  document.getElementById('generate-form').addEventListener('submit', generate);
}

boot();

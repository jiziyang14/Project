// 后端 API 封装：所有请求指向本地 127.0.0.1:8000，无任何外部网络请求
export async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  // 文件上传走 FormData，其余走 JSON
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== null) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    throw new Error(`请求失败：${path} (${resp.status})`);
  }
  return resp.json();
}

export const Questions = {
  list: () => api('/api/questions'),
  add: (q) => api('/api/questions', 'POST', q),
  ocr: (file) => {
    const fd = new FormData();
    fd.append('image', file);
    return api('/api/questions/ocr', 'POST', fd);
  },
  answer: (payload) => api('/api/answer', 'POST', payload),
  // 按需生成某题的"该划关键词"（题目未存 keywords 时即时算，不回写）
  suggestKeywords: (q) => api('/api/questions/suggest-keywords', 'POST', q),
};

export const Analysis = {
  clusters: () => api('/api/clusters'),
  errorClusters: () => api('/api/error-clusters'),
  confusion: () => api('/api/confusion'),
  friction: (questionId) => api(`/api/questions/${questionId}/friction`),
  variant: (questionId) => api('/api/variant', 'POST', { question_id: questionId }),
  article: (text, title) => api('/api/articles', 'POST', { text, title: title || '', top_n: 8 }),
  // 流式：边处理边回调每一阶段事件（SSE），供实时过程可视化
  articleStream: (text, title, onStage, onDone, onError) => {
    return fetch('/api/articles/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, title: title || '', top_n: 8 }),
    }).then(async (resp) => {
      if (!resp.ok || !resp.body) throw new Error(`请求失败：/api/articles/stream (${resp.status})`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const chunk = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 2);
          if (!chunk) continue;
          const line = chunk.split('\n').find((l) => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
          if (ev.type === 'stage' && onStage) onStage(ev);
          else if (ev.type === 'done' && onDone) onDone(ev.result);
          else if (ev.type === 'error' && onError) onError(ev.error);
        }
      }
    });
  },
  graph: () => api('/api/graph'),
};

export const Modules = {
  status: () => api('/api/modules'),
  toggle: (name) => api('/api/modules/toggle', 'POST', { name }),
  modelReady: async (code) => {
    const res = await api(`/api/models/ready/${code}`);
    return !!res.ready;
  },
};

export const Monitor = {
  status: () => api('/api/monitor'),
  insights: () => api('/api/insights'),
};

export const Boss = {
  status: () => api('/api/boss'),
};

export const Voice = {
  analyze: (wavBlob) => {
    const fd = new FormData();
    fd.append('audio', wavBlob, 'voice.wav');
    return api('/api/voice', 'POST', fd);
  },
};

export const Reading = {
  capture: (words, source = 'question') => api('/api/reading', 'POST', { words, source }),
  history: () => api('/api/reading/history'),
};

export const Screen = {
  capture: () => api('/api/screen/capture', 'POST'),
  ocr: (box) => api('/api/screen/ocr', 'POST', box),
};

export const Code = {
  analyze: (code, tle) => api('/api/code/analyze', 'POST', { code, tle }),
};

export const Train = {
  status: () => api('/api/train/status'),
  addArticle: (title, text) => api('/api/train/article', 'POST', { title, text }),
  importFolder: () => api('/api/train/import-folder', 'POST'),
  importFiles: (files) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f));
    return api('/api/train/import-files', 'POST', fd);
  },
  export: () => api('/api/train/export', 'POST'),
  items: () => api('/api/train/items'),
  saveLabel: (docId, candidates) => api('/api/train/label', 'POST', { doc_id: docId, candidates }),
  train: () => api('/api/train/rerank-run', 'POST'),
  eval: () => api('/api/train/eval', 'POST'),
  // 读题划词标注（题源 data/huaci/{学科}/pool.json）
  huaci: {
    status: () => api('/api/train/huaci-status'),
    items: (subj) => api(`/api/train/huaci-items?subj=${encodeURIComponent(subj)}`),
    saveLabel: (subj, qid, candidates) => api('/api/train/huaci-save', 'POST', { subj, qid, candidates }),
    export: (subj) => api('/api/train/huaci-export', 'POST', { subj }),
    eval: (subj) => api('/api/train/huaci-eval', 'POST', { subj }),
  },
};

// 审题练习（纯审题题库）
export const Stk = {
  status: () => api('/api/readingbank/status'),
  random: (subj, difficulty, type) => api(`/api/readingbank/random?subject=${encodeURIComponent(subj)}&difficulty=${encodeURIComponent(difficulty || '')}&ques_type=${encodeURIComponent(type || '')}`),
  list: (subj, offset, limit) => api(`/api/readingbank/list?subject=${encodeURIComponent(subj)}&offset=${offset || 0}&limit=${limit || 30}`),
  saveKeywords: (subj, id, keywords) => api('/api/readingbank/save-keywords', 'POST', { subject: subj, id, keywords }),
};
// 自定义弹窗组件：禁用原生 alert/confirm，保证视觉统一（Design.md 强制规则）
let okHandler = null;

export function showModal(title, body, onOk = null) {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalBody').textContent = body;
  document.getElementById('modalMask').classList.remove('hidden');
  okHandler = onOk;
}

export function initModal() {
  document.getElementById('modalMask').addEventListener('click', (e) => {
    // 点击遮罩或"知道了"均关闭
    if (e.target.id === 'modalMask' || e.target.id === 'modalOk') {
      closeModal();
    }
  });
}

export function closeModal() {
  document.getElementById('modalMask').classList.add('hidden');
  if (okHandler) {
    const fn = okHandler;
    okHandler = null;
    fn();
  }
}
// Right-side detail drawer (one at a time). Esc or backdrop click closes it.
import { icon } from './icons.js';
import { disposeWithin } from './charts.js';

let current = null;

export function openDrawer({ title, body, footer = '', onClose, onMount }) {
  closeDrawer();
  const back = document.createElement('div'); back.className = 'drawer-backdrop';
  const d = document.createElement('aside'); d.className = 'drawer'; d.setAttribute('role', 'dialog'); d.setAttribute('aria-modal', 'true');
  d.innerHTML = `<div class="drawer-h"><div class="row" style="min-width:0">${title}</div>
    <button class="btn ghost icon" data-close aria-label="Close">${icon('close')}</button></div>
    <div class="drawer-b">${body}</div>${footer ? `<div class="drawer-f">${footer}</div>` : ''}`;
  document.body.append(back, d);
  requestAnimationFrame(() => { back.classList.add('open'); d.classList.add('open'); });
  const esc = (e) => { if (e.key === 'Escape') closeDrawer(); };
  back.addEventListener('click', closeDrawer);
  d.querySelector('[data-close]').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', esc);
  current = { back, d, esc, onClose };
  d.querySelector('[data-close]').focus();
  onMount?.(d);
  return d;
}

export function closeDrawer() {
  if (!current) return;
  const { back, d, esc, onClose } = current; current = null;
  document.removeEventListener('keydown', esc);
  disposeWithin(d);
  back.remove(); d.remove();
  onClose?.();
}
export const drawerOpen = () => !!current;

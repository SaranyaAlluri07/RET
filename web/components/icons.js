// Inline SVG icon set (stroke icons, 24x24 grid). Usage: icon('play').
const P = {
  logo: '<path d="M12 12v10"/><path d="M12 12 4.5 7.5"/><path d="M12 12l7.5-4.5"/><path d="M12 12V2.5"/><circle cx="12" cy="12" r="1.6"/><path d="M8 22h8"/>',
  fleet: '<path d="M3 21h18"/><path d="M6 21V11"/><path d="M6 11 3 9"/><path d="M6 11l3-2"/><path d="M6 11V7"/><path d="M16 21V9"/><path d="M16 9l-3.5-2"/><path d="M16 9l3.5-2"/><path d="M16 9V4.5"/>',
  turbine: '<circle cx="12" cy="8" r="1.5"/><path d="M12 9.5V22"/><path d="M12 6.5V2"/><path d="m10.7 8.8-4.2 2.6"/><path d="m13.3 8.8 4.2 2.6"/><path d="M9 22h6"/>',
  cases: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18"/><path d="M8 14h5"/><path d="M8 17h8"/>',
  replay: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v5l3 2"/>',
  sim: '<path d="M9 3h6"/><path d="M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3"/><path d="M7 15h10"/>',
  eval: '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
  forecast: '<path d="M3 17c3-1 4-9 9-9s6 5 9 4"/><path d="M3 21h18"/><path d="M3 3v18"/>',
  about: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5v.5"/>',
  play: '<path d="M7 4.5v15l12-7.5z"/>',
  pause: '<path d="M8 4v16"/><path d="M16 4v16"/>',
  step: '<path d="M6 4.5v15l10-7.5z"/><path d="M19 4v16"/>',
  back: '<path d="M18 4.5v15L8 12z"/><path d="M5 4v16"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  reset: '<path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5"/>',
  wrench: '<path d="M14.7 6.3a4 4 0 0 0 5 5L21 13l-8 8-2-2 6.3-6.3a4 4 0 0 1-5-5L9 5.4"/><path d="M3 21l6-6"/>',
  arrowRight: '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5v.5"/>',
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4"/><path d="M12 17v.5"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  question: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6"/><path d="M12 17v.5"/>',
  db: '<ellipse cx="12" cy="5.5" rx="8" ry="2.5"/><path d="M4 5.5v13c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-13"/><path d="M4 12c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5"/>',
  // state icons (paired with every status colour so state never relies on colour alone)
  st_normal: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  st_watch: '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="2.5"/>',
  st_warning: '<path d="M12 3 2 20h20z"/><path d="M12 10v4"/><path d="M12 17v.5"/>',
  st_fault: '<path d="M8 3h8l5 5v8l-5 5H8l-5-5V8z"/><path d="M12 8v5"/><path d="M12 16v.5"/>',
  st_unknown: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6"/><path d="M12 17v.5"/>',
  st_offline: '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>',
  st_tripped: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  st_nodata: '<circle cx="12" cy="12" r="9" stroke-dasharray="3 3"/>',
};

export function icon(name, cls = '') {
  return `<svg class="ic ${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[name] || P.info}</svg>`;
}

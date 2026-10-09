
"use strict";
/* ---------- утилиты ---------- */
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";
function h(tag, a = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(a)) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v); }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
function sv(tag, a = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) if (v != null && v !== false) e.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
const fmt = (x, d = 1) => (x == null || isNaN(x) ? "—" : x.toFixed(d).replace(".", ","));
const pct = (x, d = 1) => fmt(100 * x, d) + "%";
const sgn = (x, d = 2) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const mix = (c1, c2, u) => { const p = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); const a = p(c1), b = p(c2); return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * u).toString(16).padStart(2, "0")).join(""); };
function median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

const T = D.T, N = D.N, GEO = D.geo;
const C = THEME.C, TYPE_COL = THEME.typeCol;
const S = { k: String(D.main), t: T - 1, tab: "answer", sel: null, mode: "type", iso: null, hatch: false, demo: null };
const KD = () => D.K[S.k];
const tcol = (k, i) => TYPE_COL[D.K[k].names[i]];
const typeAt = (i, t = S.t) => KD().L[t][i];
const monthLabel = t => { const [y, m] = D.months[t].split("-"); return ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"][+m - 1] + " " + y; };
const moName = i => D.mo.name[i].replace(/^(городской округ|муниципальный район|муниципальный округ)\s+/i, m => m.toLowerCase().startsWith("городской") ? "г.о. " : "").replace(/^г\.о\. город\s+/i, "г.о. ");
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = ""; tip.append(...html); tip.style.display = "block"; const w = tip.offsetWidth, hh = tip.offsetHeight; tip.style.left = clamp(ev.clientX + 14, 8, innerWidth - w - 8) + "px"; tip.style.top = clamp(ev.clientY + 14, 8, innerHeight - hh - 8) + "px"; }
const hideTip = () => { tip.style.display = "none"; };
const tipHtml = (title, sub) => [h("b", { text: title }), h("span", { class: "muted", text: sub })];

/* ---------- карта (один экземпляр, переезжает между вкладками) ---------- */
const MAP = { vb: { x: 0, y: 0, w: GEO.W, h: GEO.H }, paths: [], hatch: [] };
function regionColor(name) { let x = 2166136261; for (const c of name) { x ^= c.charCodeAt(0); x = Math.imul(x, 16777619); } const hue = ((x >>> 0) % 14) * 25.7, l = 30 + (((x >>> 5) % 4) * 6); return `hsl(${hue.toFixed(0)} 22% ${l}%)`; }
function initMap() {
  const box = h("div", { class: "mapbox" }), svg = sv("svg", { viewBox: `0 0 ${GEO.W} ${GEO.H}`, role: "img", "aria-label": "Карта типов муниципальных образований" });
  svg.append(sv("defs", {}, sv("pattern", { id: "hatchp", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, sv("rect", { width: 6, height: 6, fill: C.hatchBg }), sv("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: C.sel, "stroke-width": 1.6, opacity: .85 }))));
  const g = sv("g"); MAP.g = g; MAP.gh = sv("g"); MAP.ga = sv("g", { "pointer-events": "none" });
  for (let i = 0; i < N; i++) { const p = sv("path", { d: GEO.paths[i], class: "mo", "data-i": i, "fill-rule": "evenodd" }); MAP.paths.push(p); g.append(p); }
  svg.append(g, MAP.gh, MAP.ga); MAP.svg = svg; box.append(svg);
  const zoom = h("div", { class: "zoom" }, h("button", { "aria-label": "Приблизить", text: "+", onclick: () => zoomBy(1.6) }), h("button", { "aria-label": "Отдалить", text: "−", onclick: () => zoomBy(1 / 1.6) }), h("button", { "aria-label": "Показать всю страну", text: "⌂", onclick: resetView }));
  box.append(zoom, h("div", { class: "maphint", text: `Масштаб: колесо мыши с Ctrl, перетаскивание или кнопки. ${D.counts.incomplete} МО с неполным рядом на карте не закрашены.` }));
  MAP.box = box;
  let drag = null;
  svg.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, vb: { ...MAP.vb }, moved: false }; });
  addEventListener("pointermove", e => { if (!drag) return; const r = svg.getBoundingClientRect(), k = MAP.vb.w / r.width, dx = (e.clientX - drag.x) * k, dy = (e.clientY - drag.y) * k; if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) drag.moved = true; if (drag.moved) { MAP.vb.x = drag.vb.x - dx; MAP.vb.y = drag.vb.y - dy; applyVb(); } });
  addEventListener("pointerup", () => { setTimeout(() => (drag = null), 0); });
  svg.addEventListener("click", e => { if (drag && drag.moved) return; const i = e.target.dataset && e.target.dataset.i; if (i !== undefined) pick(+i); });
  svg.addEventListener("mousemove", e => { const i = e.target.dataset && e.target.dataset.i; if (i === undefined || (drag && drag.moved)) { hideTip(); return; } const k = KD(), ty = typeAt(+i); showTip(e, [h("b", { text: D.mo.name[+i] }), h("span", { class: "muted", text: D.mo.region[+i] }), h("div", {}, h("span", { class: "dot", style: `background:${tcol(S.k, ty)};margin-right:6px` }), k.names[ty]), h("span", { class: "muted", text: `согласие при пересборке ${pct(k.conf[+i], 0)}${k.boundary[+i] ? ", пограничное" : ""}` })]); });
  svg.addEventListener("mouseleave", hideTip);
  svg.addEventListener("wheel", e => { if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); const r = svg.getBoundingClientRect(); zoomBy(e.deltaY < 0 ? 1.25 : 0.8, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height); }, { passive: false });
}
function applyVb() { const v = MAP.vb; v.w = clamp(v.w, 40, GEO.W); v.h = v.w * GEO.H / GEO.W; v.x = clamp(v.x, -20, GEO.W - v.w + 20); v.y = clamp(v.y, -20, GEO.H - v.h + 20); MAP.svg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`); }
function zoomBy(f, fx = .5, fy = .5) { const v = MAP.vb, nw = clamp(v.w / f, 40, GEO.W); v.x += (v.w - nw) * fx; v.y += (v.h - nw * GEO.H / GEO.W) * fy; v.w = nw; applyVb(); }
function resetView() { MAP.vb = { x: 0, y: 0, w: GEO.W, h: GEO.H }; applyVb(); }
function mountMap(slotId) { const slot = $("#" + slotId); if (MAP.box.parentNode !== slot) slot.append(MAP.box); paintMap(); }
function colorFor(i) {
  const k = KD(), ty = typeAt(i);
  if (S.iso != null && ty !== S.iso && S.mode === "type") return C.dim1;
  if (S.mode === "region") return regionColor(D.mo.region[i]);
  if (S.mode === "border") return mix(C.bLo, C.accHi, clamp((1 - k.conf[i]) * 1.8, 0, 1));
  if (S.mode === "change") return k.changed[i] ? tcol(S.k, ty) : C.dim2;
  return tcol(S.k, ty);
}
function paintMap() {
  for (let i = 0; i < N; i++) { MAP.paths[i].setAttribute("fill", colorFor(i)); MAP.paths[i].classList.toggle("sel", i === S.sel); }
  MAP.gh.replaceChildren(); if (S.hatch) { const k = KD(); for (let i = 0; i < N; i++) if (k.boundary[i]) MAP.gh.append(sv("path", { d: GEO.paths[i], fill: "url(#hatchp)", class: "hatch", "fill-rule": "evenodd" })); }
  drawArcs();
}
function drawArcs() {
  MAP.ga.replaceChildren(); const i = S.sel; if (i == null) return;
  for (const j of D.nb[i]) { const x1 = GEO.px[i], y1 = GEO.py[i], x2 = GEO.px[j], y2 = GEO.py[j], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), cx = (x1 + x2) / 2 - dy * .22, cy = (y1 + y2) / 2 + dx * .22 - L * .05;
    MAP.ga.append(sv("path", { d: `M${x1} ${y1}Q${cx} ${cy} ${x2} ${y2}`, fill: "none", stroke: C.arc, "stroke-width": 1.3, opacity: .85 }), sv("circle", { cx: x2, cy: y2, r: 3.4, fill: tcol(S.k, typeAt(j)), stroke: C.bg, "stroke-width": 1.2 })); }
  MAP.ga.append(sv("circle", { cx: GEO.px[i], cy: GEO.py[i], r: 6, fill: C.sel, stroke: C.bg, "stroke-width": 1.5 }));
}

/* ---------- выбор МО ---------- */
const lab = n => { n = n.replace(/\(подобран:[^)]*\)/, "(подобран)"); return n.length > 38 ? n.slice(0, 37) + "…" : n; };
const STORY = !!document.querySelector("main.story"); /* лонгрид: все разделы на одной странице, одна карта */
function pick(i, goto = true) { S.sel = i; paintMap(); renderPassport(); renderAnsNote(); if (S.tab === "method" || STORY) renderMethod(); }
function search(v) { const i = D.mo.name.findIndex((n, s) => n + " · " + D.mo.region[s] === v); if (i >= 0) { pick(i); $("#q").value = ""; if (S.tab === "answer") return; } }

/* ---------- вкладка «Ответ» ---------- */
function renderAnswer() {
  const k = KD(), leg = $("#ans-legend"); leg.replaceChildren();
  k.names.forEach((nm, i) => leg.append(h("div", Object.assign({ class: "it" + (STORY && S.iso != null && S.iso !== i ? " dim" : "") }, STORY ? { role: "button", tabindex: "0", "aria-pressed": S.iso === i, title: "Выделить тип на карте", onclick: () => { S.iso = S.iso === i ? null : i; renderAnswer(); paintMap(); }, onkeydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.target.click(); } } } : {}), h("span", { class: "sw", style: `background:${tcol(S.k, i)}` }), h("b", { text: nm }), h("span", { class: "num muted", text: `${k.n[i]} МО` }), h("small", { text: k.note[i] }))));
  const sr = D.same_region_share, never = 1 - k.changed.reduce((a, b) => a + b, 0) / N, ar = D.methods.find(m => m.name === "GS-TKM");
  const facts = [[pct(never), "МО ни разу не меняли тип за 13 окон"], [pct(sr, 0), "из 10 самых похожих МО лежат в том же регионе: типы пересекают границы"], [fmt(ar.ari, 3), "согласие между запусками с разными seed: тип не зависит от случайности"], [fmt(k.ari_boot, 3), "согласие при пересборке на 80% МО (ARI)"]];
  $("#facts").replaceChildren(...facts.map(([b, s]) => h("div", { class: "fact" }, h("b", { text: b }), h("span", { text: s }))));
  renderAnsNote();
}
function renderAnsNote() { const n = $("#ans-note"); n.replaceChildren(); if (S.sel == null) { n.textContent = "нажмите на МО на карте"; return; } const i = S.sel; n.append(h("b", { text: D.mo.name[i] }), " · " + KD().names[typeAt(i)] + " · ", h("a", { href: "#", text: "паспорт", onclick: e => { e.preventDefault(); if (STORY) $("#passport").scrollIntoView({ behavior: "smooth", block: "center" }); else gotoTab("map"); } })); }

/* ---------- вкладка «Карта и МО» ---------- */
function renderMapLegend() {
  const k = KD(), box = $("#map-legend"); box.replaceChildren(); box.hidden = STORY && S.mode === "type"; if (box.hidden) return;
  if (S.mode !== "type") { const txt = S.mode === "region" ? "цвет — регион: один оттенок на регион; видно, что типы пересекают региональные границы" : S.mode === "border" ? "чем ярче МО, тем ближе оно к границе между типами (по 50 пересборкам на 80% МО)" : "цветом показаны только МО, менявшие тип за период; остальные приглушены"; box.append(h("span", { class: "muted", text: txt })); return; }
  k.names.forEach((nm, i) => { const c = KD().L[S.t].filter(v => v === i).length; box.append(h("button", { class: "chip" + (S.iso != null && S.iso !== i ? " dim" : ""), "aria-pressed": S.iso === i, onclick: () => { S.iso = S.iso === i ? null : i; renderMapLegend(); paintMap(); } }, h("span", { class: "sw", style: `background:${tcol(S.k, i)}` }), nm, h("span", { class: "num muted", text: String(c) }))); });
}
function renderPassport() {
  const box = $("#passport"); box.replaceChildren(); const i = S.sel;
  if (i == null) { box.append(h("h3", { text: "МО не выбрано" }), h("p", { class: "muted", text: "Нажмите на МО на карте или найдите по названию " + (STORY ? "над картой" : "вверху справа") + ". Появятся его тип по месяцам, профиль относительно типа и ближайшие по сходству МО (линии на карте)." })); return; }
  const k = KD(), ty = typeAt(i), end = k.L[T - 1][i], ch = new Set(k.L.map(r => r[i])).size > 1, same = D.nb[i].filter(j => D.mo.region[j] === D.mo.region[i]).length;
  box.append(h("h3", { text: D.mo.name[i] }), h("p", { class: "muted", text: `${D.mo.region[i]}${D.mo.pop[i] ? " · " + D.mo.pop[i].toLocaleString("ru-RU") + " жителей" : ""}` }),
    h("div", { style: "margin-top:8px" }, h("span", { class: "dot", style: `background:${tcol(S.k, ty)};margin-right:6px` }), h("b", { text: k.names[ty] }), ` в ${monthLabel(S.t)}`),
    h("p", { class: "muted", text: ch ? "За период МО меняло тип." : "За период МО тип не меняло." }));
  box.append(h("div", { class: "gap-s", style: "margin-top:6px" }, h("span", { class: "tag" + (k.boundary[i] ? " warn" : ""), text: `согласие при пересборке ${pct(k.conf[i], 0)}` }), k.boundary[i] ? h("span", { class: "tag warn", text: "пограничное: ближайший чужой тип — " + k.names[k.second[i]] }) : null));
  const rb = h("div", { class: "ribbon", role: "img", "aria-label": "Тип МО по месяцам" }); for (let t = 0; t < T; t++) rb.append(h("i", { style: `background:${tcol(S.k, k.L[t][i])};${t === S.t ? "outline:2px solid ${C.sel};outline-offset:1px" : ""}`, title: `${monthLabel(t)}: ${k.names[k.L[t][i]]}` }));
  box.append(rb, h("div", { class: "ticks" }, h("span", { text: monthLabel(0) }), h("span", { text: monthLabel(T - 1) })));
  box.append(h("h4", { style: "margin:14px 0 4px", text: "Профиль: отклонение от медианы всех МО, σ" }), h("div", { class: "muted", style: "font-size:12px;margin-bottom:4px", text: "полоса — это МО, белая метка — медиана его типа" }));
  const bars = h("div", { class: "bars" });
  D.feat.names.forEach((nm, j) => { const v = D.feat.z[i][j], m = k.tmed[end][j], w = Math.abs(v) / 3 * 50, l = v >= 0 ? 50 : 50 - w; bars.append(h("div", { class: "row" }, h("span", { text: nm }), h("div", { class: "bar" }, h("s"), h("b", { style: `left:${l}%;width:${w}%;background:${tcol(S.k, end)}` }), h("u", { style: `left:${50 + m / 3 * 50}%` })))); });
  box.append(bars, h("h4", { style: "margin:14px 0 4px", text: "Ближайшие по сходству профиля" }), h("div", { class: "muted", style: "font-size:12px", text: `из своего региона: ${same} из 10. Линии на карте ведут к этим МО.` }));
  const nbs = h("div", { class: "nbs" }); D.nb[i].forEach((j, s) => nbs.append(h("button", { onclick: () => pick(j) }, h("span", { class: "dot", style: `background:${tcol(S.k, typeAt(j))}` }), h("span", {}, D.mo.name[j], h("span", { class: "muted", text: " · " + D.mo.region[j] })), h("span", { class: "num muted", text: fmt(D.sim[i][s], 2) })))); box.append(nbs);
}

/* ---------- алгоритм Витерби (как в src/methods2.py: смена типа стоит pen) ---------- */
function viterbi(Di, K, pen) {
  const Tn = Di.length; let cost = Di[0].slice(); const back = [];
  for (let t = 1; t < Tn; t++) {
    let best = Infinity, arg = 0; for (let k = 0; k < K; k++) if (cost[k] < best) { best = cost[k]; arg = k; }
    const nb = new Array(K), nc = new Array(K);
    for (let k = 0; k < K; k++) { const stay = cost[k], sw = best + pen; if (stay <= sw) { nb[k] = k; nc[k] = Di[t][k] + stay; } else { nb[k] = arg; nc[k] = Di[t][k] + sw; } }
    back.push(nb); cost = nc;
  }
  const path = new Array(Tn); let b = 0; for (let k = 1; k < K; k++) if (cost[k] < cost[b]) b = k; path[Tn - 1] = b;
  for (let t = Tn - 1; t >= 1; t--) path[t - 1] = back[t - 1][path[t]];
  return path;
}
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const switches = p => p.reduce((s, v, i) => s + (i && v !== p[i - 1] ? 1 : 0), 0);
function moSeries(i) { const k = KD(); return Array.from({ length: T }, (_, t) => k.dist[t][i]); }
function findDemo() {
  if (S.demo && S.demo.k === S.k) return S.demo.i; const k = KD(), pen = k.lam * k.medmin; let best = -1, bi = 0;
  for (let i = 0; i < N; i++) { const Di = moSeries(i), a = switches(Di.map(argmin)), b = switches(viterbi(Di, k.names.length, pen)); const sc = (a - b) * 10 + a; if (sc > best) { best = sc; bi = i; } }
  S.demo = { k: S.k, i: bi }; return bi;
}

/* ---------- вкладка «Метод» ---------- */
const CAT = THEME.cat;
let SHARE_MED = null;
function stackBar(vals, label) {
  const row = h("div", { style: "display:grid;grid-template-columns:130px 1fr;gap:8px;align-items:center;margin:5px 0;font-size:12px" }, h("span", { class: "muted", text: label }));
  const bar = h("div", { style: "display:flex;gap:2px;height:22px" });
  vals.forEach((v, c) => bar.append(h("div", { style: `flex:${Math.max(v, .1)};background:${CAT[c][1]};border-radius:3px;color:${C.onCat};font-size:11px;display:flex;align-items:center;justify-content:center;min-width:2px;overflow:hidden`, title: `${CAT[c][0]}: ${fmt(v, 1)}%`, text: v >= 7 ? fmt(v, 0) : "" })));
  row.append(bar); return row;
}
function renderMethod() {
  const gs = D.methods.find(m => m.name === "GS-TKM");
  if (gs && $("#g-rank")) { const kk = KD(), tr = kk.flow.reduce((a, r, j) => a + r[j], 0); $("#g-rank").textContent = gs.net[0] + "-е место из " + D.methods.length; $("#g-rank-n").textContent = "по эталонному AVI (медиана по девяти оценкам, диапазон " + gs.net[1] + "–" + gs.net[2] + "); по силуэту SW — " + gs.feat[0] + "-е (диапазон " + gs.feat[1] + "–" + gs.feat[2] + ")"; $("#g-ari").textContent = fmt(gs.ari, 3); $("#g-churn").textContent = pct(1 - tr / N, 1); }
  if (S.sel == null) S.sel = findDemo(); const i = S.sel, k = KD(), end = k.L[T - 1][i];
  [...$$(".m-name"), $("#m-name")].filter(Boolean).forEach(e => { e.textContent = D.mo.name[i] + " · " + D.mo.region[i]; });
  if (!SHARE_MED) SHARE_MED = [0, 1, 2, 3, 4, 5].map(c => median(D.shares.v.map(r => r[c])));
  const inType = D.shares.v.filter((_, j) => k.L[T - 1][j] === end), typeMean = [0, 1, 2, 3, 4, 5].map(c => inType.reduce((a, r) => a + r[c], 0) / inType.length);
  const s1 = $("#s1"); s1.replaceChildren(stackBar(D.shares.v[i], "это МО"), stackBar(typeMean, "среднее по типу"), stackBar(SHARE_MED, "медиана всех МО"),
    h("div", { class: "gap-s", style: "margin:6px 0 10px" }, ...CAT.map(([n, c]) => h("span", { class: "chip", style: "padding:2px 10px;font-size:12px" }, h("span", { class: "sw", style: `background:${c}` }), n))),
    h("p", { class: "muted", text: `Траты на жителя: ${D.spend[i].toLocaleString("ru-RU")} ₽ в месяц (медиана по МО ${Math.round(median(D.spend)).toLocaleString("ru-RU")} ₽). Тип по итогам периода: ${k.names[end]}.` }));
  // шаг 2: облако МО (положение точки — географическое), цвет — тип; наведение рисует 10 ближайших по сходству
  const svg = sv("svg", { viewBox: `0 0 ${GEO.W} ${GEO.H}`, class: "chart", role: "img", "aria-label": "Облако МО: наведите курсор, чтобы увидеть самых похожих" });
  const gDots = sv("g"), gSel = sv("g", { "pointer-events": "none" }), gHov = sv("g", { "pointer-events": "none" }); svg.append(gDots, gSel, gHov);
  for (let j = 0; j < N; j++) gDots.append(sv("circle", { cx: GEO.px[j], cy: GEO.py[j], r: 1.9, fill: tcol(S.k, typeAt(j)), opacity: .78 }));
  const arcs = (g, from, w, op, col, rr) => { const x1 = GEO.px[from], y1 = GEO.py[from]; D.nb[from].forEach(j => { const x2 = GEO.px[j], y2 = GEO.py[j], dx = x2 - x1, dy = y2 - y1; g.append(sv("path", { d: `M${x1} ${y1}Q${(x1 + x2) / 2 - dy * .22} ${(y1 + y2) / 2 + dx * .22} ${x2} ${y2}`, fill: "none", stroke: col, "stroke-width": w, opacity: op }), sv("circle", { cx: x2, cy: y2, r: rr, fill: tcol(S.k, typeAt(j)), stroke: C.bg, "stroke-width": 1.2 })); }); g.append(sv("circle", { cx: x1, cy: y1, r: rr + 2.5, fill: C.sel, stroke: C.bg, "stroke-width": 1.5 })); };
  arcs(gSel, i, 1.5, .9, C.arc, 4);
  let last = -1;
  svg.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), k = GEO.W / r.width, mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k; let b = -1, bd = 14 * 14;
    for (let j = 0; j < N; j++) { const dx = GEO.px[j] - mx, dy = GEO.py[j] - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; b = j; } }
    if (b < 0) { gHov.replaceChildren(); hideTip(); last = -1; svg.style.cursor = "default"; return; }
    svg.style.cursor = "pointer"; if (b !== last) { gHov.replaceChildren(); arcs(gHov, b, 1.2, .6, C.arcHov, 3.4); last = b; }
    const sm = D.nb[b].filter(j => D.mo.region[j] === D.mo.region[b]).length, ty = typeAt(b);
    showTip(e, [h("b", { text: D.mo.name[b] }), h("span", { class: "muted", text: D.mo.region[b] }), h("div", {}, h("span", { class: "dot", style: `background:${tcol(S.k, ty)};margin-right:6px` }), KD().names[ty]), h("span", { class: "muted", text: `из 10 самых похожих в своём регионе: ${sm}. Нажмите, чтобы выбрать МО` })]);
  });
  svg.addEventListener("mouseleave", () => { gHov.replaceChildren(); hideTip(); last = -1; });
  svg.addEventListener("click", () => { if (last >= 0) pick(last); });
  const same = D.nb[i].filter(j => D.mo.region[j] === D.mo.region[i]).length, list = h("div", { class: "nbs", style: "margin-top:10px" });
  D.nb[i].forEach((j, s) => list.append(h("button", { onclick: () => pick(j) }, h("span", { class: "dot", style: `background:${tcol(S.k, typeAt(j))}` }), h("span", {}, D.mo.name[j], h("span", { class: "muted", text: " · " + D.mo.region[j] })), h("span", { class: "tag" + (D.mo.region[j] === D.mo.region[i] ? "" : " oth"), text: D.mo.region[j] === D.mo.region[i] ? "свой регион" : "другой регион" }))));
  $("#s2").replaceChildren(...(STORY ? [] : [h("div", { class: "muted", style: "font-size:13px;margin-bottom:6px", text: "Наведите курсор на любую точку: дуги покажут 10 самых похожих на неё МО. Нажмите на точку, чтобы выбрать это МО." })]), svg,
    h("p", { style: "margin-top:8px" }, h("b", { text: `${same} из 10` }), ` ближайших к выбранному МО лежат в том же регионе; по всей стране таких в среднем ${pct(D.same_region_share, 0)}. Соседи по сети — экономики того же типа из разных регионов.`), list);
  renderS3(); renderS4();
  const sy = D.synth.find(r => r.key === "knn_graph"), sc = D.synth.find(r => r.key === "comp_graph"), sd = D.synth.find(r => r.key === "drift"); $("#m-not").textContent = ` Выигрыш от сети не гарантирован: на синтетике с известной истиной, где граф построен из тех же признаков, как в нашей обработке, качество с графовым сглаживанием ${fmt(sy.nmi, 3)} против ${fmt(sy.nmi0, 3)} без него (NMI); граф помогает, когда несёт дополнительную информацию (${fmt(sc.nmi, 3)} против ${fmt(sc.nmi0, 3)}). При дрейфе центров типов метод уступает простому k-means по окнам (${fmt(sd.nmi, 3)} против ${fmt(sd.nmi_km, 3)}). Сеть нужна для согласия разбиения со структурой связей и для поиска аналогов. Подробности — в главе 6.`;
}
function renderS3() {
  const i = S.sel, k = KD(), raw = D.x_raw[i], sm = D.xs[i], cent = k.centers, box = $("#s3"), dw = D.dimw, px = (a, d) => a[d] / dw[d];
  const alpha = box.dataset.alpha ? +box.dataset.alpha : 0, x = raw.map((v, d) => (1 - alpha) * v + alpha * sm[d]);
  const dist = a => cent.map(c => a.reduce((s, v, d) => s + (v - c[d]) ** 2, 0)), d0 = dist(raw), d1 = dist(sm), da = dist(x), maxd = Math.max(...d0, ...d1, ...da);
  const W = 660, rowH = 24, top = 26, Hh = top + D.dims.length * rowH + 8, X0 = 260, X1 = W - 20, sx = v => X0 + (clamp(v, -4, 4) + 4) / 8 * (X1 - X0), svg = sv("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "chart", role: "img", "aria-label": "Профиль до и после сглаживания" });
  svg.append(sv("line", { x1: sx(0), x2: sx(0), y1: top - 6, y2: Hh - 4, class: "axis" }));
  [-4, -2, 2, 4].forEach(v => svg.append(sv("line", { x1: sx(v), x2: sx(v), y1: top - 6, y2: Hh - 4, class: "grid" }), sv("text", { x: sx(v), y: 10, "text-anchor": "middle" }, String(v))));
  D.dims.forEach((nm, d) => { const y = top + d * rowH + 7; svg.append(sv("text", { x: X0 - 8, y: y + 4, "text-anchor": "end" }, nm), sv("line", { x1: sx(px(raw, d)), x2: sx(px(sm, d)), y1: y, y2: y, stroke: C.conn, "stroke-width": 3, "stroke-linecap": "round" }), sv("circle", { cx: sx(px(raw, d)), cy: y, r: 4, fill: "none", stroke: C.sel, "stroke-width": 1.5 }), sv("circle", { cx: sx(px(sm, d)), cy: y, r: 4, fill: C.acc }), alpha > 0 && alpha < 1 ? sv("circle", { cx: sx(px(x, d)), cy: y, r: 3, fill: C.sel }) : null); });
  const nm0 = k.names[argmin(d0)], nm1 = k.names[argmin(d1)], nma = k.names[argmin(da)];
  const bars = h("div", { style: "margin-top:10px" }); k.names.forEach((n, t) => { const best = t === argmin(da); bars.append(h("div", { style: "display:grid;grid-template-columns:170px 1fr 52px;gap:8px;align-items:center;font-size:12px;margin:3px 0" }, h("span", { style: best ? "font-weight:600" : "color:var(--muted)" }, h("span", { class: "dot", style: `background:${tcol(S.k, t)};margin-right:6px` }), n), h("div", { class: "bar", style: "height:11px" }, h("b", { style: `left:0;width:${da[t] / maxd * 100}%;height:11px;background:${best ? "var(--accent)" : C.g4}` })), h("span", { class: "num muted", text: fmt(da[t], 1) }))); });
  if (!box.querySelector(".view")) box.replaceChildren(h("div", { class: "field", style: "margin-bottom:6px" }, h("label", { for: "alpha", text: "исходный профиль" }), h("input", { id: "alpha", type: "range", min: 0, max: 100, value: 0, "aria-label": "Доля сглаживания", oninput: e => { box.dataset.alpha = e.target.value / 100; renderS3(); } }), h("span", { text: "сглаженный" })), h("div", { class: "view" }));
  box.querySelector(".view").replaceChildren(svg, h("div", { class: "muted", style: "font-size:12px;margin-top:4px", text: "белый кружок — исходное значение признака, зелёная точка — после усреднения с соседями; расстояние в σ" }),
    h("h4", { style: "margin-top:12px", text: "Расстояние до центров типов (меньше — ближе)" }), bars,
    h("p", { style: "margin-top:8px" }, "Без сглаживания ближайший тип: ", h("b", { text: nm0 }), ", после: ", h("b", { text: nm1 }), nm0 === nm1 ? ". У этого МО сглаживание ничего не меняет." : ". Сеть изменила тип этого МО."),
    h("p", { class: "muted", text: `Для всей выборки: у ${pct(k.diff_s0, 1)} МО тип со сглаживанием и без него разный (ARI между разбиениями ${fmt(k.ari_s0, 2)}).` }));
}
const PEN = v => D.sens.find(r => r.factor === "transition_penalty" && r.value === v);
function renderS4() {
  const i = S.sel, k = KD(), K = k.names.length, box = $("#s4"), lam = box.dataset.lam != null ? +box.dataset.lam : k.lam, Di = moSeries(i), pen = lam * k.medmin;
  const ind = Di.map(argmin), vp = viterbi(Di, K, pen), W = 640, rowH = 34, top = 14, left = 150, Hh = top + K * rowH + 26, sx = t => left + t / (T - 1) * (W - left - 14), sy = c => top + c * rowH + rowH / 2;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "chart", role: "img", "aria-label": "Траектория типа по месяцам" });
  k.names.forEach((n, c) => svg.append(sv("line", { x1: left, x2: W - 10, y1: sy(c), y2: sy(c), class: "grid" }), sv("text", { x: left - 8, y: sy(c) + 4, "text-anchor": "end" }, k.short[c])));
  [0, 3, 6, 9, T - 1].forEach(t => svg.append(sv("text", { x: sx(t), y: Hh - 8, "text-anchor": t === T - 1 ? "end" : t === 0 ? "start" : "middle" }, monthLabel(t))));
  ind.forEach((c, t) => svg.append(sv("circle", { cx: sx(t), cy: sy(c), r: 5, fill: tcol(S.k, c), opacity: .45 })));
  let d = ""; vp.forEach((c, t) => { d += (t ? `L${sx(t)} ${sy(vp[t - 1])}` : "M") + (t ? "" : `${sx(t)} ${sy(c)}`) + `L${sx(t)} ${sy(c)}`; });
  svg.append(sv("path", { d, fill: "none", stroke: C.sel, "stroke-width": 2.4, "stroke-linejoin": "round" }), ...vp.map((c, t) => sv("circle", { cx: sx(t), cy: sy(c), r: 3.4, fill: tcol(S.k, c), stroke: C.bg, "stroke-width": 1 })));
  const a = switches(ind), b = switches(vp);
  if (!box.querySelector(".view")) box.replaceChildren(h("div", { class: "field", style: "margin-bottom:6px" }, h("label", { for: "lam", text: "штраф τ, доли медианы" }), h("input", { id: "lam", type: "range", min: 0, max: 100, value: Math.round(k.lam * 100), "aria-label": "Штраф за смену типа", oninput: e => { box.dataset.lam = e.target.value / 100; renderS4(); } }), h("b", { class: "num", id: "lamv" })), h("div", { class: "view" }));
  $("#lamv").textContent = fmt(lam, 2);
  box.querySelector(".view").replaceChildren(svg, h("div", { class: "muted", style: "font-size:12px;margin-top:4px", text: "прозрачные точки — ближайший центр в каждом месяце независимо; белая линия — траектория по Витерби при выбранном штрафе" }),
    h("p", { style: "margin-top:8px" }, "Смен типа за 13 окон: ", h("b", { text: String(a) }), " при независимом выборе, ", h("b", { text: String(b) }), ` при штрафе ${fmt(lam, 2)} медианы. `, Math.abs(lam - k.lam) < 1e-9 ? "Это основное значение штрафа метода." : ""),
    a === 0 ? h("p", { class: "hint-flat" }, "У этого МО ближайший центр не меняется ни в одном месяце, поэтому штраф ничего не меняет. ", h("button", { class: "btn", onclick: () => pick(findDemo()), text: "показать МО со сменами типа" })) : h("span"),
    h("p", { class: "muted", text: `Для основной типологии доля сменивших тип за год: ${pct(PEN("0").changed)} без штрафа, ${pct(PEN("0.5").changed)} при основном штрафе, ${pct(PEN("1.0").changed)} при двойном (ARI с основной типологией при отсутствии штрафа ${fmt(PEN("0").ari, 2)}).` }));
}

/* ---------- общие элементы графиков ---------- */
function card(title, sub, ...kids) { return h("div", { class: "card" }, h("div", { class: "sec-title" }, h("h3", { text: title }), sub ? h("span", { class: "muted", style: "font-size:13px", text: sub }) : null), ...kids); }
function twin(chart, headers, rows) {
  const tb = h("div", { class: "scroll-x twin", hidden: true }, h("table", {}, h("thead", {}, h("tr", {}, ...headers.map((x, i) => h("th", { class: i ? "r" : "", text: x })))), h("tbody", {}, ...rows.map(r => h("tr", {}, ...r.map((c, i) => h("td", { class: i ? "r num" : "", text: String(c) })))))));
  const btn = h("button", { class: "btn", style: "font-size:12px;padding:3px 10px", "aria-expanded": "false", onclick: () => { tb.hidden = !tb.hidden; btn.setAttribute("aria-expanded", String(!tb.hidden)); btn.textContent = tb.hidden ? "таблица" : "график"; chart.style.display = tb.hidden ? "" : "none"; } }, "таблица");
  return h("div", {}, h("div", { class: "twin-bar", style: "margin-bottom:8px" }, btn), chart, tb);
}
function hov(el, fn) { el.addEventListener("mousemove", e => showTip(e, fn())); el.addEventListener("mouseleave", hideTip); return el; }
const hit = (x, y, w, hh, fn) => hov(sv("rect", { x, y, width: w, height: hh, fill: "rgba(0,0,0,0)", style: "cursor:default" }), fn);
const tl = (title, ...rows) => [h("b", { text: title }), ...rows.map(r => h("div", { class: "muted", text: r }))];
const axisText = (svg, x, y, t, anchor = "middle") => svg.append(sv("text", { x, y, "text-anchor": anchor }, t));

/* ---------- вкладка «Динамика» ---------- */
function renderDyn() {
  const k = KD(), K = k.names.length, box = $("#dyn"); box.replaceChildren();
  const flow = k.flow, tot = flow.flat().reduce((a, b) => a + b, 0), stay = flow.reduce((a, r, i) => a + r[i], 0), moved = tot - stay, never = 1 - k.changed.reduce((a, b) => a + b, 0) / N, mainK = D.K[String(D.main)];
  box.append(h("div", {}, h("h2", { text: "Большинство МО тип не меняют; часть смен — общий сдвиг потребления" }), h("p", { class: "lead", style: "margin-top:6px", text: `${STORY ? "" : "Наведите курсор на графики: подсказки покажут значения, кнопка «таблица» — те же данные таблицей. "}За 13 окон ${pct(never)} МО ни разу не меняли тип; к концу периода тип отличается от начального у ${moved} из ${tot} (${pct(moved / tot)}). Для основной типологии (K = ${D.main}) из ${D.trans.n} смен ${D.trans.robust} устойчивы в описательном смысле: новый тип сохраняется на трёх конечных точках, нижняя граница согласия не ниже 0,75.` })));
  // матрица потоков
  const mx = Math.max(...flow.flatMap((r, i) => r.filter((_, j) => j !== i))), tbl = h("table", { class: "heat" }, h("caption", { class: "sr", text: "Число МО по типу в начале и в конце периода" }));
  tbl.append(h("thead", {}, h("tr", {}, h("th", { text: "из \\ в" }), ...k.names.map((n, j) => h("th", { style: "text-align:center;font-size:12px" }, h("span", { class: "dot", style: `background:${tcol(S.k, j)};margin-right:4px` }), k.short[j])))));
  tbl.append(h("tbody", {}, ...flow.map((r, i) => h("tr", {}, h("th", {}, h("span", { class: "dot", style: `background:${tcol(S.k, i)};margin-right:6px` }), k.names[i]), ...r.map((v, j) => { const off = i !== j, u = off ? clamp(v / mx, 0, 1) : 0; return h("td", { style: `background:${off ? mix(C.hLo, C.acc, Math.sqrt(u) * .9) : C.diag};color:${off && u > .45 ? C.onHeat : "var(--text)"};font-weight:${off && v ? 600 : 400}`, onmousemove: e => showTip(e, tl(`${k.names[i]} → ${k.names[j]}`, i === j ? `${v} МО остались в типе` : `${v} МО сменили тип`)), onmouseleave: hideTip, text: String(v) }); })))));
  const c1 = card("Откуда и куда", "диагональ — остались в типе, цвет — число смен", h("div", { class: "scroll-x" }, tbl));
  // смены по месяцам
  const W = 560, Hh = 190, pl = 34, pb = 26, cm = k.changes_by_month, my = Math.max(...cm, 1), bw = (W - pl - 8) / cm.length, svg = sv("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "chart", role: "img", "aria-label": "Число смен типа по месяцам" });
  [0, .5, 1].forEach(f => { const y = Hh - pb - f * (Hh - pb - 12); svg.append(sv("line", { x1: pl, x2: W - 8, y1: y, y2: y, class: "grid" }), sv("text", { x: pl - 6, y: y + 4, "text-anchor": "end" }, String(Math.round(f * my)))); });
  cm.forEach((v, t) => { const hh = v / my * (Hh - pb - 12), x = pl + t * bw + 2, r = sv("rect", { x, y: Hh - pb - hh, width: bw - 4, height: hh, rx: 2, fill: C.acc }); r.addEventListener("mousemove", e => showTip(e, tipHtml(`${monthLabel(t + 1)}`, `смен типа: ${v}`))); r.addEventListener("mouseleave", hideTip); svg.append(r); });
  [0, 3, 6, 9].forEach(t => axisText(svg, pl + t * bw + bw / 2, Hh - 8, monthLabel(t + 1)));
  const c2 = card("Смены типа по окнам", "сколько МО поменяли тип по сравнению с предыдущим окном", twin(svg, ["окно до", "смен"], cm.map((v, t) => [monthLabel(t + 1), v])));
  box.append(h("div", { class: "cols3" }, c1, c2));
  // общий дрейф
  const row = (label, v, col) => h("div", { style: "display:grid;grid-template-columns:230px 1fr 54px;gap:10px;align-items:center;font-size:13px;margin:6px 0" }, h("span", { text: label }), h("div", { class: "bar", style: "height:14px" }, h("b", { style: `left:0;width:${v * 100 * 2.5}%;height:14px;background:${col}` })), h("span", { class: "num", text: pct(v, 1) }));
  const c3 = card("Общий сдвиг потребления", "основная типология: доля сменивших тип за год", h("p", { class: "muted", text: "Номинальные расходы и доля маркетплейсов растут во всех территориях. Если вычесть из каждого окна медиану признаков расходов по территориям, смен становится заметно меньше." }), row("основная типология", D.drift.main, "var(--accent)"), row("вычтен только уровень расходов", D.drift.level, C.g1), row("вычтены все координаты расходов", D.drift.centered, C.g5),
    h("p", { style: "margin-top:8px" }, "ARI типологии с вычитанием медианы и основной ", h("b", { text: fmt(D.drift.ari_centered, 2) }), ". Часть смен типа отражает сдвиг страны целиком, а не изменение положения территории относительно остальных."),
    h("p", { class: "muted", style: "font-size:12px", text: "Описательная оценка: центры типов обучены на всех окнах, и общий сдвиг уводит территории от центров. Причины смен метод не оценивает." }));
  // маркетплейсы
  const W2 = 600, H2 = 260, pl2 = 38, pr = 150, pb2 = 24, all = k.marketplaces.flat(), y0 = Math.floor(Math.min(...all)), y1 = Math.ceil(Math.max(...all)), sx = t => pl2 + t / (T - 1) * (W2 - pl2 - pr), sy = v => H2 - pb2 - (v - y0) / (y1 - y0) * (H2 - pb2 - 10);
  const s2 = sv("svg", { viewBox: `0 0 ${W2} ${H2}`, class: "chart", role: "img", "aria-label": "Доля маркетплейсов в расходах по типам" });
  for (let v = y0; v <= y1; v += Math.max(1, Math.round((y1 - y0) / 5))) s2.append(sv("line", { x1: pl2, x2: W2 - pr, y1: sy(v), y2: sy(v), class: "grid" }), sv("text", { x: pl2 - 6, y: sy(v) + 4, "text-anchor": "end" }, v + "%"));
  [0, 3, 6, 9, 12].forEach(t => axisText(s2, sx(t), H2 - 6, monthLabel(t)));
  const ends = k.marketplaces.map((r, c) => ({ c, y: sy(r[T - 1]) })).sort((a, b) => a.y - b.y); for (let q = 1; q < ends.length; q++) if (ends[q].y - ends[q - 1].y < 14) ends[q].y = ends[q - 1].y + 14;
  k.marketplaces.forEach((r, c) => s2.append(sv("polyline", { points: r.map((v, t) => `${sx(t)},${sy(v)}`).join(" "), fill: "none", stroke: tcol(S.k, c), "stroke-width": 2.2, "stroke-linejoin": "round" })));
  ends.forEach(e => s2.append(sv("text", { x: W2 - pr + 8, y: e.y + 4, class: "lbl" }, k.short[e.c])));
  const cross = sv("line", { y1: 8, y2: H2 - pb2, stroke: C.cross, "stroke-width": 1, opacity: 0 }); s2.append(cross);
  s2.addEventListener("mousemove", e => { const r = s2.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W2, t = clamp(Math.round((px - pl2) / (W2 - pl2 - pr) * (T - 1)), 0, T - 1); cross.setAttribute("x1", sx(t)); cross.setAttribute("x2", sx(t)); cross.setAttribute("opacity", 1);
    showTip(e, [h("b", { text: monthLabel(t) }), ...k.names.map((n, c) => h("div", {}, h("span", { class: "dot", style: `background:${tcol(S.k, c)};margin-right:6px` }), h("b", { text: fmt(k.marketplaces[c][t], 1) + "% " }), h("span", { class: "muted", text: n })))]); });
  s2.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  const c4 = card("Доля маркетплейсов в расходах", "годовое окно; медиана по МО каждого типа, типы по итогам периода", twin(s2, ["окно до", ...k.names], D.months.map((m, t) => [monthLabel(t), ...k.names.map((_, c) => fmt(k.marketplaces[c][t], 1) + "%")])),
    h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: "Структура расходов, включая долю маркетплейсов, входит в признаки метода, поэтому график описывает типы, а не независимо проверяет гипотезу." }));
  box.append(h("div", { class: "cols3" }, c3, c4));
  box.append(yoyCard(k));
}

/* изменение структуры расходов за год по типам (таблица-«тепловая карта») */
function yoyCard(k) {
  const yy = k.yoy, cols = [...CAT.map(c => c[0]), "уровень трат"], rowsY = k.names.map((_, t) => [...yy.share[t], yy.level[t]]);
  const fs = v => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), 1);
  const cellBg = (v, mx) => { const a = Math.min(1, Math.abs(v) / mx) * .55; return v >= 0 ? `rgba(79,208,106,${a.toFixed(2)})` : `rgba(235,170,45,${a.toFixed(2)})`; };
  const head = h("tr", {}, h("th", { text: "тип" }), ...cols.map((c, j) => h("th", { class: "r" }, c, h("small", { text: j < 6 ? "п.п." : "%" }))));
  const body = rowsY.map((r, t) => h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${tcol(S.k, t)};margin-right:8px` }), k.names[t]), ...r.map((v, j) => h("td", { class: "r num", style: `background:${cellBg(v, j < 6 ? 5 : 20)}`, text: fs(v) + (j < 6 ? "" : "%") }))));
  const ytab = h("table", { class: "yoy" }, h("thead", {}, head), h("tbody", {}, ...body));
  const rngOf = j => { const v = rowsY.map(r => r[j]); return [Math.min(...v), Math.max(...v)]; };
  const order = [0, 1, 2, 3, 4, 5].sort((a, b) => Math.abs(rngOf(b)[0] + rngOf(b)[1]) - Math.abs(rngOf(a)[0] + rngOf(a)[1]));
  const rg = j => { const [lo, hi] = rngOf(j); return fs(lo) + "…" + fs(hi) + " п.п."; };
  const note = `Сильнее всего изменилась доля категории «${cols[order[0]]}» (${rg(order[0])} по типам), затем «${cols[order[1]]}» (${rg(order[1])}). Номинальный уровень трат вырос на ${fmt(rngOf(6)[0], 1)}–${fmt(rngOf(6)[1], 1)}% в зависимости от типа. Это описательная статистика по типам, а не проверка гипотезы.`;
  const rowsT = k.names.map((n, t) => [n, ...rowsY[t].map((v, j) => fs(v) + (j < 6 ? " п.п." : "%"))]);
  return card("Что изменилось в расходах за год", "медиана изменения по МО каждого типа: год до декабря 2024 против года до декабря 2023; доли категорий в п.п., уровень трат номинальный, в %", twin(h("div", { class: "scroll-x" }, ytab), ["тип", ...cols], rowsT), h("p", { class: "yoy-note", text: note }));
}

/* ---------- вкладка «Надёжность» ---------- */
function renderRel() {
  const k = KD(), K = k.names.length, box = $("#rel"); box.replaceChildren();
  box.append(h("div", {}, h("h2", { text: "Насколько типам можно верить" }), h("p", { class: "lead", style: "margin-top:6px", text: "Типы устойчивы при пересборке на подвыборках, метод не зависит от случайной инициализации, но по индексам одного окна он не лидирует, а сеть помогает лишь при дополнительной информации. Наведите курсор на любой график, чтобы увидеть значения." })));
  // 1. Хеннинг
  const W = 600, rh = 44, top = 24, Hh = top + K * rh + 8, X0 = 150, X1 = W - 20, sx = v => X0 + (v - 0.4) / 0.6 * (X1 - X0), hs = sv("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "chart", role: "img", "aria-label": "Устойчивость типов на подвыборках" });
  [.5, .6, .7, .8, .9, 1].forEach(v => hs.append(sv("line", { x1: sx(v), x2: sx(v), y1: top - 6, y2: Hh - 4, class: "grid" }), sv("text", { x: sx(v), y: 10, "text-anchor": "middle" }, fmt(v, 1))));
  [[.75, "порог 0,75"], [.85, "высокая 0,85"]].forEach(([v, t]) => hs.append(sv("line", { x1: sx(v), x2: sx(v), y1: top - 6, y2: Hh - 4, stroke: C.thr, "stroke-width": 1 })));
  k.hennig.forEach((r, c) => { const y = top + c * rh + 14; hs.append(sv("text", { x: X0 - 10, y: y + 4, "text-anchor": "end", class: "lbl" }, k.short[c]), sv("line", { x1: sx(r.min), x2: sx(r.mean), y1: y, y2: y, stroke: C.g2, "stroke-width": 2 }), sv("rect", { x: X0, y: y - 7, width: sx(r.mean) - X0, height: 14, rx: 3, fill: tcol(S.k, c), opacity: .9 }), sv("circle", { cx: sx(r.min), cy: y, r: 4, fill: C.bg, stroke: C.fg, "stroke-width": 1.5 }), sv("text", { x: sx(r.mean) + 8, y: y + 4, class: "lbl" }, fmt(r.mean, 3)), hit(0, y - 20, W, 40, () => tl(k.names[c], `${r.n} МО`, `средний Жаккар ${fmt(r.mean, 3)}`, `худшая подвыборка ${fmt(r.min, 3)}`, `подвыборок выше 0,75: ${pct(r.share, 0)}`))); });
  const c1 = card("Устойчивость типов на подвыборках", "критерий Хеннига: 50 повторов на 80% МО", twin(hs, ["тип", "МО", "Жаккар (среднее)", "минимум", "подвыборок выше 0,75"], k.hennig.map((r, c) => [k.names[c], r.n, fmt(r.mean, 3), fmt(r.min, 3), pct(r.share, 0)])),
    h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: "Столбец — средний Жаккар типа, кружок — худшая подвыборка. Выше правой линии тип «высокоустойчив», выше левой — «устойчив»." }));
  // 2. выбор K
  const W2 = 520, H2 = 230, pl = 36, pb = 28, ks = D.kcurve, kx = i => pl + (i + .5) / ks.length * (W2 - pl - 10), ky = v => H2 - pb - (v - .5) / .5 * (H2 - pb - 12), s2 = sv("svg", { viewBox: `0 0 ${W2} ${H2}`, class: "chart", role: "img", "aria-label": "Согласие при пересборке в зависимости от числа типов" });
  [.5, .6, .7, .8, .9, 1].forEach(v => s2.append(sv("line", { x1: pl, x2: W2 - 10, y1: ky(v), y2: ky(v), class: "grid" }), sv("text", { x: pl - 6, y: ky(v) + 4, "text-anchor": "end" }, fmt(v, 1))));
  s2.append(sv("line", { x1: pl, x2: W2 - 10, y1: ky(.85), y2: ky(.85), stroke: C.thr }));
  ks.forEach((r, i) => { if (r.K === D.main) s2.append(sv("rect", { x: kx(i) - 16, y: 8, width: 32, height: H2 - pb - 8, fill: C.acc, opacity: .12 })); axisText(s2, kx(i), H2 - 8, "K=" + r.K); });
  const kHits = ks.map((r, i) => hit(kx(i) - 22, 8, 44, H2 - pb - 8, () => tl("K = " + r.K, `среднее согласие при пересборке (ARI): ${fmt(r.ari, 3)}`, `минимум по пересборкам: ${fmt(r.ari_low, 3)}`, `наименьший тип: ${pct(r.min_share, 1)}`, r.ok ? "допустим по протоколу" : "не проходит порог устойчивости")));
  s2.append(sv("polyline", { points: ks.map((r, i) => `${kx(i)},${ky(r.ari)}`).join(" "), fill: "none", stroke: C.accHi, "stroke-width": 2.2 }), sv("polyline", { points: ks.map((r, i) => `${kx(i)},${ky(r.ari_low)}`).join(" "), fill: "none", stroke: C.amber, "stroke-width": 2.2 }),
    ...ks.map((r, i) => sv("circle", { cx: kx(i), cy: ky(r.ari), r: 4, fill: r.ok ? C.accHi : C.bg, stroke: C.accHi, "stroke-width": 1.6 })), ...ks.map((r, i) => sv("circle", { cx: kx(i), cy: ky(r.ari_low), r: 4, fill: C.amber })), ...kHits);
  s2.append(sv("text", { x: W2 - 12, y: ky(ks[ks.length - 1].ari) - 8, "text-anchor": "end", class: "lbl" }, "среднее ARI"), sv("text", { x: W2 - 12, y: ky(ks[ks.length - 1].ari_low) + 16, "text-anchor": "end", class: "lbl" }, "минимум"));
  const c2 = card("Сколько типов выбрать", "основной K выделен; линия — порог 0,85", twin(s2, ["K", "ARI (среднее)", "ARI (минимум)", "наименьший тип", "допустим"], ks.map(r => [r.K, fmt(r.ari, 3), fmt(r.ari_low, 3), pct(r.min_share, 1), r.ok ? "да" : "нет"])),
    h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: "Допустим вариант с долей наименьшего типа не ниже 3% и средним ARI не ниже 0,85 при 10 пересборках; выбирается самый детальный. Зависимость немонотонна: K = 4 и 6 порог устойчивости не проходят. Правило сформулировано по ходу анализа и не регистрировалось заранее." }));
  box.append(h("div", { class: "cols3" }, c1, c2));
  // 3. сиды
  const ms = [...D.methods].sort((a, b) => a.ari - b.ari), W3 = 540, rh3 = 24, H3 = 14 + ms.length * rh3 + 22, X03 = 250, s3 = sv("svg", { viewBox: `0 0 ${W3} ${H3}`, class: "chart", role: "img", "aria-label": "Согласие методов между запусками" });
  [0, .25, .5, .75, 1].forEach(v => { const x = X03 + v * (W3 - X03 - 52); s3.append(sv("line", { x1: x, x2: x, y1: 4, y2: H3 - 18, class: "grid" }), sv("text", { x, y: H3 - 4, "text-anchor": "middle" }, fmt(v, 2))); });
  ms.forEach((m, i) => { const y = 10 + i * rh3, w = m.ari * (W3 - X03 - 52), us = m.name === "GS-TKM"; s3.append(sv("text", { x: X03 - 8, y: y + 12, "text-anchor": "end", class: us ? "lbl" : "" }, lab(m.name)), sv("rect", { x: X03, y: y + 2, width: w, height: 14, rx: 3, fill: us ? C.acc : C.g4 }), sv("text", { x: X03 + w + 6, y: y + 13, class: us ? "lbl" : "" }, fmt(m.ari, 3)), hit(0, y, W3, rh3, () => tl(m.name, `согласие между сидами (ARI): ${fmt(m.ari, 3)}`, m.churn == null ? "churn не считался" : `churn в месяц: ${fmt(m.churn, 3)}`, `SW ${fmt(m.sw, 3)}, AVI ${fmt(m.avi, 3)}, Q ${fmt(m.q, 3)}`))); });
  const c3 = card("Зависит ли результат от случайной инициализации", "согласие разбиений при трёх разных seed (последнее окно), ARI", twin(s3, ["метод", "ARI между сидами", "churn в месяц"], ms.map(m => [m.name, fmt(m.ari, 3), m.churn == null ? "—" : fmt(m.churn, 3)])), h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: "1 — запуски дают одно и то же разбиение. У DMoN, CANUS и KEFRiN разные запуски находят заметно разные типы." }));
  // 4. места методов
  const mode = box.dataset.rank || "net", rk = [...D.methods].sort((a, b) => a[mode][0] - b[mode][0] || a[mode][1] - b[mode][1]), W4 = 560, rh4 = 25, H4 = 24 + rk.length * rh4 + 6, X04 = 250, mxr = D.methods.length, rx = v => X04 + (v - 1) / (mxr - 1) * (W4 - X04 - 20), s4 = sv("svg", { viewBox: `0 0 ${W4} ${H4}`, class: "chart", role: "img", "aria-label": "Места методов с диапазонами" });
  for (let v = 1; v <= mxr; v += 2) s4.append(sv("line", { x1: rx(v), x2: rx(v), y1: 16, y2: H4 - 4, class: "grid" }), sv("text", { x: rx(v), y: 10, "text-anchor": "middle" }, String(v)));
  rk.forEach((m, i) => { const y = 28 + i * rh4, us = m.name === "GS-TKM", [md, lo, hi] = m[mode]; s4.append(sv("text", { x: X04 - 8, y: y + 4, "text-anchor": "end", class: us ? "lbl" : "" }, lab(m.name)), sv("line", { x1: rx(lo), x2: rx(hi), y1: y, y2: y, stroke: us ? C.accHi : C.g2, "stroke-width": 3, "stroke-linecap": "round" }), sv("circle", { cx: rx(md), cy: y, r: 5.5, fill: us ? C.acc : C.g1, stroke: C.bg, "stroke-width": 2 }), hit(0, y - 12, W4, rh4, () => tl(m.name, `медианное место ${md}, диапазон по девяти оценкам: ${lo}–${hi}`, `SW ${fmt(m.sw, 3)}, AVI ${fmt(m.avi, 3)}, Q ${fmt(m.q, 3)}`, `согласие между seed ${fmt(m.ari, 3)}`))); });
  const seg = h("span", { class: "seg" }, h("button", { "aria-pressed": mode === "net", text: "по AVI (граф)", onclick: () => { box.dataset.rank = "net"; renderRel(); } }), h("button", { "aria-pressed": mode === "feat", text: "по SW (признаки)", onclick: () => { box.dataset.rank = "feat"; renderRel(); } }));
  const me = D.methods.find(m => m.name === "GS-TKM");
  const c4 = card("Место по индексам качества", "медиана и диапазон по девяти оценкам (3 даты × 3 seed); левее — лучше", seg, h("div", { style: "margin-top:8px" }), twin(s4, ["метод", "медиана", "минимум", "максимум", "SW", "AVI", "Q"], rk.map(m => [m.name, m[mode][0], m[mode][1], m[mode][2], fmt(m.sw, 3), fmt(m.avi, 3), fmt(m.q, 3)])),
    h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: `Место по эталонному AVI (согласие со структурой связей) и по силуэту SW (компактность в пространстве признаков). Порядок разный: методы на модулярности лидируют по AVI, k-means без сети — по SW. У GS-TKM место ${me.net[0]}-е по AVI и ${me.feat[0]}-е по SW из ${D.methods.length}; остальные индексы в отчёте.` }));
  box.append(h("div", { class: "cols3" }, c3, c4));
  // 5. синтетика
  const W5 = 600, rh5 = 40, H5 = 12 + D.synth.length * rh5 + 24, X05 = 230, bx = v => v * (W5 - X05 - 60), s5 = sv("svg", { viewBox: `0 0 ${W5} ${H5}`, class: "chart", role: "img", "aria-label": "Когда сеть помогает на синтетических данных" });
  D.synth.forEach((r, i) => { const y = 8 + i * rh5; s5.append(sv("text", { x: X05 - 10, y: y + 20, "text-anchor": "end", class: "lbl" }, r.regime), sv("rect", { x: X05, y: y + 3, width: bx(r.nmi), height: 13, rx: 3, fill: C.acc }), sv("text", { x: X05 + bx(r.nmi) + 6, y: y + 14 }, fmt(r.nmi, 3)), sv("rect", { x: X05, y: y + 19, width: bx(r.nmi0), height: 13, rx: 3, fill: C.g3 }), sv("text", { x: X05 + bx(r.nmi0) + 6, y: y + 30 }, fmt(r.nmi0, 3)), hit(0, y, W5, rh5, () => tl(r.regime, `NMI с графовым сглаживанием ${fmt(r.nmi, 3)}, без него ${fmt(r.nmi0, 3)}, k-means по окнам ${fmt(r.nmi_km, 3)}`, `ложных смен: ${fmt(r.false, 3)} со сглаживанием, ${fmt(r.false0, 3)} без`, `мигранты, распознанные в месяц смены: ${fmt(r.mig, 2)}, через два месяца: ${fmt(r.mig2, 2)}`))); });
  s5.append(sv("rect", { x: X05, y: H5 - 14, width: 12, height: 8, rx: 2, fill: C.acc }), sv("text", { x: X05 + 18, y: H5 - 6 }, "с графовым сглаживанием (GS-TKM)"), sv("rect", { x: X05 + 210, y: H5 - 14, width: 12, height: 8, rx: 2, fill: C.g3 }), sv("text", { x: X05 + 228, y: H5 - 6 }, "без сглаживания"));
  const c5 = card("Когда сеть помогает: синтетика с известной истиной", "качество восстановления типов, NMI", twin(s5, ["режим", "со сглаживанием", "без сглаживания", "k-means по окнам", "ложных смен со сглаживанием", "ложных смен без"], D.synth.map(r => [r.regime, fmt(r.nmi, 3), fmt(r.nmi0, 3), fmt(r.nmi_km, 3), fmt(r.false, 3), fmt(r.false0, 3)])),
    h("p", { style: "margin-top:6px" }, "Сглаживание по сети помогает только тогда, когда сеть несёт информацию, которой нет в признаках. Когда сеть построена из тех же признаков (как в нашей обработке), выигрыша нет; при дрейфе центров типов метод уступает простому k-means по окнам."));
  // 6. нулевые модели
  const NMAP = { SW: "SW", CH: "CH", S_Dbw: "S_Dbw", AVI: "AVI (вершинный)", AVU: "AVU (вершинный)", ANUI: "ANUI (вершинный)", MQ: "MQ (вершинный)", Q: "Q", AVI_ref: "AVI (эталон)", AVU_ref: "AVU (эталон)", ANUI_ref: "ANUI (эталон)", MQ_ref: "MQ (эталон)" }, pmin = 1 / (D.n_null + 1);
  const nz = (r, n) => r.z[n] == null ? "—" : `${sgn(r.z[n], 1)} (${r.p[n] <= pmin + 1e-9 ? "p ≤ " + fmt(pmin, 3) : "p = " + fmt(r.p[n], 3)})`;
  const nt = h("table", {}, h("thead", {}, h("tr", {}, ...["индекс", "перестановка меток", "внутри регионов", "перестройка рёбер"].map((x, i) => h("th", { class: i ? "r" : "", text: x })))), h("tbody", {}, ...D.null.map(r => h("tr", {}, h("td", { text: NMAP[r.index] || r.index }), h("td", { class: "r num", text: nz(r, "labels") }), h("td", { class: "r num", text: nz(r, "region") }), h("td", { class: "r num", text: nz(r, "config") })))));
  const c6 = card("Индексы против нулевых моделей", `z-оценка (положительная — лучше нуля), ${D.n_null} нулевых разбиений каждого вида, среднее по трём окнам`, h("div", { class: "scroll-x" }, nt),
    h("p", { class: "muted", style: "font-size:12px;margin-top:6px", text: "Все индексы, кроме эталонного AVU, лучше нулевых. Эталонный AVU хуже нуля: его определение на уровне пар кластеров не отличает разбиение от случайного. Графовые индексы циклические: граф использован при сглаживании; z велики из-за малого разброса нулевых значений и не измеряют величину эффекта." }));
  box.append(c5, c6);
}

/* ---------- вкладка «Проверка» ---------- */
function renderCheck() {
  const box = $("#check"); box.replaceChildren(); const flt = box.dataset.f || "all", L = D.ledger, cnt = s => L.filter(r => r.s === s).length;
  box.append(h("div", {}, h("h2", { text: "Что подтвердилось, что нет и что с оговорками" }), h("p", { class: "lead", style: "margin-top:6px", text: "Каждое утверждение работы с результатом проверки. Статус обозначен знаком и словом, а не только цветом." })));
  const ST = { ok: ["✓", "подтвердилось", "ok"], no: ["✗", "не подтвердилось", "no"], cv: ["~", "с оговоркой", "cv"] };
  const seg = h("div", { class: "chips" }, ...[["all", `все ${L.length}`], ["ok", `подтвердилось ${cnt("ok")}`], ["no", `не подтвердилось ${cnt("no")}`], ["cv", `с оговоркой ${cnt("cv")}`]].map(([f, t]) => h("button", { class: "chip", "aria-pressed": flt === f, onclick: () => { box.dataset.f = f; renderCheck(); }, text: t })));
  const list = h("div", {}); L.filter(r => flt === "all" || r.s === flt).forEach(r => { const [ic, st, cl] = ST[r.s]; list.append(h("div", { class: "led" }, h("span", { class: "ic " + cl, "aria-hidden": "true", text: ic }), h("div", {}, h("div", { class: "t" }, r.t, h("span", { class: "st", style: `color:var(--${cl === "ok" ? "ok" : cl === "no" ? "bad" : "amber"})`, text: st })), h("div", { class: "r", text: r.r }), h("div", { class: "w", text: "методологический отчёт, " + r.w })))); });
  box.append(h("div", { class: "card" }, seg, list));
  // доказательства
  const fm4 = x => fmt(x, 4).replace("-", "−"), ci = (lo, hi) => `[${fm4(lo)}; ${fm4(hi)}]`;
  const tbl = (head, rows) => h("table", {}, h("thead", {}, h("tr", {}, ...head.map((x, i) => h("th", { class: i ? "r" : "", text: x })))), h("tbody", {}, ...rows.map(r => h("tr", {}, ...r.map((c, i) => h("td", { class: i ? "r num" : "", text: String(c) }))))));
  const LENS = { combined: "общая линза", spend: "линза потребления" };
  const et = tbl(["показатель", "n", "прирост R² сверх региона [интервал по регионам]", "q (BH)", "после контроля населения и зарплаты [интервал]"], D.valid.map(r => [`${r.name} (${LENS[r.lens]})`, r.n, `${fm4(r.d)} ${ci(r.lo, r.hi)}`, fmt(r.q, 3), r.dc == null ? "—" : `${fm4(r.dc)} ${ci(r.dclo, r.dchi)}`]));
  const FACT = { spending_weight: "вес блока расходов", neighbors: "число соседей", smoothing_steps: "шаги сглаживания", transition_penalty: "штраф (доли медианы)", seed: "seed k-means", unreported_coordinate: "координата нераспределённой занятости", city_policy: "Москва и Санкт-Петербург", window_months: "длина окна, мес.", window_centering: "вычет медианы по территориям" };
  const VAL = { excluded: "исключена", original: "внутригородские МО раздельно", drop: "столицы исключены", level: "только уровень расходов", all_spending: "все координаты расходов" };
  const st = tbl(["параметр", "значение", "ARI с основной типологией", "сменили тип за год"], D.sens.map(r => [FACT[r.factor] || r.factor, VAL[r.value] || (r.factor === "seed" ? String(+r.value + 1) : r.value.replace(".", ",")), fmt(r.ari, 3), pct(r.changed, 1)]));
  const cv = tbl(["схема", "модель", "MAE, лог. пункты", "R²", "прирост от типа [интервал]"], D.cv.map(r => [r.protocol, r.model, fm4(r.mae), fmt(r.r2, 3), `${fm4(r.gain)} ${ci(r.glo, r.ghi)}`]));
  const ca = tbl(["месяцев истории", "МО", "присвоено", "согласие среди присвоенных", "нижняя граница Вильсона"], D.cal.map(r => [r.months, r.n, r.selected, pct(r.agree, 1), pct(r.wilson, 1)]));
  const evs = tbl(["территория", "регион", "окно", "из → в", "согласие", "нижняя граница"], D.events.map(r => [r.name, r.region, monthLabel(D.months.indexOf(r.month)), `${r.from + 1} → ${r.to + 1}`, fmt(r.agree, 2), fmt(r.lb, 2)]));
  box.append(h("div", { class: "cols3" },
    h("details", {}, h("summary", { text: "Внешняя проверка: показатели, не входившие в разбиение" }), h("div", { class: "scroll-x", style: "margin-top:8px" }, et)),
    h("details", {}, h("summary", { text: "Практическая проверка: прогноз роста расходов с типом и без" }), h("div", { class: "scroll-x", style: "margin-top:8px" }, cv)),
    h("details", {}, h("summary", { text: "Чувствительность к параметрам" }), h("div", { class: "scroll-x", style: "margin-top:8px" }, st))));
  box.append(h("div", { class: "cols3" },
    h("details", {}, h("summary", { text: "Присвоение типа МО с неполной историей: калибровка" }), h("div", { class: "scroll-x", style: "margin-top:8px" }, ca)),
    h("details", {}, h("summary", { text: `Устойчивые в описательном смысле смены типа (K = ${D.main})` }), h("div", { class: "scroll-x", style: "margin-top:8px" }, evs))));
  box.append(h("p", { class: "muted", style: "font-size:13px", text: "Подробности, формулы и список файлов результатов — в методологическом отчёте и в репозитории проекта." }));
}

/* ---------- вкладки, фильтры, запуск ---------- */
const PANELS = { answer: ["answer", "ans-mapslot", renderAnswer], method: ["method", null, renderMethod], map: ["map", "map-mapslot", () => { renderMapLegend(); renderPassport(); }], dyn: ["dyn", null, renderDyn], rel: ["rel", null, renderRel], check: ["check", null, renderCheck] };
function gotoTab(name, push = true) {
  S.tab = name; $$(".tab").forEach(b => { const on = b.dataset.tab === name; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; }); $$(".panel").forEach(p => p.classList.toggle("on", p.id === "p-" + name));
  const showF = ["answer", "map", "dyn"].includes(name); $("#filters").hidden = !showF; $("#monthfield").hidden = !["answer", "map"].includes(name); $("#searchfield").hidden = !["answer", "map"].includes(name);
  if (name === "answer" && !["type", "region"].includes(S.mode)) S.mode = "type"; if (name === "map" && S.mode === "region") S.mode = "type";
  $$("#ans-color button").forEach(b => b.setAttribute("aria-pressed", b.dataset.c === S.mode)); $$("#map-color button").forEach(b => b.setAttribute("aria-pressed", b.dataset.c === S.mode));
  const cfg = PANELS[name]; if (cfg[1]) mountMap(cfg[1]); cfg[2](); if (push) { try { history.replaceState(null, "", "#" + name); } catch (e) { /* файл открыт с диска: браузер запрещает менять адрес, это не мешает работе */ } } { const cv = document.querySelector(".cover"); if (!cv) window.scrollTo(0, 0); else if (push) window.scrollTo({ top: cv.offsetHeight - 4 }); } /* с титульным листом вкладка открывается сразу под ним, а при загрузке страницы остаётся виден сам титульник */
}
function setK(k) { S.k = k; S.iso = null; if (STORY) { $$("#kseg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.k === k)); storyAll(); return; } $$("#kseg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.k === k)); if (S.tab === "method") { $("#s3").replaceChildren(); $("#s4").replaceChildren(); } PANELS[S.tab][1] && paintMap(); PANELS[S.tab][2](); }
function setT(t) { S.t = t; $("#month").value = t; $("#mlab").textContent = monthLabel(t); if (STORY || PANELS[S.tab][1]) { paintMap(); renderPassport(); renderMapLegend(); } }
/* ---------- лонгрид ---------- */
/* появление блоков при прокрутке: короткий сдвиг и проявление, без эффектов на графиках; при reduce-motion и без IntersectionObserver всё видно сразу */
function revealOnScroll() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) return () => {};
  const sel = ".chap-head>*, .chap>.cue, .chap>.abstract, .chap>.note, .figure, .facts .fact, .trio>div, .parts>li, .gains>div, .params>div, .honest li, .repro>div, .chap-end, .sub, .edge-tab, .stack>.card, .stack>.cols3, .callout, .signoff";
  const els = $$(sel).filter(e => !e.closest(".rv") || e.matches(".chap-head>*"));
  const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting || en.boundingClientRect.top < 0) { en.target.classList.add("in"); io.unobserve(en.target); } }), { threshold: .08, rootMargin: "0px 0px -6% 0px" });
  els.forEach(e => { const sib = [...e.parentElement.children].filter(c => c.matches(sel)), i = Math.min(sib.indexOf(e), 4); e.style.setProperty("--d", (i * 70) + "ms"); e.classList.add("rv"); io.observe(e); });
  addEventListener("beforeprint", () => els.forEach(e => e.classList.add("in")));
  return () => els.forEach(e => { if (!e.classList.contains("in") && e.getBoundingClientRect().top < innerHeight * .94) e.classList.add("in"); }); /* страховка: блоки, мимо которых страница «перепрыгнула» (якорь, быстрая прокрутка) */
}
function renderEdges() {
  const box = $("#edge-tab"); if (!box || !D.edge) return; box.replaceChildren();
  const bar = v => h("span", { class: "ebar", "aria-hidden": "true" }, h("i", { style: `width:${Math.round(100 * v)}%` })), t = h("table", { class: "etab" }, h("thead", {}, h("tr", {}, ...["правило", "рёбер", "рёбер внутри региона", "пересечение рёбер с основной сетью", `типы совпадают с основным правилом: ${D.main} типов (ARI)`, "силуэт SW"].map(x => h("th", { text: x })))));
  const tb = h("tbody"); D.edge.forEach((r, i) => tb.append(h("tr", { class: i === 0 ? "base" : "" }, h("td", { text: r.name + (i === 0 ? " (основное)" : "") }), h("td", { class: "num", text: r.edges.toLocaleString("ru-RU") }), h("td", { class: "num", text: pct(r.within, 0) }), h("td", { class: "num", text: fmt(r.jacc, 2) }), h("td", { class: "num" }, bar(r.ari), h("span", { text: fmt(r.ari, 2) })), h("td", { class: "num", text: fmt(r.sw, 3) })))); t.append(tb);
  box.append(h("div", { class: "tscroll" }, t), h("p", { class: "muted", style: "font-size:12px", text: "ARI = 1 — типы те же, что при основном правиле; чем ниже, тем сильнее зависимость типов от определения похожести. Силуэт SW — признаковая оценка, не зависящая от сети." }));
}

/* ---------- материалы ---------- */
function saveBlob(name, data, type) { const b = new Blob([data], { type }), u = URL.createObjectURL(b), a = h("a", { href: u, download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 4000); }
const fromB64 = t => { const bin = atob(t), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; };
function initDownloads() {
  const box = $("#downloads"); if (!box) return; if (D.repo) $("#repo-link").replaceChildren(" (", h("a", { href: D.repo, target: "_blank", rel: "noopener", text: "GitHub" }), ")");
  const items = [];
  if (D.reportPDF) items.push(["Методологический отчёт, PDF", () => saveBlob("sberindex_methodology.pdf", fromB64(D.reportPDF), "application/pdf")]);
  if (D.reportMD) items.push(["Текст отчёта, Markdown", () => saveBlob("sberindex_methodology.md", D.reportMD, "text/markdown")]);
  if (D.repo) items.push(["Код и данные, ZIP (с GitHub)", () => window.open(D.repo + "/archive/refs/heads/main.zip", "_blank", "noopener")]);
  Object.entries(D.files || {}).forEach(([name, text]) => items.push([`Таблица ${name}`, () => saveBlob(name, text, "text/csv")]));
  box.replaceChildren(...items.map(([t, f]) => h("button", { class: "btn", style: "margin:0 8px 8px 0", onclick: f, text: t })));
}

function storyAll() { renderAnswer(); renderMapLegend(); renderPassport(); renderMethod(); renderDyn(); renderRel(); paintMap(); }
function storyStart() {
  S.tab = "story"; mountMap("ans-mapslot"); renderCheck(); renderEdges(); storyAll();
  const links = $$(".sn a"), secs = links.map(a => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean), bar = $("#prog");
  const upd = () => { const y = scrollY + innerHeight * .35; let cur = -1; secs.forEach((e, i) => { if (e.offsetTop <= y) cur = i; }); links.forEach((a, i) => { if (i === cur) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }); const mx = document.documentElement.scrollHeight - innerHeight; bar.style.transform = `scaleX(${mx > 0 ? clamp(scrollY / mx, 0, 1) : 0})`; revealCheck(); };
  const revealCheck = revealOnScroll();
  let raf = 0; addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; upd(); }); }, { passive: true }); addEventListener("resize", upd); upd();
}
function init() {
  initMap(); $("#month").max = T - 1; $("#month").value = S.t; $("#mlab").textContent = monthLabel(S.t);
  $("#mo-list").replaceChildren(...D.mo.name.map((n, i) => h("option", { value: n + " · " + D.mo.region[i] })));
  $$(".tab").forEach(b => { b.addEventListener("click", () => gotoTab(b.dataset.tab)); b.addEventListener("keydown", e => { const tabs = $$(".tab"), i = tabs.indexOf(b); if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); gotoTab(n.dataset.tab); } }); });
  $$("#kseg button").forEach(b => b.addEventListener("click", () => setK(b.dataset.k)));
  $("#month").addEventListener("input", e => setT(+e.target.value));
  let timer = null; $("#play").addEventListener("click", () => { if (timer) { clearInterval(timer); timer = null; $("#play").textContent = "▶"; return; } $("#play").textContent = "❚❚"; if (S.t >= T - 1) setT(0); timer = setInterval(() => { if (S.t >= T - 1) { clearInterval(timer); timer = null; $("#play").textContent = "▶"; return; } setT(S.t + 1); }, 450); });
  $("#q").addEventListener("change", e => { search(e.target.value); });
  $$("#ans-color button").forEach(b => b.addEventListener("click", () => { S.mode = b.dataset.c; $$("#ans-color button").forEach(x => x.setAttribute("aria-pressed", x === b)); paintMap(); }));
  $$("#map-color button").forEach(b => b.addEventListener("click", () => { S.mode = b.dataset.c; $$("#map-color button").forEach(x => x.setAttribute("aria-pressed", x === b)); renderMapLegend(); paintMap(); }));
  $("#hatch").addEventListener("change", e => { S.hatch = e.target.checked; paintMap(); });
  $("#m-rand").addEventListener("click", () => { pick(Math.floor(Math.random() * N)); }); $("#m-demo").addEventListener("click", () => { pick(findDemo()); });
  $("#nofail").textContent = D.ledger.filter(r => r.s === "no").length + " ✗"; initDownloads();
  if (STORY) { storyStart(); return; }
  const start = (location.hash || "").slice(1); gotoTab(PANELS[start] ? start : "answer", false);
}
init();


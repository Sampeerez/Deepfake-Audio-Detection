# -*- coding: utf-8 -*-
"""
app.py — Punto de entrada principal.

Configura página, CSS global y navegación multi-página ANTES de delegar en
cada página, de forma que el primer render ya sea estable (sin flash de
navegación auto-descubierta ni de estilos sin aplicar).

NOTA: el directorio de páginas se llama `app_pages/` (no `pages/`) a propósito:
un directorio `pages/` activa la navegación automática v1 de Streamlit, que
aparece un instante ("app", "0 Home", …) antes de que st.navigation la
reemplace. Renombrarlo elimina ese estado intermedio.

Lanzar con:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

# Única llamada a set_page_config de toda la app: las páginas individuales NO
# deben llamarla (el título de pestaña lo aporta cada st.Page).
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_helpers import PAGE_CSS  # noqa: E402

# CSS global inyectado aquí (no en cada página): ocupa siempre la misma
# posición en el árbol de elementos, así Streamlit lo reconcilia entre páginas
# y no hay flash de estilos por defecto al navegar.
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── Auto-cierre de desplegables al salir el ratón ────────────────────────────
# Streamlit/baseweb solo cierran los selectbox con un clic fuera. Inyectamos un
# script (iframe de altura 0, same-origin: el sandbox de st.iframe incluye
# allow-same-origin + allow-scripts) que cierra el desplegable abierto en cuanto
# el puntero no está ni sobre el popover ni sobre su control. Cierra con Escape
# y, como respaldo, simulando un clic fuera. El guard evita listeners duplicados.
with st.container(key="ddac_host"):
    st.iframe(
        """
<script>
(function(){
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  if (!doc || doc.__ddAutoCloseV4) return;
  doc.__ddAutoCloseV4 = true;
  var win = window.parent;

  // ── Animated particle-network ("spider-web") background ──────────────────
  function startWeb() {
    if (doc.getElementById('bgWeb')) return;
    var canvas = doc.createElement('canvas');
    canvas.id = 'bgWeb';
    doc.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    var W, H, pts;
    var N = 80, MAXD = 155;
    function resize() { W = canvas.width = win.innerWidth; H = canvas.height = win.innerHeight; }
    function init() {
      pts = [];
      for (var i = 0; i < N; i++) {
        pts.push({ x: Math.random() * W, y: Math.random() * H,
                   vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35 });
      }
    }
    function frame() {
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < N; i++) {
        var p = pts[i]; p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      }
      for (var i = 0; i < N; i++) {
        for (var j = i + 1; j < N; j++) {
          var a = pts[i], b = pts[j], dx = a.x - b.x, dy = a.y - b.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < MAXD) {
            ctx.strokeStyle = 'rgba(79,139,249,' + (1 - d / MAXD) * 0.20 + ')';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      ctx.fillStyle = 'rgba(130,177,255,0.55)';
      for (var i = 0; i < N; i++) {
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 1.6, 0, Math.PI * 2); ctx.fill();
      }
      win.requestAnimationFrame(frame);
    }
    resize(); init();
    win.addEventListener('resize', function () { resize(); init(); });
    win.requestAnimationFrame(frame);
  }
  startWeb();

  function openMenu() {
    return doc.querySelector('ul[role="listbox"], [role="listbox"], [data-baseweb="menu"]');
  }
  function closeMenu() {
    var el = doc.activeElement || doc.body;
    var esc = { key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true, cancelable: true };
    el.dispatchEvent(new KeyboardEvent('keydown', esc));
    doc.dispatchEvent(new KeyboardEvent('keydown', esc));
    doc.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
    doc.body.dispatchEvent(new MouseEvent('mouseup',   { bubbles: true, cancelable: true }));
  }

  doc.addEventListener('mousemove', function (e) {
    var menu = openMenu();
    if (!menu) return;                                 // ningún desplegable abierto
    var pop = menu.closest('[data-baseweb="popover"]') || menu;
    var t = e.target;
    var overPopover = pop.contains(t);
    var overControl = !!(t.closest && t.closest('[data-baseweb="select"]'));
    if (!overPopover && !overControl) closeMenu();     // fuera de ambos -> cerrar
  }, true);

  // Hide the sidebar resize/drag handle robustly (regardless of its CSS class):
  // any element inside the sidebar whose computed cursor is col-resize.
  function killResizeHandle() {
    var sb = doc.querySelector('section[data-testid="stSidebar"]');
    if (!sb) return;
    sb.querySelectorAll('div').forEach(function (d) {
      try {
        if (win.getComputedStyle(d).cursor === 'col-resize') {
          d.style.setProperty('display', 'none', 'important');
        }
      } catch (e) {}
    });
  }
  killResizeHandle();
  win.setInterval(killResizeHandle, 1500);             // catch re-renders on nav
})();
</script>
""",
        # st.iframe rejects height=0 (unlike the old components.html); use the
        # minimum positive height and let the ddac_host CSS collapse it to 0.
        height=1,
    )

# Collect a finished background benchmark (runs on every page) and clear the
# "operation in progress" flag so run/train buttons re-enable everywhere.
import src.jobs as _jobs  # noqa: E402

_fut = st.session_state.get("bench_future")
if _fut is not None and _fut.done():
    try:
        _classic_rows, _cnn_rows = _fut.result()
        if _jobs.cancel_requested():            # cancelled → discard partial work
            _classic_rows, _cnn_rows = [], []
            st.session_state["bench_cancelled"] = True
        # "Eval" = eval-only: keep just the eval rows (Split starts with "eval").
        elif st.session_state.get("bench_score") == "Eval":
            _classic_rows = [r for r in _classic_rows
                             if str(r.get("Split", "")).startswith("eval")]
            _cnn_rows = [r for r in _cnn_rows
                         if str(r.get("Split", "")).startswith("eval")]
        st.session_state.setdefault("experiment_rows", []).extend(_classic_rows)
        st.session_state.setdefault("cnn_runs", []).extend(_cnn_rows)
        if not _jobs.cancel_requested():
            st.session_state["bench_done"] = True   # enables "See full comparison"
    except Exception as _exc:  # noqa: BLE001 — surface compute errors
        st.session_state["bench_error"] = str(_exc)
    st.session_state["bench_future"] = None
    st.session_state["op_running"] = False

# Collect a finished background CNN training the same way: store the model and
# its results so the CNN Learning page shows them, exactly as if trained inline.
_cnn_fut = st.session_state.get("cnn_future")
if _cnn_fut is not None and _cnn_fut.done():
    try:
        _model, _hist, _results = _cnn_fut.result()
        if _jobs.cancel_requested():            # cancelled → keep nothing
            st.session_state["cnn_cancelled"] = True
        else:
            _pend = st.session_state.get("cnn_pending", {})
            st.session_state["cnn_model"]        = _model
            st.session_state["cnn_history"]      = _hist
            st.session_state["cnn_results"]      = _results
            st.session_state["cnn_dev"]          = _pend.get("dev", [])
            st.session_state["cnn_train_corpus"] = _pend.get("corpus", "—")
            st.session_state["cnn_arch_trained"] = _pend.get("arch", "—")
            # The leaderboard/Results honours the 3-button score: "Eval" keeps
            # only the eval row (when one was produced), the rest keep all.
            _board = _results
            if _pend.get("score") == "Eval":
                _ev = [r for r in _results if "[EVAL]" in str(r.get("Model", ""))]
                _board = _ev or _results
            st.session_state.setdefault("cnn_runs", []).extend(_board)
    except Exception as _exc:  # noqa: BLE001 — surface compute errors
        st.session_state["cnn_error"] = str(_exc)
    st.session_state["cnn_future"] = None
    st.session_state.pop("cnn_pending", None)

# Global background-job banner, pinned to the bottom of the sidebar. Rendered
# BEFORE the page runs (so it appears on every page, even ones that call
# st.stop()) and as a self-refreshing fragment (so it updates its progress every
# 2 s on its own and triggers a single full rerun when the job finishes — no
# disruptive whole-app polling).
from src.ui_helpers import (  # noqa: E402
    op_banner_fragment, op_in_progress, render_full_cta,
)

if op_in_progress():
    with st.sidebar:
        op_banner_fragment()
else:
    # Same bottom-of-sidebar spot: a one-click launch for the full comparison.
    render_full_cta()

pg = st.navigation([
    st.Page("app_pages/0_Home.py",               title="Home",               icon=":material/home:", default=True),
    st.Page("app_pages/1_Signal_Explorer.py",    title="Signal Explorer",    icon=":material/graphic_eq:", url_path="signal_explorer"),
    st.Page("app_pages/2_Benchmark.py",          title="Benchmark",          icon=":material/science:", url_path="benchmark"),
    st.Page("app_pages/3_Detection_Analysis.py", title="Detection Analysis", icon=":material/insights:", url_path="detection_analysis"),
    st.Page("app_pages/4_Methodology.py",        title="Methodology",        icon=":material/menu_book:", url_path="methodology"),
])
pg.run()

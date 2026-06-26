(function(){
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  if (!doc || doc.__ddAutoCloseV4) return;
  doc.__ddAutoCloseV4 = true;
  var win = window.parent;

  // ── Kill the default header flash on navigation ──────────────────────────────
  // Streamlit's header background is the (near-black) theme colour. The per-page
  // PAGE_CSS hides it, but that CSS is applied as a delta a beat AFTER the header
  // is drawn, so on each page change a black bar flashes over the animated
  // background. A <style> injected ONCE into the PARENT document head persists
  // across every client-side navigation, so the header is transparent (and the
  // toolbar/decoration/status gone) from the very first paint — no flash.
  if (!doc.getElementById('swChromeHide')) {
    var sc = doc.createElement('style');
    sc.id = 'swChromeHide';
    sc.textContent =
      // Make the header AND every descendant transparent (the black bar that
      // flashed on navigation was a nested element's near-black background, not
      // the header's own). Do NOT collapse the header or display:none the toolbar
      // here — the collapsed-sidebar expand control lives in that region, so
      // hiding it removed the only way to re-open the sidebar. Transparency alone
      // kills the black flash; PAGE_CSS still collapses the header to height:0.
      '[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important;}' +
      '[data-testid="stHeader"] *{background:transparent!important;}' +
      '[data-testid="stDecoration"]{display:none!important;}' +
      // Hide the Vega-Lite tooltip whenever there is no chart in the DOM.
      // display:none!important beats any inline style Vega sets, so this
      // is timing-proof: once the chart unmounts the tooltip is gone.
      'body:not(.has-vega-chart) #vg-tooltip-element{display:none!important;}';
    (doc.head || doc.documentElement).appendChild(sc);
  }

  // ── Kill the stray "black rectangle" of chart data (Model/Front-end/p(spoof)/…)
  // at the top-left after navigating. It's the Vega-Lite tooltip element, appended
  // to <body>: if you hover a bar (e.g. Detection Analysis' per-model chart) and
  // then change page client-side, the chart unmounts WITHOUT firing mouseout, so
  // Vega never hides its tooltip — it stays visible at its last spot. Hide it as
  // soon as the cursor moves anywhere that is NOT over a chart (Vega recreates the
  // element on the next genuine hover, so real tooltips keep working). Also clear
  // it on any sidebar-nav click so it never survives the page change at all.
  function hideVegaTip() {
    var tip = doc.getElementById('vg-tooltip-element');
    if (tip) { tip.classList.remove('visible'); tip.style.visibility = 'hidden';
               tip.style.opacity = '0'; }
  }
  doc.addEventListener('mousemove', function (ev) {
    var tip = doc.getElementById('vg-tooltip-element');
    if (!tip || !tip.classList.contains('visible')) return;
    var t = ev.target;
    var overChart = t && t.closest && t.closest(
      '[data-testid="stVegaLiteChart"], .vega-embed, .marks, canvas');
    if (!overChart) hideVegaTip();
  }, true);
  doc.addEventListener('click', function () { hideVegaTip(); }, true);
  // MutationObserver: when the chart is removed from the DOM on page navigation
  // (Streamlit unmounts it client-side), hide the tooltip immediately — no mouse
  // event needed. This catches keyboard navigation and programmatic page switches.
  // Toggle has-vega-chart on <body> so the CSS rule above can hide the
  // tooltip whenever no chart exists in the DOM. This is timing-proof: the
  // CSS display:none!important fires as soon as the class is absent, regardless
  // of any inline styles Vega may set. Run on every DOM mutation (debounced)
  // and on a 1 s fallback timer to cover any edge case.
  function _updVC() {
    doc.body.classList.toggle('has-vega-chart',
      !!doc.querySelector('[data-testid="stVegaLiteChart"]'));
  }
  if (win.MutationObserver) {
    var _vct;
    new win.MutationObserver(function() {
      clearTimeout(_vct); _vct = win.setTimeout(_updVC, 50);
    }).observe(doc.body, { childList: true, subtree: true });
  }
  _updVC();
  win.setInterval(_updVC, 1000);

  // ── Ambient background canvas — modes: 'starwars' (default), 'network', 'off'.
  //    The render loop reads win.__swBg / win.__swTheme / win.__reduceMotion every
  //    frame, so the Settings page can switch it live without re-init. The Konami
  //    hyperspace jump overlays any mode.
  function startWeb() {
    if (doc.getElementById('bgWeb')) return;
    var canvas = doc.createElement('canvas');
    canvas.id = 'bgWeb';
    doc.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    var W, H, pts, stars, hstars, ships, ds, shipTimer = 0, shoot = null, shootTimer = 0;
    var N = 80, MAXD = 155, NSMAX = 440, NH = 440, LX = 0, frameN = 0;

    // ── Sprites: inline SVG (crisp at any size, theme-neutral metallic hulls) ──
    var SVG_TIE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80"><defs><radialGradient id="b" cx="42%" cy="36%" r="66%"><stop offset="0" stop-color="#eef1f7"/><stop offset="55%" stop-color="#9aa3b4"/><stop offset="100%" stop-color="#3c4350"/></radialGradient><linearGradient id="w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cfd6e2"/><stop offset="1" stop-color="#6b7280"/></linearGradient></defs><g stroke="#2b303b" stroke-width="1.6" fill="url(#w)"><polygon points="16,8 36,16 36,64 16,72 8,40"/><polygon points="104,8 84,16 84,64 104,72 112,40"/></g><g stroke="#3a4150" stroke-width="1"><line x1="22" y1="12" x2="22" y2="68"/><line x1="29" y1="14" x2="29" y2="66"/><line x1="98" y1="12" x2="98" y2="68"/><line x1="91" y1="14" x2="91" y2="66"/><line x1="12" y1="40" x2="32" y2="40"/><line x1="108" y1="40" x2="88" y2="40"/></g><rect x="36" y="37" width="16" height="6" fill="#7a8290"/><rect x="68" y="37" width="16" height="6" fill="#7a8290"/><circle cx="60" cy="40" r="17" fill="url(#b)" stroke="#2b303b" stroke-width="1.6"/><circle cx="60" cy="40" r="8" fill="#222834"/><circle cx="56" cy="36" r="3" fill="#aeb6c4" opacity="0.7"/></svg>';
    var SVG_XWING = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 80"><defs><linearGradient id="f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e7ebf2"/><stop offset="1" stop-color="#7b8290"/></linearGradient></defs><g stroke="#2b303b" stroke-width="1.3" fill="#9aa3b2"><polygon points="34,40 78,28 80,33 40,42"/><polygon points="34,40 78,52 80,47 40,38"/></g><g stroke="#2b303b" stroke-width="1.1" fill="#828b9a"><polygon points="36,40 70,21 73,24 44,40"/><polygon points="36,40 70,59 73,56 44,40"/></g><g fill="#ff5a4d"><circle cx="80" cy="28" r="2.8"/><circle cx="80" cy="52" r="2.8"/><circle cx="72" cy="21" r="2.3"/><circle cx="72" cy="59" r="2.3"/></g><path d="M30 34 L120 37 Q140 40 120 43 L30 46 Q22 40 30 34 Z" fill="url(#f)" stroke="#2b303b" stroke-width="1.3"/><path d="M118 38 L136 40 L118 42 Z" fill="#dfe5ee" stroke="#2b303b" stroke-width="0.8"/><path d="M92 34 q9 -3 15 2 l-13 3 Z" fill="#27303f"/><circle cx="84" cy="36" r="3.2" fill="#3a4658" stroke="#2b303b" stroke-width="0.6"/></svg>';
    var SVG_DEST = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 70"><defs><linearGradient id="h" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d3dae6"/><stop offset="1" stop-color="#565d6a"/></linearGradient></defs><polygon points="170,40 22,22 22,54" fill="url(#h)" stroke="#2b303b" stroke-width="1.4"/><g stroke="#39414f" stroke-width="0.8" opacity="0.8"><line x1="62" y1="28" x2="62" y2="52"/><line x1="98" y1="31" x2="98" y2="49"/><line x1="130" y1="34" x2="130" y2="46"/><line x1="22" y1="38" x2="170" y2="40"/></g><rect x="34" y="14" width="22" height="9" fill="#9aa3b2" stroke="#2b303b" stroke-width="0.8"/><rect x="40" y="7" width="10" height="8" fill="#aeb6c4" stroke="#2b303b" stroke-width="0.6"/><rect x="18" y="24" width="5" height="28" fill="#39414f"/></svg>';
    var SVG_FALCON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 130 90"><defs><radialGradient id="fg" cx="42%" cy="38%" r="66%"><stop offset="0" stop-color="#d9dee8"/><stop offset="60%" stop-color="#9aa2b0"/><stop offset="100%" stop-color="#5a616e"/></radialGradient></defs><path d="M86 34 L122 30 L122 36 L92 42 Z" fill="#7b828f" stroke="#2b303b" stroke-width="1.2"/><path d="M86 56 L122 60 L122 54 L92 48 Z" fill="#7b828f" stroke="#2b303b" stroke-width="1.2"/><ellipse cx="55" cy="45" rx="52" ry="40" fill="url(#fg)" stroke="#2b303b" stroke-width="1.4"/><circle cx="50" cy="45" r="13" fill="none" stroke="#39414f" stroke-width="1"/><circle cx="50" cy="45" r="5" fill="#39414f"/><g stroke="#39414f" stroke-width="0.7" opacity="0.7"><line x1="50" y1="45" x2="14" y2="30"/><line x1="50" y1="45" x2="14" y2="60"/><line x1="50" y1="45" x2="40" y2="8"/><line x1="50" y1="45" x2="40" y2="82"/><line x1="50" y1="45" x2="86" y2="22"/><line x1="50" y1="45" x2="86" y2="68"/></g><ellipse cx="92" cy="70" rx="11" ry="6" fill="#aeb6c4" stroke="#2b303b" stroke-width="1" transform="rotate(20 92 70)"/></svg>';
    var SVG_DS = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><defs><radialGradient id="d" cx="38%" cy="33%" r="74%"><stop offset="0" stop-color="#c8cedb"/><stop offset="55%" stop-color="#8a92a2"/><stop offset="100%" stop-color="#262d3a"/></radialGradient><radialGradient id="ss" cx="70%" cy="70%" r="75%"><stop offset="55%" stop-color="rgba(8,12,20,0)"/><stop offset="100%" stop-color="rgba(8,12,20,0.85)"/></radialGradient><radialGradient id="di" cx="50%" cy="42%" r="60%"><stop offset="0" stop-color="#3a4252"/><stop offset="70%" stop-color="#5a6273"/><stop offset="100%" stop-color="#7c8494"/></radialGradient></defs><circle cx="60" cy="60" r="56" fill="url(#d)"/><g stroke="#2c3340" stroke-width="0.7" fill="none" opacity="0.75"><path d="M8 44 H112"/><path d="M6 76 H114"/><path d="M14 92 H106"/><path d="M20 104 H100"/></g><path d="M5 57 H115" stroke="#161c28" stroke-width="3.2"/><path d="M6 63 H114" stroke="#2a3340" stroke-width="1.2"/><circle cx="43" cy="40" r="15" fill="url(#di)" stroke="#1d2430" stroke-width="1"/><circle cx="43" cy="40" r="9" fill="none" stroke="#1d2430" stroke-width="0.8"/><circle cx="43" cy="40" r="2.6" fill="#aeb6c4"/><circle cx="82" cy="74" r="3.2" fill="#2c3340"/><circle cx="68" cy="92" r="2.1" fill="#2c3340"/><circle cx="90" cy="52" r="1.8" fill="#2c3340"/><circle cx="60" cy="60" r="56" fill="url(#ss)"/><circle cx="60" cy="60" r="56" fill="none" stroke="#9fb0cc" stroke-width="0.8" opacity="0.5"/></svg>';
    function svgImg(s) { var i = new Image(); i.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(s); return i; }
    var SPR = { tie: svgImg(SVG_TIE), xwing: svgImg(SVG_XWING), destroyer: svgImg(SVG_DEST), falcon: svgImg(SVG_FALCON) };
    var DSIMG = svgImg(SVG_DS);

    function rnd(a, b) { return a + Math.random() * (b - a); }
    function light() { return win.__swTheme === 'light'; }
    function still() { return !!win.__reduceMotion; }
    function intens() { var m = win.__swIntensity; return m === 'Subtle' ? 0.5 : (m === 'Busy' ? 1.7 : 1); }
    // Retina-crisp: draw in CSS pixels, back the canvas at devicePixelRatio.
    function resize() {
      var dpr = Math.min(win.devicePixelRatio || 1, 2);
      W = win.innerWidth; H = win.innerHeight;
      canvas.width = Math.floor(W * dpr); canvas.height = Math.floor(H * dpr);
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      measureSidebar();
    }
    // Left wall of the visible canvas = the sidebar's right edge. Everything is
    // clipped/bounced to the right of it so ships and the Death Star are never
    // hidden behind the opaque sidebar. Re-measured periodically (the sidebar can
    // be collapsed on touch devices without firing a resize event).
    function measureSidebar() {
      try {
        var sb = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sb) { LX = 0; return; }
        var r = sb.getBoundingClientRect();
        var x = (r.right > 0 && r.left < 4) ? r.right : 0;   // 0 when off-canvas
        LX = Math.max(0, Math.min(x, W * 0.5));
      } catch (e) { LX = 0; }
    }
    function init() {
      pts = [];
      for (var i = 0; i < N; i++)
        pts.push({ x: Math.random() * W, y: Math.random() * H,
                   vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35 });
      stars = [];
      for (var s = 0; s < NSMAX; s++)
        stars.push({ x: Math.random() * W, y: Math.random() * H, r: Math.random() * 1.4 + 0.2,
                     tw: Math.random() * Math.PI * 2, sp: rnd(0.004, 0.03), dx: rnd(-0.05, 0.05),
                     hue: Math.random() });
      hstars = [];
      for (var k = 0; k < NH; k++)
        hstars.push({ x: (Math.random() - 0.5) * W, y: (Math.random() - 0.5) * H,
                      z: Math.random() * W, sp: 6 + Math.random() * 14 });
      ds = { x: W * 0.82, y: H * 0.2, r: Math.min(W, H) * 0.09, vx: -0.05, vy: 0.015 };
      ships = [];
      shipTimer = Date.now() + rnd(2500, 6000);
      shootTimer = Date.now() + rnd(3000, 9000);
    }

    // — Hyperspace (Konami) —
    function drawHyper() {
      ctx.fillStyle = 'rgba(2,4,12,0.32)'; ctx.fillRect(0, 0, W, H);
      var cx = (LX + W) / 2, cy = H / 2;
      for (var s = 0; s < hstars.length; s++) {
        var p = hstars[s], oz = p.z; p.z -= p.sp;
        if (p.z < 1) { p.x = (Math.random() - 0.5) * W; p.y = (Math.random() - 0.5) * H; p.z = W; oz = W; }
        var sx = cx + p.x / p.z * W, sy = cy + p.y / p.z * H;
        var ox = cx + p.x / oz * W,  oy = cy + p.y / oz * H;
        var br = Math.min(1, (W - p.z) / W + 0.15);
        ctx.strokeStyle = 'rgba(160,200,255,' + br + ')';
        ctx.lineWidth = Math.max(0.5, (W - p.z) / W * 2.4);
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(sx, sy); ctx.stroke();
      }
    }

    // — Particle network (legacy option) —
    function drawNetwork() {
      for (var i = 0; i < N; i++) {
        var p = pts[i]; if (!still()) { p.x += p.vx; p.y += p.vy; }
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      }
      var lc = light() ? '40,70,140' : '79,139,249';
      for (var i = 0; i < N; i++)
        for (var j = i + 1; j < N; j++) {
          var a = pts[i], b = pts[j], dx = a.x - b.x, dy = a.y - b.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < MAXD) {
            ctx.strokeStyle = 'rgba(' + lc + ',' + (1 - d / MAXD) * (light() ? 0.16 : 0.20) + ')';
            ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      ctx.fillStyle = light() ? 'rgba(60,90,160,0.5)' : 'rgba(130,177,255,0.55)';
      for (var i = 0; i < N; i++) { ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 1.6, 0, Math.PI * 2); ctx.fill(); }
    }

    // — Star Wars ambient (default) —
    function drawStars() {
      var count = Math.min(stars.length, Math.floor(220 * intens()));
      for (var i = 0; i < count; i++) {
        var s = stars[i];
        if (!still()) { s.tw += s.sp; s.x += s.dx; if (s.x < 0) s.x += W; if (s.x > W) s.x -= W; }
        var a = 0.3 + 0.5 * Math.abs(Math.sin(s.tw));
        // a few stars get a faint blue/amber tint, the rest are white
        var col = s.hue > 0.9 ? '180,205,255' : (s.hue < 0.08 ? '255,225,190' : '224,234,255');
        ctx.fillStyle = 'rgba(' + col + ',' + a * 0.9 + ')';
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
        if (s.r > 1.1) {   // soft glow on the brightest stars
          ctx.fillStyle = 'rgba(' + col + ',' + a * 0.18 + ')';
          ctx.beginPath(); ctx.arc(s.x, s.y, s.r * 2.4, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
    function drawShooting() {
      if (!shoot && !still() && Date.now() > shootTimer)
        shoot = { x: rnd(W * 0.05, W * 0.6), y: rnd(0, H * 0.35), vx: rnd(7, 11), vy: rnd(2.5, 4.5), life: 1 };
      if (!shoot) return;
      if (!still()) { shoot.x += shoot.vx; shoot.y += shoot.vy; shoot.life -= 0.018; }
      var L = 70, tx = shoot.x - L, ty = shoot.y - L * shoot.vy / shoot.vx;
      var g = ctx.createLinearGradient(shoot.x, shoot.y, tx, ty);
      var a = Math.max(0, shoot.life);
      g.addColorStop(0, 'rgba(205,228,255,' + a * 0.95 + ')'); g.addColorStop(1, 'rgba(205,228,255,0)');
      ctx.strokeStyle = g; ctx.lineWidth = 2; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(shoot.x, shoot.y); ctx.lineTo(tx, ty); ctx.stroke(); ctx.lineCap = 'butt';
      if (shoot.life <= 0 || shoot.x > W || shoot.y > H) { shoot = null; shootTimer = Date.now() + rnd(7000, 16000); }
    }
    function drawDeathStar() {
      if (!still()) {
        // Gentle wandering drift: occasionally pick a new heading, bounce off the
        // edges of the visible box (always to the right of the sidebar).
        if (Math.random() < 0.008) { ds.vx = rnd(-0.16, 0.16); ds.vy = rnd(-0.05, 0.05); }
        ds.x += ds.vx; ds.y += ds.vy;
        var lb = LX + ds.r + 10, rb = W - ds.r - 10;
        if (lb < rb) {
          if (ds.x < lb) { ds.x = lb; ds.vx =  Math.abs(ds.vx); }
          if (ds.x > rb) { ds.x = rb; ds.vx = -Math.abs(ds.vx); }
        }
        if (ds.y < H * 0.08) { ds.y = H * 0.08; ds.vy =  Math.abs(ds.vy); }
        if (ds.y > H * 0.40) { ds.y = H * 0.40; ds.vy = -Math.abs(ds.vy); }
      }
      if (!DSIMG.complete || !DSIMG.naturalWidth) return;
      var d = ds.r * 2;
      ctx.save(); ctx.globalAlpha = 0.95;
      ctx.drawImage(DSIMG, ds.x - ds.r, ds.y - ds.r, d, d);
      ctx.restore();
    }
    // Tatooine binary sunset for the Light Side (a space scene reads wrong on
    // white). Warm sky wash + two saturated sun discs so they show on light bg.
    function drawTwinSuns() {
      var m = Math.min(W, H), t = Date.now() * 0.00002;
      var sky = ctx.createLinearGradient(0, 0, 0, H * 0.7);
      sky.addColorStop(0, 'rgba(255,196,128,0.18)');
      sky.addColorStop(1, 'rgba(255,196,128,0)');
      ctx.fillStyle = sky; ctx.fillRect(0, 0, W, H * 0.7);
      function sun(cx, cy, rad, core, mid, edge) {
        var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
        g.addColorStop(0, core); g.addColorStop(0.22, mid); g.addColorStop(1, edge);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.fill();
      }
      var s1x = W * 0.8 + Math.sin(t) * W * 0.01, s1y = H * 0.2 + Math.cos(t) * H * 0.008;
      var s2x = W * 0.66 + Math.cos(t * 1.2) * W * 0.01, s2y = H * 0.31;
      sun(s1x, s1y, m * 0.16, 'rgba(255,236,170,0.98)', 'rgba(255,210,130,0.55)', 'rgba(255,205,130,0)');
      sun(s2x, s2y, m * 0.11, 'rgba(255,178,96,0.92)',  'rgba(255,150,80,0.5)',   'rgba(255,150,80,0)');
    }
    // Ships fly a WANDERING route, not a straight line: a steady horizontal
    // velocity plus a sine sway and a slowly-changing vertical drift. They bank
    // (tilt) into the direction they are actually moving for a realistic feel.
    function spawnShip() {
      var rt = Math.random();
      var type = rt < 0.4 ? 'tie' : (rt < 0.72 ? 'xwing' : (rt < 0.9 ? 'falcon' : 'destroyer'));
      var big = type === 'destroyer', fromLeft = Math.random() < 0.5, dir = fromLeft ? 1 : -1;
      var sp = big ? rnd(0.4, 0.8) : rnd(1.3, 2.6);
      ships.push({ type: type, dir: dir, vx: dir * sp,
                   x: fromLeft ? LX - 60 : W + 240,
                   baseY: rnd(H * 0.12, big ? H * 0.4 : H * 0.82), y: null,
                   vy: rnd(-0.3, 0.3) * (big ? 0.4 : 1),
                   amp: big ? rnd(2, 6) : rnd(10, 30), freq: rnd(0.004, 0.011),
                   phase: rnd(0, Math.PI * 2), t: rnd(0, 1000),
                   w: big ? rnd(170, 250) : rnd(48, 84) });
    }
    // The Episode IV opening shot: a small ship fleeing, a Star Destroyer looming
    // behind it (same heading). A quiet recurring easter egg.
    function spawnChase() {
      var fromLeft = Math.random() < 0.5, dir = fromLeft ? 1 : -1;
      var y = rnd(H * 0.18, H * 0.5), lead = fromLeft ? LX - 80 : W + 120;
      ships.push({ type: Math.random() < 0.5 ? 'falcon' : 'xwing', dir: dir, vx: dir * rnd(2.2, 2.8),
                   x: lead, baseY: y, y: null, vy: rnd(-0.25, 0.25),
                   amp: rnd(10, 22), freq: rnd(0.006, 0.012), phase: rnd(0, Math.PI * 2),
                   t: rnd(0, 1000), w: rnd(54, 74) });
      ships.push({ type: 'destroyer', dir: dir, vx: dir * rnd(1.7, 2.1),
                   x: lead - dir * rnd(240, 320), baseY: y + rnd(-12, 24), y: null,
                   vy: rnd(-0.12, 0.12), amp: rnd(2, 5), freq: rnd(0.004, 0.008),
                   phase: rnd(0, Math.PI * 2), t: rnd(0, 1000), w: rnd(200, 250) });
    }
    function drawShip(sh) {
      if (!still()) {
        sh.t += 1; sh.x += sh.vx; sh.baseY += sh.vy;
        if (Math.random() < 0.012) sh.vy = rnd(-0.45, 0.45) * (sh.type === 'destroyer' ? 0.4 : 1);
        var topB = H * 0.07, botB = sh.type === 'destroyer' ? H * 0.42 : H * 0.86;
        if (sh.baseY < topB) sh.vy =  Math.abs(sh.vy);
        if (sh.baseY > botB) sh.vy = -Math.abs(sh.vy);
      }
      var img = SPR[sh.type];
      if (!img || !img.complete || !img.naturalWidth) return;
      var prevY = (sh.y === null || sh.y === undefined) ? sh.baseY : sh.y;
      sh.y = sh.baseY + (still() ? 0 : Math.sin(sh.t * sh.freq + sh.phase) * sh.amp);
      var dy = sh.y - prevY;
      var w = sh.w, h = w * img.naturalHeight / img.naturalWidth;
      var bank = Math.atan2(dy, Math.abs(sh.vx) + 0.001) * sh.dir;   // tilt into travel
      bank = Math.max(-0.32, Math.min(0.32, bank));
      ctx.save();
      ctx.translate(sh.x, sh.y);
      ctx.scale(sh.dir, 1);
      if (!still()) ctx.rotate(bank);
      ctx.globalAlpha = light() ? 0.88 : 0.96;
      if (sh.type !== 'tie') {           // sub-light engine glow at the rear
        var gx = -w * 0.46, gr = w * (sh.type === 'destroyer' ? 0.1 : 0.16);
        var eg = ctx.createRadialGradient(gx, 0, 0, gx, 0, gr);
        eg.addColorStop(0, 'rgba(150,210,255,0.7)'); eg.addColorStop(1, 'rgba(150,210,255,0)');
        ctx.fillStyle = eg; ctx.beginPath(); ctx.arc(gx, 0, gr, 0, Math.PI * 2); ctx.fill();
      }
      ctx.drawImage(img, -w / 2, -h / 2, w, h);
      ctx.restore();
    }

    function frame() {
      var mode = win.__swBg || 'starwars';
      if ((frameN++ % 20) === 0) measureSidebar();
      ctx.clearRect(0, 0, W, H);
      if (win.__hyperUntil && Date.now() < win.__hyperUntil) { drawHyper(); win.requestAnimationFrame(frame); return; }
      if (mode === 'off') { win.requestAnimationFrame(frame); return; }
      // Clip everything to the right of the sidebar so nothing hides behind it.
      ctx.save();
      ctx.beginPath(); ctx.rect(LX, 0, Math.max(0, W - LX), H); ctx.clip();
      if (mode === 'network') { drawNetwork(); ctx.restore(); win.requestAnimationFrame(frame); return; }
      if (light()) {
        drawTwinSuns();
      } else {
        drawStars();
        drawShooting();
        if (win.__swDeathStar !== 0) drawDeathStar();
      }
      if (win.__swShips !== 0) {
        if (!still() && Date.now() > shipTimer) {
          if (Math.random() < 0.16) spawnChase(); else spawnShip();
          shipTimer = Date.now() + rnd(7000, 17000) / intens();
        }
        for (var i = ships.length - 1; i >= 0; i--) {
          drawShip(ships[i]);
          if (ships[i].x < LX - 360 || ships[i].x > W + 360) ships.splice(i, 1);
        }
      }
      ctx.restore();
      win.requestAnimationFrame(frame);
    }
    resize(); init();
    win.addEventListener('resize', function () { resize(); init(); });
    win.requestAnimationFrame(frame);
  }
  startWeb();

  // ── Konami code → 3.5 s hyperspace jump on the background canvas ───────────
  var KON = [38,38,40,40,37,39,37,39,66,65], kpos = 0;
  doc.addEventListener('keydown', function (e) {
    kpos = (e.keyCode === KON[kpos]) ? kpos + 1 : (e.keyCode === KON[0] ? 1 : 0);
    if (kpos === KON.length) { kpos = 0; win.__hyperUntil = Date.now() + 3500; }
  });

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

// Timeline bar: every recorded event of the current timeline on the cycle axis.
// Height = significance; colour = epistemic status. Hover for the title, click to inspect.

const EPI = {
  SIMULATED: "#9fb3ff", PROGRAMMED: "#8c8fa6", EMERGENT: "#7dffb0", MEASURED: "#7fd8ff", INFERRED: "#ffd27f",
  HYPOTHESIS: "#ffb36b", EXTERNAL: "#ff6b8a",
};

export class TimelineBar {
  constructor(canvas, tip, onPick) {
    this.c = canvas;
    this.tip = tip;
    this.onPick = onPick;
    this.events = [];
    this.now = 1;
    this.hover = null;
    canvas.addEventListener("mousemove", (e) => this._hover(e));
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.classList.add("hidden"); this.draw(); });
    canvas.addEventListener("click", (e) => {
      const ev = this._at(e);
      if (ev) this.onPick(ev);
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(events, now) {
    this.events = events;
    this.now = Math.max(1, now || 1);
    this.draw();
  }

  _x(utick, w) {
    return 12 + (utick / this.now) * (w - 24);
  }

  _at(e) {
    const r = this.c.getBoundingClientRect();
    const x = e.clientX - r.left;
    let best = null, bd = 6;
    for (const ev of this.events) {
      const d = Math.abs(this._x(ev.utick, r.width) - x) - ev.significance;
      if (d < bd) { bd = d; best = ev; }
    }
    return best;
  }

  _hover(e) {
    const ev = this._at(e);
    this.hover = ev;
    if (ev) {
      this.tip.innerHTML = `<b>${ev.label || "E-" + ev.id}</b> · cycle ${ev.utick}<br>${ev.title.replace(/</g, "&lt;")}`;
      this.tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 330)}px`;
      this.tip.style.top = `${e.clientY - 64}px`;
      this.tip.classList.remove("hidden");
    } else {
      this.tip.classList.add("hidden");
    }
    this.draw();
  }

  draw() {
    const c = this.c, dpr = Math.min(devicePixelRatio || 1, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = "rgba(140,160,255,0.25)";
    g.beginPath();
    g.moveTo(12, h - 14);
    g.lineTo(w - 12, h - 14);
    g.stroke();
    g.fillStyle = "rgba(138,147,184,0.8)";
    g.font = "10px ui-monospace, monospace";
    g.fillText("0", 10, h - 2);
    const label = `cycle ${this.now.toLocaleString("en-US")}`;
    g.fillText(label, w - 12 - g.measureText(label).width, h - 2);
    for (const ev of this.events) {
      const x = this._x(ev.utick, w);
      const len = 6 + ev.significance * 9;
      g.strokeStyle = EPI[ev.epistemic] || "#9fb3ff";
      g.globalAlpha = ev.significance === 0 ? 0.35 : 0.9;
      g.lineWidth = ev === this.hover ? 3 : ev.significance >= 2 ? 2 : 1;
      g.beginPath();
      g.moveTo(x, h - 14);
      g.lineTo(x, h - 14 - len);
      g.stroke();
      if (ev.significance >= 3) {
        g.fillStyle = "#b38cff";
        g.beginPath();
        g.arc(x, h - 16 - len, 3, 0, Math.PI * 2);
        g.fill();
      }
    }
    g.globalAlpha = 1;
  }
}

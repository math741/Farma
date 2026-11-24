// Sonification (not speech): a drone whose pitch follows the planet's global temperature, whose
// brightness follows its population, and soft tones when organisms emit vibration.
// Every parameter is read from the live state; nothing is a recording.

export class Sonifier {
  constructor() {
    this.ctx = null;
    this.on = false;
  }

  start() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      this.ctx = new AC();
      const c = this.ctx;
      this.master = c.createGain();
      this.master.gain.value = 0.0;
      this.master.connect(c.destination);
      this.filter = c.createBiquadFilter();
      this.filter.type = "lowpass";
      this.filter.frequency.value = 400;
      this.filter.connect(this.master);
      this.oscs = [0, 7.02, 12].map((semi) => {
        const o = c.createOscillator();
        o.type = "sine";
        const g = c.createGain();
        g.gain.value = semi === 0 ? 0.5 : 0.22;
        o.connect(g).connect(this.filter);
        o.start();
        return { o, semi };
      });
      this.chime = c.createGain();
      this.chime.gain.value = 0.0;
      this.chime.connect(this.master);
    }
    this.ctx.resume();
    this.master.gain.setTargetAtTime(0.16, this.ctx.currentTime, 1.2);
    this.on = true;
    return true;
  }

  stop() {
    if (this.ctx) this.master.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.4);
    this.on = false;
  }

  /** world: summary from the status stream; emission: mean emission of the visible organisms (0..1). */
  update(world, emission) {
    if (!this.on || !this.ctx || !world) return;
    const t = this.ctx.currentTime;
    const temp = Number.isFinite(world.global_temp) ? world.global_temp : 10;
    const base = 55 * Math.pow(2, Math.max(-1, Math.min(1.5, (temp + 10) / 40)));
    for (const { o, semi } of this.oscs) o.frequency.setTargetAtTime(base * Math.pow(2, semi / 12), t, 2.0);
    const pop = world.population / Math.max(1, world.slots);
    this.filter.frequency.setTargetAtTime(220 + 2400 * Math.sqrt(pop), t, 1.5);
    if (emission > 0.02 && Math.random() < Math.min(0.9, emission * 6)) this._tone(base * 8 * (1 + Math.floor(Math.random() * 4) / 4));
  }

  _tone(freq) {
    const c = this.ctx;
    const o = c.createOscillator(), g = c.createGain();
    o.type = "triangle";
    o.frequency.value = freq;
    g.gain.value = 0.0;
    g.gain.setTargetAtTime(0.05, c.currentTime, 0.01);
    g.gain.setTargetAtTime(0.0, c.currentTime + 0.08, 0.25);
    o.connect(g).connect(this.master);
    o.start();
    o.stop(c.currentTime + 1.5);
  }
}

// Observatory ↔ backend: token-authenticated JSON, the GOB1 binary container, and the live WebSocket.
//
// The token arrives once in the URL (?token=…), is kept in sessionStorage for reloads and removed
// from the address bar so it does not end up in screenshots or screen shares.

const params = new URLSearchParams(location.search);
let TOKEN = params.get("token") || "";
try {
  if (TOKEN) sessionStorage.setItem("genesis.token", TOKEN);
  else TOKEN = sessionStorage.getItem("genesis.token") || "";
} catch (e) { /* storage unavailable (private mode): the URL token still works for this page */ }
if (params.has("token")) {
  params.delete("token");
  const q = params.toString();
  history.replaceState(null, "", location.pathname + (q ? "?" + q : "") + location.hash);
}

export const hasToken = () => !!TOKEN;

export class ApiError extends Error {
  constructor(status, body) {
    let msg = body;
    try { msg = JSON.parse(body).error || body; } catch (e) { /* plain text */ }
    super(`${status}: ${msg}`);
    this.status = status;
  }
}

function url(path, query) {
  const u = new URL(path, location.origin);
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, v);
  }
  u.searchParams.set("token", TOKEN);
  return u.toString();
}

export async function api(path, query) {
  const r = await fetch(url(path, query), { cache: "no-store" });
  if (!r.ok) throw new ApiError(r.status, await r.text());
  return r.json();
}

export async function post(path, body) {
  const r = await fetch(url(path), {
    method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new ApiError(r.status, await r.text());
  return r.json();
}

export async function bin(path, query) {
  const r = await fetch(url(path, query), { cache: "no-store" });
  if (!r.ok) throw new ApiError(r.status, await r.text());
  return decode(await r.arrayBuffer());
}

// --------------------------------------------------------------- GOB1
const DTYPES = {
  float32: Float32Array, int32: Int32Array, uint32: Uint32Array, int16: Int16Array,
  uint16: Uint16Array, uint8: Uint8Array, int8: Int8Array,
};

/** Decode b"GOB1" | u32 header length | JSON header | raw 4-byte-aligned arrays → typed-array views. */
export function decode(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 71 || u8[1] !== 79 || u8[2] !== 66 || u8[3] !== 49) throw new Error("not a GOB1 message");
  const hl = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  const base = 8 + hl;
  const arrays = {};
  for (const s of header.arrays) {
    const T = DTYPES[s.dtype];
    if (!T) throw new Error(`unsupported dtype ${s.dtype}`);
    const a = new T(buf, base + s.offset, s.nbytes / T.BYTES_PER_ELEMENT);
    a.shape = s.shape;
    arrays[s.name] = a;
  }
  return { kind: header.kind, meta: header.meta || {}, arrays };
}

// --------------------------------------------------------------- live stream
export class Stream {
  constructor(handlers) {
    this.h = handlers;
    this.sub = { type: "subscribe", world: 0, timeline: "PRIME", roi: null, max_agents: 3000, rate: 10 };
    this.ws = null;
    this.backoff = 600;
    this.closed = false;
    this.connected = false;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(TOKEN)}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.connected = true;
      this.backoff = 600;
      ws.send(JSON.stringify(this.sub));
      this.h.open && this.h.open();
    };
    ws.onmessage = (ev) => {
      try {
        if (typeof ev.data === "string") this.h.json && this.h.json(JSON.parse(ev.data));
        else this.h.binary && this.h.binary(decode(ev.data));
      } catch (e) {
        console.warn("stream message dropped:", e);
      }
    };
    ws.onclose = () => {
      const was = this.connected;
      this.connected = false;
      if (this.ws === ws) this.ws = null;
      this.h.close && this.h.close(was);
      if (!this.closed) {
        setTimeout(() => this.connect(), this.backoff);
        this.backoff = Math.min(this.backoff * 1.7, 10000);
      }
    };
    ws.onerror = () => { /* onclose follows */ };
    this.ws = ws;
  }

  subscribe(patch) {
    Object.assign(this.sub, patch);
    this.send(this.sub);
  }

  send(msg) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }
}

type Listener = (data: unknown) => void;

class WsManager {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<Listener>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.ws?.send(
          JSON.stringify({
            action: "subscribe",
            topics: ["alerts", "progress", "prices"],
          })
        );
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const topic = data.type || "message";
          this.listeners.get(topic)?.forEach((fn) => fn(data));
          this.listeners.get("*")?.forEach((fn) => fn(data));
        } catch {
          // ignore parse errors
        }
      };

      this.ws.onclose = () => {
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 5000);
  }

  on(topic: string, fn: Listener) {
    if (!this.listeners.has(topic)) this.listeners.set(topic, new Set());
    this.listeners.get(topic)!.add(fn);
    return () => {
      this.listeners.get(topic)?.delete(fn);
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}

export const wsManager = new WsManager();

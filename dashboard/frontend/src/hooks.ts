import { useCallback, useEffect, useRef, useState } from "react";

/** Poll an async function on an interval. Pauses when `enabled` is false. */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  enabled = true
): { data: T | null; error: string | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const refresh = useCallback(async () => {
    try {
      const result = await fnRef.current();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    const tick = async () => {
      if (!active) return;
      await refresh();
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [enabled, intervalMs, refresh]);

  return { data, error, refresh };
}

export type SSEStatus = "idle" | "open" | "done" | "error";

/**
 * Subscribe to a Server-Sent Events endpoint with named event handlers.
 * Re-subscribes whenever `url` changes or `restartKey` is bumped.
 */
export function useSSE(
  url: string | null,
  handlers: Record<string, (data: unknown) => void>,
  restartKey: number = 0
): SSEStatus {
  const [status, setStatus] = useState<SSEStatus>("idle");
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    if (!url) {
      setStatus("idle");
      return;
    }
    const source = new EventSource(url);
    setStatus("open");

    const registered: [string, EventListener][] = [];
    for (const eventName of Object.keys(handlersRef.current)) {
      const listener: EventListener = (event) => {
        const messageEvent = event as MessageEvent;
        let parsed: unknown = messageEvent.data;
        try {
          parsed = JSON.parse(messageEvent.data);
        } catch {
          // leave as string
        }
        handlersRef.current[eventName]?.(parsed);
        if (eventName === "done") {
          setStatus("done");
          source.close();
        }
      };
      source.addEventListener(eventName, listener);
      registered.push([eventName, listener]);
    }

    source.onerror = () => {
      // EventSource auto-reconnects; surface a transient error state.
      setStatus((prev) => (prev === "done" ? prev : "error"));
    };

    return () => {
      for (const [name, listener] of registered) {
        source.removeEventListener(name, listener);
      }
      source.close();
    };
  }, [url, restartKey]);

  return status;
}

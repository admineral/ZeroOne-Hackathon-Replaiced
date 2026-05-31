import { useEffect, useRef, useState } from "react";
import { useSSE } from "../hooks";

interface Props {
  run: string;
  liveKey: number;
  live: boolean;
}

export function LogDrawer({ run, liveKey, live }: Props) {
  const [which, setWhich] = useState<"out" | "err">("out");
  const [text, setText] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setText("");
  }, [run, which, liveKey]);

  const streamUrl = live ? `/api/logs/stream?run=${run}&which=${which}` : null;
  useSSE(
    streamUrl,
    {
      log: (data) => {
        const payload = data as { chunk: string };
        setText((prev) => prev + payload.chunk);
      },
      reset: () => setText(""),
      done: () => undefined,
    },
    liveKey
  );

  useEffect(() => {
    if (boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="card">
      <div className="card-head">
        <h3>Live job logs</h3>
        <div className="log-tabs">
          <button
            className={`log-tab ${which === "out" ? "active" : ""}`}
            onClick={() => setWhich("out")}
          >
            stdout
          </button>
          <button
            className={`log-tab ${which === "err" ? "active" : ""}`}
            onClick={() => setWhich("err")}
          >
            stderr
          </button>
        </div>
      </div>
      <div className="logbox" ref={boxRef}>
        {text || (live ? "Waiting for output…" : "Start a run to stream logs here.")}
      </div>
    </div>
  );
}

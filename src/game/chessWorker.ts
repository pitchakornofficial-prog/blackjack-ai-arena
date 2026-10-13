/**
 * Chess Deep Exploration Web Worker
 * Runs board computation on a separate OS thread, keeping the main thread
 * free to handle GPU dispatch and React state updates concurrently.
 *
 * Message protocol:
 *   IN:  { type: "step_batch", state: DeepExploreState, count: number }
 *   OUT: { type: "batch_done", state: DeepExploreState, positionsThisBatch: number }
 *   IN:  { type: "ping" }
 *   OUT: { type: "pong" }
 */

import { stepDeepExplore, DeepExploreState } from "./chessTraining";

// Web Worker context
const ctx: Worker = self as unknown as Worker;

ctx.onmessage = (event: MessageEvent) => {
  const msg = event.data;

  if (msg.type === "ping") {
    ctx.postMessage({ type: "pong" });
    return;
  }

  if (msg.type === "step_batch") {
    const { state, count } = msg as { type: "step_batch"; state: DeepExploreState; count: number };
    const positionsBefore = state.totalPositionsExplored;
    let curr: DeepExploreState = state;

    // Run N chess moves synchronously on this worker thread
    for (let i = 0; i < count; i++) {
      curr = stepDeepExplore(curr);
    }

    const positionsThisBatch = curr.totalPositionsExplored - positionsBefore;

    ctx.postMessage({
      type: "batch_done",
      state: curr,
      positionsThisBatch
    });
    return;
  }
};

export {}; // Mark as ESModule so TypeScript doesn't complain about global scope

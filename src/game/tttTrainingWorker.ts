import { TTTAlgorithm, TTTBatchResult, TTTQTable, trainTTTBatch } from "./tttTraining";

const ctx: Worker = self as unknown as Worker;

ctx.onmessage = (event: MessageEvent) => {
  const request = event.data as {
    type: "train_batch";
    qTable: TTTQTable;
    batchGames: number;
    learningRate: number;
    discountFactor: number;
    epsilon: number;
    algorithm: TTTAlgorithm;
  };
  if (request.type !== "train_batch") return;

  const result: TTTBatchResult = trainTTTBatch(
    request.qTable,
    request.batchGames,
    request.learningRate,
    request.discountFactor,
    request.epsilon,
    request.algorithm
  );
  ctx.postMessage({ type: "training_batch_done", result });
};

export {};

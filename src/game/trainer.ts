import { Card, createShoe, scoreHand, Action, basicStrategy } from "./engine";

export type QTable = Record<string, Record<"hit" | "stand" | "double", number>>;

export type BatchResult = {
  episodes: number;
  wins: number;
  losses: number;
  pushes: number;
  winRate: number;
  lossRate: number;
  pushRate: number;
  avgReward: number;
  epsilon: number;
};

export type MetricPoint = {
  step: number;
  timeSeconds: number;
  totalEpisodes: number;
  winRate: number;
  lossRate: number;
  pushRate: number;
  avgReward: number;
  epsilon: number;
};

export function createQTable(): QTable {
  return {};
}

export function encodeState(playerTotal: number, dealerUpcard: number, isSoft: boolean): string {
  return `${playerTotal}_${dealerUpcard}_${isSoft ? 1 : 0}`;
}

export function getQValues(qTable: QTable, stateKey: string): Record<"hit" | "stand" | "double", number> {
  if (!qTable[stateKey]) {
    qTable[stateKey] = { hit: 0, stand: 0, double: 0 };
  }
  return qTable[stateKey];
}

export function getBestAction(qTable: QTable, stateKey: string, canDouble = false): "hit" | "stand" | "double" {
  const q = getQValues(qTable, stateKey);
  const actions: ("hit" | "stand" | "double")[] = canDouble ? ["hit", "stand", "double"] : ["hit", "stand"];
  let best = actions[0];
  let maxVal = q[best];
  for (const a of actions) {
    if (q[a] > maxVal) {
      maxVal = q[a];
      best = a;
    }
  }
  return best;
}

export function trainBatch(
  qTable: QTable,
  batchSize: number,
  alpha: number,
  gamma: number,
  epsilon: number,
  algorithm: "qlearning" | "dqn" | "baseline" = "qlearning"
): { result: BatchResult; updatedEpsilon: number } {
  let wins = 0;
  let losses = 0;
  let pushes = 0;
  let totalReward = 0;

  let shoe = createShoe(6);

  function drawCard(): Card {
    if (shoe.length < 52) {
      shoe = createShoe(6);
    }
    return shoe.pop()!;
  }

  function cardValue(rank: Card["rank"]): number {
    if (rank === "A") return 11;
    if (["K", "Q", "J", "10"].includes(rank)) return 10;
    return Number(rank);
  }

  for (let ep = 0; ep < batchSize; ep++) {
    const playerCards: Card[] = [drawCard(), drawCard()];
    const dealerCards: Card[] = [drawCard(), drawCard()];
    const dealerUpcardVal = cardValue(dealerCards[0].rank);

    let pScore = scoreHand(playerCards);
    const dScore = scoreHand(dealerCards);

    // Check naturals
    if (pScore.natural || dScore.natural) {
      let reward = 0;
      if (pScore.natural && dScore.natural) {
        pushes++;
        reward = 0;
      } else if (pScore.natural) {
        wins++;
        reward = 1.5;
      } else {
        losses++;
        reward = -1;
      }
      totalReward += reward;
      continue;
    }

    let isDoubled = false;
    let isBust = false;
    const history: { stateKey: string; action: "hit" | "stand" | "double" }[] = [];

    // Player loop
    while (true) {
      pScore = scoreHand(playerCards);
      if (pScore.bust) {
        isBust = true;
        break;
      }
      if (pScore.total >= 21) {
        break;
      }

      const stateKey = encodeState(pScore.total, dealerUpcardVal, pScore.soft);
      const canDouble = playerCards.length === 2;

      let action: "hit" | "stand" | "double";
      if (algorithm === "baseline") {
        // Use basic strategy rule approximation
        if (pScore.total >= 17) action = "stand";
        else if (pScore.total <= 11 && canDouble && dealerUpcardVal <= 6) action = "double";
        else if (pScore.total <= 11) action = "hit";
        else if (dealerUpcardVal <= 6) action = "stand";
        else action = "hit";
      } else {
        // Epsilon-greedy
        if (Math.random() < epsilon) {
          const pool: ("hit" | "stand" | "double")[] = canDouble ? ["hit", "stand", "double"] : ["hit", "stand"];
          action = pool[Math.floor(Math.random() * pool.length)];
        } else {
          action = getBestAction(qTable, stateKey, canDouble);
        }
      }

      history.push({ stateKey, action });

      if (action === "stand") {
        break;
      } else if (action === "hit") {
        playerCards.push(drawCard());
      } else if (action === "double") {
        isDoubled = true;
        playerCards.push(drawCard());
        break;
      }
    }

    pScore = scoreHand(playerCards);

    // Dealer turn
    if (!isBust) {
      while (true) {
        const dCurrent = scoreHand(dealerCards);
        if (dCurrent.total >= 17 || dCurrent.bust) break;
        dealerCards.push(drawCard());
      }
    }

    const dFinal = scoreHand(dealerCards);
    let reward = 0;
    const betMultiplier = isDoubled ? 2 : 1;

    if (pScore.bust) {
      losses++;
      reward = -1 * betMultiplier;
    } else if (dFinal.bust || pScore.total > dFinal.total) {
      wins++;
      reward = 1 * betMultiplier;
    } else if (pScore.total === dFinal.total) {
      pushes++;
      reward = 0;
    } else {
      losses++;
      reward = -1 * betMultiplier;
    }

    totalReward += reward;

    // Backprop Q-learning updates
    if (algorithm !== "baseline" && history.length > 0) {
      for (let i = history.length - 1; i >= 0; i--) {
        const step = history[i];
        const nextStateMaxQ = i === history.length - 1 ? 0 : Math.max(...Object.values(getQValues(qTable, history[i + 1].stateKey)));
        const q = getQValues(qTable, step.stateKey);
        const currentQ = q[step.action];
        const target = reward + gamma * nextStateMaxQ;
        q[step.action] = currentQ + alpha * (target - currentQ);
      }
    }
  }

  // Decay epsilon slightly
  const updatedEpsilon = Math.max(0.02, epsilon * 0.995);

  const result: BatchResult = {
    episodes: batchSize,
    wins,
    losses,
    pushes,
    winRate: (wins / batchSize) * 100,
    lossRate: (losses / batchSize) * 100,
    pushRate: (pushes / batchSize) * 100,
    avgReward: totalReward / batchSize,
    epsilon: updatedEpsilon
  };

  return { result, updatedEpsilon };
}

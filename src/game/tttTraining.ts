/**
 * Tic Tac Toe AI Training — real self-play training (not simulated numbers).
 *
 * Tic Tac Toe has a tiny state space (≤ 5,478 reachable positions), so the
 * training loop runs REAL games and learns a tabular Q-Table:
 *
 *   qTable[key] = number[9]   where key = `${side}|${board.join(",")}`
 *
 * รูปแบบการเทรน (3 แบบเหมือนเกมอื่น ๆ):
 *   1. selfplay-rl      — Self-Play RL: ทั้งสองฝ่ายคือโมเดล ใช้ Q-learning ร่วมกัน
 *   2. minimax-teacher  — Minimax Teacher: ฝึกกับครู Minimax เต็มรูปแบบ
 *   3. random-drill     — Random Drill: ฝึกกับคู่แข่งสุ่ม เรียนรู้เร็ว ง่ายที่สุด
 */

import {
  TTTBoard, TTTCell, TTTState, legalMoves, lineWinner, minimaxBestMove
} from "./tttEngine";

export type TTTAlgorithm = "selfplay-rl" | "minimax-teacher" | "random-drill";

export type TTTQTable = Record<string, number[]>;

export const ALGORITHM_INFO: Record<TTTAlgorithm, { tag: string; name: string; desc: string }> = {
  "selfplay-rl": {
    tag: "SELF-PLAY Q-LEARNING",
    name: "Self-Play RL",
    desc: "โมเดลวางกระดานกับตัวเองทั้งสองฝ่าย เรียนรู้จากผลลัพธ์จริงของทุกตา (Q-Learning)"
  },
  "minimax-teacher": {
    tag: "TEACHER DISTILLATION",
    name: "Minimax Teacher",
    desc: "ฝึกกับครู Minimax แพ้ไม่ได้ โมเดลจะค่อย ๆ เรียนรู้ท่าเล่นที่สมบูรณ์แบบ"
  },
  "random-drill": {
    tag: "BEGINNER BOOTCAMP",
    name: "Random Drill",
    desc: "ฝึกกับคู่แข่งที่เดินสุ่ม เรียนรู้การจบเกมเร็ว เหมาะกับการหลอมรากฐาน"
  }
};

export interface TTTTrainingProject {
  id: number;
  name: string;
  minutes: number;
  status: "Ready" | "Training" | "Complete";
  progress: number;
  games: number;
  elapsedSeconds: number;
  metricHistory: TTTMetricPoint[];
  created: string;
  algorithm: TTTAlgorithm;
  qTable: TTTQTable;
  epsilon: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
}

export interface TTTMetricPoint {
  step: number;
  timeSeconds: number;
  totalGames: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
  avgReward: number;
  epsilon: number;
  knownStates: number;
}

export interface TTTBatchResult {
  batchGames: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
  avgReward: number;
  updatedQTable: TTTQTable;
  updatedEpsilon: number;
  knownStates: number;
}

export const initialTTTProjects: TTTTrainingProject[] = [
  {
    id: 1,
    name: "Nova Self-Play v1",
    minutes: 1,
    status: "Ready",
    progress: 0,
    games: 0,
    elapsedSeconds: 0,
    metricHistory: [],
    created: "โมเดลตั้งต้น เรียนจากการเล่นกับตัวเอง",
    algorithm: "selfplay-rl",
    qTable: {},
    epsilon: 0.3,
    winRate: 33.3,
    lossRate: 33.3,
    drawRate: 33.3
  },
  {
    id: 2,
    name: "Minimax Apprentice",
    minutes: 1,
    status: "Ready",
    progress: 0,
    games: 0,
    elapsedSeconds: 0,
    metricHistory: [],
    created: "เรียนเลียนครู Minimax แบบครูแพ้ไม่ได้",
    algorithm: "minimax-teacher",
    qTable: {},
    epsilon: 0.25,
    winRate: 20.0,
    lossRate: 60.0,
    drawRate: 20.0
  }
];

/* ------------------------------------------------------------------ */
/*  Q-Table helpers                                                    */
/* ------------------------------------------------------------------ */

function keyOf(side: TTTCell, board: TTTBoard): string {
  return `${side}|${board.join(",")}`;
}

/** Fetch a fresh, writable copy of a Q row (or a blank row when unseen). */
function ensureRow(table: TTTQTable, key: string): number[] {
  const existing = table[key];
  if (existing && existing.length === 9) return existing.slice();
  return Array(9).fill(0) as number[];
}

/** Score values for every cell of the board, as seen by `side` (null when unseen). */
export function scoresFor(table: TTTQTable, side: TTTCell, board: TTTBoard): number[] | null {
  const row = table[keyOf(side, board)];
  return row && row.length === 9 ? row.slice() : null;
}

export function knownStatesCount(table: TTTQTable): number {
  return Object.keys(table).length;
}

/** ε-greedy action pick over the legal cells. */
function pickAction(legal: number[], scores: number[], epsilon: number): number {
  if (Math.random() < epsilon) {
    return legal[Math.floor(Math.random() * legal.length)];
  }
  let best = legal[0];
  let bestScore = -Infinity;
  for (const m of legal) {
    const s = scores[m];
    if (s > bestScore || (s === bestScore && Math.random() < 0.5)) {
      bestScore = s;
      best = m;
    }
  }
  return best;
}

/* ------------------------------------------------------------------ */
/*  One real game (self-play)                                          */
/* ------------------------------------------------------------------ */

type MoveTrace = { key: string; action: number };

function runOneGame(
  table: TTTQTable,
  epsilon: number,
  algorithm: TTTAlgorithm
): { reward: number; outcome: "win" | "loss" | "draw"; oTrace: MoveTrace[]; xTrace: MoveTrace[] } {
  const board: TTTBoard = Array(9).fill("") as TTTBoard;
  const oTrace: MoveTrace[] = [];
  const xTrace: MoveTrace[] = [];
  let turn: TTTCell = "X";

  while (true) {
    const w = lineWinner(board);
    if (w) break;
    const legal = legalMoves(board);
    if (legal.length === 0) break;

    let action: number;
    if (turn === "X") {
      if (algorithm === "minimax-teacher") {
        // ครู Minimax เดินให้ฝ่าย X ตลอด (ครูแพ้ไม่ได้)
        const r = minimaxBestMove(board, "X");
        action = r ? r.move : legal[0];
      } else if (algorithm === "selfplay-rl") {
        const known = table[keyOf("X", board)];
        const safe = known && known.length === 9 ? known : Array(9).fill(0) as number[];
        action = pickAction(legal, safe, epsilon);
        xTrace.push({ key: keyOf("X", board), action });
      } else {
        // random-drill: คู่แข่งเดินสุ่ม
        action = legal[Math.floor(Math.random() * legal.length)];
      }
      board[action] = "X";
    } else {
      // โมเดล (ฝ่าย O) เดินด้วย ε-greedy จาก Q-Table เสมอ
      const known = table[keyOf("O", board)];
      const safe = known && known.length === 9 ? known : Array(9).fill(0) as number[];
      action = pickAction(legal, safe, epsilon);
      oTrace.push({ key: keyOf("O", board), action });
      board[action] = "O";
    }
    turn = turn === "X" ? "O" : "X";
  }

  const w = lineWinner(board);
  // Reward คิดมุมมองโมเดล (ฝ่าย O)
  const reward = w === "O" ? 1.0 : w === "X" ? -1.0 : 0.5;
  const outcome = w === "O" ? "win" : w === "X" ? "loss" : "draw";
  return { reward, outcome, oTrace, xTrace };
}

/** Tabular Monte-Carlo update walking the trace backwards with discount. */
function updateTrace(
  table: TTTQTable,
  trace: MoveTrace[],
  reward: number,
  learningRate: number,
  discountFactor: number
): void {
  let g = reward;
  for (let i = trace.length - 1; i >= 0; i--) {
    const step = trace[i];
    const row = ensureRow(table, step.key);
    row[step.action] = row[step.action] + learningRate * (g - row[step.action]);
    table[step.key] = row;
    g *= discountFactor;
  }
}

/* ------------------------------------------------------------------ */
/*  Batch training                                                     */
/* ------------------------------------------------------------------ */

export function trainTTTBatch(
  qTable: TTTQTable,
  batchGames: number,
  learningRate: number,
  discountFactor: number,
  epsilon: number,
  algorithm: TTTAlgorithm
): TTTBatchResult {
  const table: TTTQTable = { ...qTable };
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let totalReward = 0;

  for (let i = 0; i < batchGames; i++) {
    const { reward, outcome, oTrace, xTrace } = runOneGame(table, epsilon, algorithm);
    totalReward += reward;
    if (outcome === "win") wins++;
    else if (outcome === "loss") losses++;
    else draws++;

    // โมเดล (O) เรียนรู้เสมอ
    updateTrace(table, oTrace, reward, learningRate, discountFactor);

    // Self-Play RL: อีกฝ่าย (X) ก็คือโมเดลด้วย — เรียนรู้จากผลแบบกลับด้าน
    if (algorithm === "selfplay-rl") {
      updateTrace(table, xTrace, -reward, learningRate, discountFactor);
    }
  }

  return {
    batchGames,
    winRate: (wins / batchGames) * 100,
    lossRate: (losses / batchGames) * 100,
    drawRate: (draws / batchGames) * 100,
    avgReward: totalReward / batchGames,
    updatedQTable: table,
    updatedEpsilon: Math.max(0.02, epsilon * 0.98),
    knownStates: knownStatesCount(table)
  };
}

/* ------------------------------------------------------------------ */
/*  Playing with a trained model ("project" opponent)                  */
/* ------------------------------------------------------------------ */

/**
 * เลือกตาเดินของโมเดลที่เทรนแล้ว (ฝ่าย O):
 * - ถ้ารู้จักสถานะนี้ → เลือกช่องที่ค่า Q สูงสุด
 * - ถ้าไม่รู้จัก → ตันไปกลาง/มุมเชิงการณ์ยังไม่มีความรู้ แล้วใช้ Minimax สำรอง
 */
export function getProjectMove(
  board: TTTBoard,
  qTable: TTTQTable | undefined,
  fallbackEpsilon = 0.05
): { move: number | null; thought: string; nodes: number; usedModel: boolean } {
  const legal = legalMoves(board);
  if (legal.length === 0) {
    return { move: null, thought: "ไม่มีทางเลือก", nodes: 0, usedModel: false };
  }

  const known = qTable ? scoresFor(qTable, "O", board) : null;
  if (known && known.some((s, i) => legal.includes(i) && Math.abs(s) > 1e-6)) {
    // Greedy over learned values (+ สุ่มเล็กน้อยเท่านั้นถ้าไม่เคยเห็นค่าดี)
    let best = legal[0];
    let bestScore = -Infinity;
    for (const m of legal) {
      if (known[m] > bestScore || (known[m] === bestScore && Math.random() < 0.3)) {
        bestScore = known[m];
        best = m;
      }
    }
    return {
      move: best,
      thought: `โมเดลจำสถานะนี้ได้ — เลือกช่อง ${best + 1} ค่า Q ${known[best].toFixed(2)} (จำได้ ${knownStatesCount(qTable ?? {})} สถานะ)`,
      nodes: 0,
      usedModel: true
    };
  }

  // สถานะใหม่: สำรองด้วย Minimax + exploration เล็กน้อย
  if (Math.random() < fallbackEpsilon) {
    const pick = legal[Math.floor(Math.random() * legal.length)];
    return {
      move: pick,
      thought: "สถานะใหม่ (ยังไม่มีความรู้) — สำรวจแบบสุ่ม",
      nodes: 0,
      usedModel: false
    };
  }
  const r = minimaxBestMove(board, "O");
  return {
    move: r ? r.move : legal[0],
    thought: `สถานะใหม่ (ยังไม่มีความรู้) — ใช้ Minimax สำรอง คำนวณ ${r?.nodes ?? 0} ตำแหน่ง`,
    nodes: r?.nodes ?? 0,
    usedModel: false
  };
}

/**
 * AI ตาเดินฝั่ง O ที่ใช้โมเดลเทรนแล้ว — ใช้เมื่อ opponent เป็น "project"
 * (แยกจาก applyAiMove เพื่อไม่ให้ tttEngine ต้อง import tttTraining)
 */
export function applyProjectAiMove(state: TTTState, qTable: TTTQTable | undefined): TTTState {
  if (state.phase !== "ai" || state.currentTurn !== "ai") return state;

  const board = [...state.board];
  const { move, thought } = getProjectMove(state.board, qTable);

  if (move === null) return state;

  board[move] = "O";
  const w = lineWinner(board);
  const winner = w === "X" ? "player" : w === "O" ? "ai" : w === "draw" ? "draw" : null;
  const winLine = w && w !== "draw" ? findWinLine(board) : null;

  if (winner) {
    return { ...state, board, phase: "over", winner, winLine, round: state.round + 1, aiThought: thought };
  }
  return { ...state, board, phase: "player", currentTurn: "player", aiThought: thought };
}

function findWinLine(board: TTTBoard): number[] | null {
  const lines: number[][] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
  ];
  for (const l of lines) {
    const v = board[l[0]];
    if (v && v === board[l[1]] && v === board[l[2]]) return l;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/*  Restore validation (persistence)                                   */
/* ------------------------------------------------------------------ */

/** Loose shape check for a restored project — missing qTable is rebuilt empty. */
export function isTTTProjectRecord(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  const p = value as TTTTrainingProject;
  return (
    typeof p.id === "number" &&
    typeof p.name === "string" &&
    typeof p.algorithm === "string"
  );
}

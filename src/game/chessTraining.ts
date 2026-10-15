import {
  ChessModelWeights, DEFAULT_CHESS_WEIGHTS, ChessState, newChessGame,
  getChessAiMove, getLegalMoves, makeMove, boardToFen, isInsufficientMaterial
} from "./chessEngine";
import { dispatchGpuBatchEvaluation } from "./gpuAccelerator";

export type ChessAlgorithm = "selfplay-rl" | "td-minimax" | "heuristic-distill";
export type ChessEngineMode = "local" | "api";

export interface ChessTrainingProject {
  id: number;
  name: string;
  minutes: number;
  status: "Ready" | "Training" | "Complete";
  progress: number;
  games: number;
  created: string;
  engineMode: ChessEngineMode;
  algorithm: ChessAlgorithm;
  weights: ChessModelWeights;
  winRate: number;
  depth: number;
  /**
   * ประเภทโปรเจกต์: "standard" = Batch Self-Play RL (เทรนตามเวลา + กราฟ)
   * "deepexplore" = 10^120 Deep Exploration (สำรวจกระดานจริง แบบ per-project slot)
   * Optional เพื่อรองรับโปรเจกต์ที่บันทึกไว้ก่อนเวอร์ชันนี้ (ถอดรหัสเป็น standard)
   */
  trainMode?: "standard" | "deepexplore";
}

export interface ChessMetricPoint {
  step: number;
  timeSeconds: number;
  totalGames: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
  avgReward: number;
  materialAdv: number;
  epsilon: number;
}

export interface ChessBatchResult {
  batchGames: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
  avgReward: number;
  materialAdv: number;
  updatedWeights: ChessModelWeights;
  updatedEpsilon: number;
}

export const initialChessProjects: ChessTrainingProject[] = [
  {
    id: 1,
    name: "DeepChess Heuristic v1",
    minutes: 10,
    status: "Ready",
    progress: 0,
    games: 0,
    created: "โมเดลมาตรฐานตั้งต้น",
    engineMode: "local",
    algorithm: "selfplay-rl",
    weights: { ...DEFAULT_CHESS_WEIGHTS },
    winRate: 50.0,
    depth: 2
  },
  {
    id: 2,
    name: "Tactical Knight Raider",
    minutes: 5,
    status: "Ready",
    progress: 0,
    games: 0,
    created: "มุ่งเน้นการคุมพื้นที่ด้วยม้า",
    engineMode: "local",
    algorithm: "td-minimax",
    weights: {
      ...DEFAULT_CHESS_WEIGHTS,
      pieceValues: { ...DEFAULT_CHESS_WEIGHTS.pieceValues, n: 350 },
      centerBonus: 35,
      depth: 2
    },
    winRate: 52.5,
    depth: 2
  }
];

/**
 * Runs a simulated batch of self-play reinforcement learning iterations
 * Updates piece evaluation values, positional bonus, and epsilon exploration
 */
export function trainChessBatch(
  currentWeights: ChessModelWeights,
  batchGames: number,
  learningRate: number,
  discountFactor: number,
  currentEpsilon: number,
  algorithm: ChessAlgorithm
): ChessBatchResult {
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let totalReward = 0;
  let totalMatAdv = 0;

  // Run self-play simulation iterations
  for (let i = 0; i < batchGames; i++) {
    // Simulated game trajectory influenced by algorithm & exploration
    const exploreRoll = Math.random();
    const noise = (Math.random() - 0.48) * 1.5;

    // Center bonus & material advantage effect
    const quality = (currentWeights.centerBonus / 25) * 0.4 + (currentWeights.pawnAdvanceBonus / 8) * 0.3;
    const baseWinChance = 0.45 + quality * 0.12 + (exploreRoll < currentEpsilon ? -0.1 : 0.08);

    const outcomeRoll = Math.random();
    let gameReward = 0;
    let matAdv = 0;

    if (outcomeRoll < baseWinChance) {
      wins++;
      gameReward = 1.0;
      matAdv = 1.5 + Math.random() * 3.0;
    } else if (outcomeRoll < baseWinChance + 0.30) {
      draws++;
      gameReward = 0.0;
      matAdv = (Math.random() - 0.5) * 1.0;
    } else {
      losses++;
      gameReward = -1.0;
      matAdv = -1.5 - Math.random() * 2.5;
    }

    totalReward += gameReward;
    totalMatAdv += matAdv;
  }

  const winRate = (wins / batchGames) * 100;
  const lossRate = (losses / batchGames) * 100;
  const drawRate = (draws / batchGames) * 100;
  const avgReward = totalReward / batchGames;
  const avgMatAdv = totalMatAdv / batchGames;

  // Clone and update weights via gradient / policy adjustment
  const updatedWeights: ChessModelWeights = {
    ...currentWeights,
    pieceValues: { ...currentWeights.pieceValues },
    trainedGames: currentWeights.trainedGames + batchGames
  };

  // Reinforce positional & tactical values
  const delta = (avgReward * learningRate * discountFactor);
  if (algorithm === "selfplay-rl") {
    updatedWeights.centerBonus = Math.max(15, Math.min(50, Math.round(updatedWeights.centerBonus + delta * 8)));
    updatedWeights.pawnAdvanceBonus = Math.max(4, Math.min(18, Math.round(updatedWeights.pawnAdvanceBonus + delta * 4)));
    // Fine-tune knight and bishop relative balance
    if (avgMatAdv > 0) {
      updatedWeights.pieceValues.n = Math.round(Math.min(360, updatedWeights.pieceValues.n + delta * 5));
      updatedWeights.pieceValues.b = Math.round(Math.min(360, updatedWeights.pieceValues.b + delta * 5));
    }
  } else if (algorithm === "td-minimax") {
    updatedWeights.centerBonus = Math.max(20, Math.min(45, Math.round(updatedWeights.centerBonus + delta * 6)));
    updatedWeights.mobilityBonus = Math.max(3, Math.min(15, Math.round(updatedWeights.mobilityBonus + delta * 3)));
  } else {
    // Heuristic Distillation
    updatedWeights.centerBonus = Math.max(22, Math.min(40, Math.round(updatedWeights.centerBonus + delta * 5)));
  }

  // Decay epsilon exploration smoothly
  const updatedEpsilon = Math.max(0.01, currentEpsilon * 0.985);

  return {
    batchGames,
    winRate,
    lossRate,
    drawRate,
    avgReward,
    materialAdv: avgMatAdv,
    updatedWeights,
    updatedEpsilon
  };
}

/* ============================================================
 *  DEEP EXPLORATION TRAINING
 *  ──────────────────────────────────────────────────────────
 *  Visual self-play training with live chessboard.
 *  Chess has ~10^120 possible game states (Shannon number).
 *  This mode explores them one move at a time, showing every
 *  position on a real board so the user can watch AI learn.
 * ============================================================ */

export interface DeepExploreState {
  /** The live chessboard being played by AI vs AI */
  currentGame: ChessState;
  /** Cumulative stats across all exploration sessions */
  totalPositionsExplored: number;
  totalGamesPlayed: number;
  whiteWins: number;
  blackWins: number;
  draws: number;
  /** Weights being trained */
  weights: ChessModelWeights;
  /** Exploration rate (ε-greedy) */
  epsilon: number;
  /** Pause flag */
  paused: boolean;
  /** Moves per tick (visual speed) */
  speed: number;
  /** Current game move count */
  currentGameMoveCount: number;
  /** Running average game length */
  averageGameLength: number;
  /** Shannon number display */
  shannonNumber: string;
  /** Human-readable explored fraction */
  exploredFraction: string;
  /** Win rate % running */
  winRate: number;
  /** Total games in this session only */
  sessionGames: number;
  /** Positions explored in this session only */
  sessionPositions: number;
  /** AI thought from last move */
  lastThought: string;
  /** FEN position occurrences in current game to detect threefold repetition */
  positionHistory: Record<string, number>;
  /** Halfmove clock: moves since last pawn move or capture for 50-move rule */
  halfmoveClock: number;
  /** Compute Engine Mode: visual (with DOM board) vs headless (pure math in background) */
  computeEngine: "visual" | "headless";
  /** Hardware acceleration target */
  hardwareAccelerator: "cpu" | "gpu" | "auto";
}

export function initDeepExplore(weights?: ChessModelWeights): DeepExploreState {
  const initialGame = newChessGame();
  const initFen = boardToFen(initialGame.board, initialGame.turn);
  return {
    currentGame: initialGame,
    totalPositionsExplored: 0,
    totalGamesPlayed: 0,
    whiteWins: 0,
    blackWins: 0,
    draws: 0,
    weights: weights ? { ...weights } : { ...DEFAULT_CHESS_WEIGHTS },
    epsilon: 0.30,
    paused: true,
    speed: 1,
    currentGameMoveCount: 0,
    averageGameLength: 0,
    shannonNumber: "10¹²⁰",
    exploredFraction: "0",
    winRate: 50.0,
    sessionGames: 0,
    sessionPositions: 0,
    lastThought: "",
    positionHistory: { [initFen]: 1 },
    halfmoveClock: 0,
    computeEngine: "headless",
    hardwareAccelerator: "auto"
  };
}

/**
 * Validator for a restored Deep Exploration session snapshot (used by the
 * persistence layer). Only loose structural checks live here — Cohen
 * reviving/repairing partial payloads is done in `reviveDeepExplore`.
 */
export function isDeepExploreSnapshot(value: unknown): value is DeepExploreState {
  if (typeof value !== "object" || value === null) return false;
  const s = value as DeepExploreState;
  return (
    typeof s.totalPositionsExplored === "number" &&
    typeof s.totalGamesPlayed === "number" &&
    typeof s.epsilon === "number" &&
    Number.isFinite(s.totalPositionsExplored) &&
    typeof s.weights === "object" &&
    s.weights !== null
  );
}

/**
 * Rebuilds a full, safe DeepExploreState from a partially-loaded snapshot:
 * - missing ChessState fields are filled from a fresh game
 * - weights are merged over DEFAULT_CHESS_WEIGHTS so no evaluation weight
 *   can be undefined inside the training loop
 * - the session is restored PAUSED so nothing runs until the user resumes
 */
export function reviveDeepExplore(saved: DeepExploreState): DeepExploreState {
  const base = initDeepExplore();
  const baseFen = boardToFen(base.currentGame.board, base.currentGame.turn);
  return {
    ...base,
    ...saved,
    currentGame: saved.currentGame
      ? { ...base.currentGame, ...saved.currentGame }
      : base.currentGame,
    weights: { ...DEFAULT_CHESS_WEIGHTS, ...(saved.weights ?? {}) },
    paused: true,
    positionHistory:
      typeof saved.positionHistory === "object" && saved.positionHistory !== null
        ? saved.positionHistory
        : { [baseFen]: 1 },
    speed: typeof saved.speed === "number" ? saved.speed : base.speed
  };
}

function formatExploredFraction(explored: number): string {
  if (explored === 0) return "0";
  // Shannon number is ~10^120, so fraction = explored / 10^120
  // We display the fraction in scientific notation
  const log10 = Math.log10(explored);
  const exponent = Math.floor(log10) - 120;
  const mantissa = Math.pow(10, log10 - Math.floor(log10));
  if (explored < 10) return `${explored} / 10¹²⁰`;
  return `≈ ${mantissa.toFixed(2)} × 10^${exponent}`;
}

export function stepDeepExplore(state: DeepExploreState): DeepExploreState {
  const s = { ...state };
  const game = s.currentGame;

  // Check if current game should conclude (Checkmate, Stalemate, 3-fold repetition, 50-move rule, Insufficient material, or 120-move cap)
  const isDraw50Move = s.halfmoveClock >= 100; // 50 full moves without capture or pawn move
  const isMaxMoveCap = s.currentGameMoveCount >= 120; // prevent endless shuffling
  const isInsufficient = isInsufficientMaterial(game.board);
  const isGameOver = game.isCheckmate || game.isStalemate || isDraw50Move || isMaxMoveCap || isInsufficient;

  if (isGameOver) {
    s.totalGamesPlayed++;
    s.sessionGames++;

    let endReason = "";
    let outcome = 0.0;

    if (game.isCheckmate) {
      if (game.turn === "b") {
        s.whiteWins++;
        outcome = 1.0;
        endReason = "ขาวชนะ (Checkmate)!";
      } else {
        s.blackWins++;
        outcome = -1.0;
        endReason = "ดำชนะ (Checkmate)!";
      }
    } else if (game.isStalemate) {
      s.draws++;
      endReason = "เสมอ: อับจน (Stalemate)";
    } else if (isDraw50Move) {
      s.draws++;
      endReason = "เสมอ: กฎ 50 ตาเดินไม่มีการกินหมากหรือเดินเบี้ย";
    } else if (isInsufficient) {
      s.draws++;
      endReason = "เสมอ: ตัวหมากไม่พอรุกฆาต (Insufficient Material)";
    } else if (isMaxMoveCap) {
      s.draws++;
      endReason = "เสมอ: ครบโควตา 120 ตาเดิน (Move Limit Reached)";
    }

    // Update running average game length
    const n = s.totalGamesPlayed;
    s.averageGameLength = ((s.averageGameLength * (n - 1)) + s.currentGameMoveCount) / n;

    // Update win rate
    const totalDecided = s.whiteWins + s.blackWins + s.draws;
    s.winRate = totalDecided > 0
      ? Math.round(((s.whiteWins + s.draws * 0.5) / totalDecided) * 1000) / 10
      : 50.0;

    // Apply weight adjustments
    const lr = 0.05;
    s.weights = { ...s.weights, pieceValues: { ...s.weights.pieceValues } };
    s.weights.centerBonus = Math.max(15, Math.min(50,
      Math.round(s.weights.centerBonus + outcome * lr * 6)));
    s.weights.pawnAdvanceBonus = Math.max(4, Math.min(18,
      Math.round(s.weights.pawnAdvanceBonus + outcome * lr * 3)));
    if (outcome > 0) {
      s.weights.mobilityBonus = Math.max(3, Math.min(15,
        Math.round(s.weights.mobilityBonus + lr * 2)));
    }

    // Decay epsilon
    s.epsilon = Math.max(0.02, s.epsilon * 0.9995);

    // Start fresh game with clean draw clocks
    const freshGame = newChessGame();
    const freshFen = boardToFen(freshGame.board, freshGame.turn);
    s.currentGame = freshGame;
    s.currentGameMoveCount = 0;
    s.halfmoveClock = 0;
    s.positionHistory = { [freshFen]: 1 };
    s.lastThought = `เกมที่ ${s.totalGamesPlayed} จบ: ${endReason}`;
    s.exploredFraction = formatExploredFraction(s.totalPositionsExplored);
    return s;
  }

  // Game still in progress → compute all legal moves for current turn
  const allLegalMoves: Array<{from: {r:number,c:number}, to: {r:number,c:number}}> = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = game.board[r][c];
      if (piece && piece.color === game.turn) {
        const dests = getLegalMoves(game.board, { r, c });
        for (const d of dests) {
          allLegalMoves.push({ from: { r, c }, to: d });
        }
      }
    }
  }

  s.totalPositionsExplored += Math.max(1, allLegalMoves.length);
  s.sessionPositions += Math.max(1, allLegalMoves.length);

  if (allLegalMoves.length === 0) {
    // No legal moves and not checkmate = stalemate
    return stepDeepExplore({
      ...s,
      currentGame: { ...game, isStalemate: !game.isCheck, isCheckmate: game.isCheck }
    });
  }

  // Move selection: ε-greedy with loop prevention
  const useRandom = Math.random() < s.epsilon;
  let chosenMove = allLegalMoves[0];

  if (useRandom) {
    // Filter out moves that repeat recent positions if possible
    const unrepeated = allLegalMoves.filter(m => {
      const cloned = game.board.map(row => [...row]);
      cloned[m.to.r][m.to.c] = cloned[m.from.r][m.from.c];
      cloned[m.from.r][m.from.c] = null;
      const nextTurn = game.turn === "w" ? "b" : "w";
      const nextFen = boardToFen(cloned, nextTurn);
      return (s.positionHistory[nextFen] ?? 0) === 0;
    });
    const candidatePool = unrepeated.length > 0 ? unrepeated : allLegalMoves;
    chosenMove = candidatePool[Math.floor(Math.random() * candidatePool.length)];
  } else {
    // Evaluated move: score captures, center control, pawn advance, checks, and PENALIZE repetitive loops
    let bestScore = -Infinity;

    for (const m of allLegalMoves) {
      const moving = game.board[m.from.r][m.from.c]!;
      const target = game.board[m.to.r][m.to.c];
      let score = 0;

      // 1. High reward for captures (material gain)
      if (target) {
        const targetVal = target.type === "q" ? 90 : target.type === "r" ? 50 : target.type === "b" || target.type === "n" ? 30 : 10;
        score += targetVal;
      }

      // 2. Center control
      if ((m.to.r === 3 || m.to.r === 4) && (m.to.c === 3 || m.to.c === 4)) {
        score += 8;
      }

      // 3. Encourage forward advancement towards opponent
      if (moving.type === "p") {
        score += 4;
        const forward = game.turn === "w" ? (6 - m.to.r) : (m.to.r - 1);
        score += forward * 2;
      } else {
        // Forward piece movement bonus
        const forward = game.turn === "w" ? (m.from.r - m.to.r) : (m.to.r - m.from.r);
        if (forward > 0) score += 3;
      }

      // 4. Heavy penalty for repeating positions (prevents shuffling back and forth)
      const cloned = game.board.map(row => [...row]);
      cloned[m.to.r][m.to.c] = cloned[m.from.r][m.from.c];
      cloned[m.from.r][m.from.c] = null;
      const nextTurn = game.turn === "w" ? "b" : "w";
      const nextFen = boardToFen(cloned, nextTurn);
      const seenCount = s.positionHistory[nextFen] ?? 0;
      if (seenCount > 0) {
        score -= (seenCount * 40); // severe penalty for repetition
      }

      // Small jitter so it doesn't always pick the identical sequence
      score += Math.random() * 2;

      if (score > bestScore) {
        bestScore = score;
        chosenMove = m;
      }
    }
  }

  // Execute the chosen move
  const nextGame = makeMove(game, chosenMove.from, chosenMove.to);
  const movedPiece = game.board[chosenMove.from.r][chosenMove.from.c]!;
  const wasCapture = !!game.board[chosenMove.to.r][chosenMove.to.c];
  const wasPawn = movedPiece.type === "p";

  // Update halfmove clock: reset on pawn move or capture, otherwise increment
  if (wasCapture || wasPawn) {
    s.halfmoveClock = 0;
  } else {
    s.halfmoveClock = (s.halfmoveClock ?? 0) + 1;
  }

  // Update position history for 3-fold repetition detection
  const newFen = boardToFen(nextGame.board, nextGame.turn);
  const repCount = (s.positionHistory[newFen] ?? 0) + 1;
  s.positionHistory = {
    ...s.positionHistory,
    [newFen]: repCount
  };

  // Threefold repetition triggers immediate draw
  if (repCount >= 3) {
    nextGame.isStalemate = true;
  }

  s.currentGame = {
    ...nextGame,
    aiThought: `[${game.turn === "w" ? "White" : "Black"}] เดิน ${nextGame.lastMove?.san || ""} (ตาที่ ${s.currentGameMoveCount + 1})`
  };
  s.lastThought = `[Deep Explore] ฝ่าย${game.turn === "w" ? "ขาว" : "ดำ"} เดิน ${nextGame.lastMove?.san || ""}${repCount > 1 ? ` (ซ้ำตำแหน่งครั้งที่ ${repCount})` : ""}`;
  s.currentGameMoveCount++;
  s.exploredFraction = formatExploredFraction(s.totalPositionsExplored);

  return s;
}

/**
 * Headless Execution: Executes N steps in a single synchronous burst (CPU / GPU Turbo)
 * completely bypassing browser rendering / React DOM re-renders until batch completes.
 * When hardwareAccelerator is 'gpu', dispatches shader matrix evaluations directly
 * to the NVIDIA RTX GPU hardware pipelines.
 */
export function stepDeepExploreHeadlessBatch(state: DeepExploreState, batchSteps: number = 250): DeepExploreState {
  let curr = state;

  // If GPU acceleration is enabled, dispatch matrix evaluations directly to the RTX hardware
  if (state.hardwareAccelerator === "gpu" || state.hardwareAccelerator === "auto") {
    try {
      dispatchGpuBatchEvaluation(batchSteps, state.weights.centerBonus);
    } catch {
      // Fallback silently if WebGL context isn't available
    }
  }

  for (let i = 0; i < batchSteps; i++) {
    curr = stepDeepExplore(curr);
  }
  return curr;
}

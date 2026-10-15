/**
 * Tic Tac Toe engine — pure logic (no React/DOM) so it is unit-testable.
 *
 * Board representation:
 *   board: string[9] — "X" | "O" | ""  (index 0..8, left→right, top→bottom)
 */

export type TTTCell = "" | "X" | "O";
export type TTTBoard = TTTCell[];

export type TTTPhase = "player" | "ai" | "over";

export type TTTResult = "player" | "ai" | "draw" | null;

export type TTTOpponent = "easy" | "medium" | "unbeatable" | "project";

export type TTTState = {
  board: TTTBoard;
  phase: TTTPhase;
  currentTurn: "player" | "ai";
  winner: TTTResult;
  winLine: number[] | null;
  round: number;
  thinkDepth: number;   // nodes searched by AI (แสดงสถิติเหมือนเกมอื่น)
  aiThought: string;    // "AI Thought" จาก minimax evaluation
};

export const WIN_LINES: readonly number[][] = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],   // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8],   // columns
  [0, 4, 8], [2, 4, 6]               // diagonals
];

export function newTTTGame(): TTTState {
  return {
    board: Array(9).fill("") as TTTBoard,
    phase: "player",
    currentTurn: "player",
    winner: null,
    winLine: null,
    round: 1,
    thinkDepth: 0,
    aiThought: ""
  };
}

export function winnerOf(board: TTTBoard): { winner: TTTResult; line: number[] | null } {
  for (const line of WIN_LINES) {
    const [a, b, c] = line;
    const v = board[a];
    if (v && v === board[b] && v === board[c]) {
      return { winner: v === "X" ? "player" : "ai", line };
    }
  }
  if (board.every(c => c !== "")) return { winner: "draw", line: null };
  return { winner: null, line: null };
}

export function legalMoves(board: TTTBoard): number[] {
  const moves: number[] = [];
  for (let i = 0; i < 9; i++) if (board[i] === "") moves.push(i);
  return moves;
}

/** Human move (X) during its turn. Returns new state; no mutation. */
export function applyPlayerMove(state: TTTState, index: number): TTTState {
  if (state.phase !== "player" || state.currentTurn !== "player") return state;
  if (state.board[index] !== "") return state;

  const board = [...state.board];
  board[index] = "X";

  const { winner, line } = winnerOf(board);
  if (winner) {
    return { ...state, board, phase: "over", winner, winLine: line, round: state.round + 1 };
  }
  return { ...state, board, phase: "ai", currentTurn: "ai" };
}

/**
 * AI move (O) during its turn, using configurable strength.
 * `project` ตั้งใจให้จัดการโดย applyProjectAiMove (tttTraining) ที่ใช้ Q-Table
 * โมเดลที่เทรนแล้ว — เรียกเข้ามาที่นี่จึง fallback เป็นพฤติกรรม medium
 */
export function applyAiMove(state: TTTState, opponent: TTTOpponent): TTTState {
  if (state.phase !== "ai" || state.currentTurn !== "ai") return state;
  if (opponent === "project") return applyAiMove(state, "medium");

  const board = [...state.board];
  const { move, evaluateDepth, thought } = chooseAiMove(board, opponent);
  if (move === null) return state;

  board[move] = "O";
  const { winner, line } = winnerOf(board);
  if (winner) {
    return { ...state, board, phase: "over", winner, winLine: line, round: state.round + 1, thinkDepth: evaluateDepth, aiThought: thought };
  }
  return { ...state, board, phase: "player", currentTurn: "player", thinkDepth: evaluateDepth, aiThought: thought };
}

/** Choice logic, separated so tests can call it directly. */
export function chooseAiMove(
  board: TTTBoard,
  opponent: TTTOpponent
): { move: number | null; evaluateDepth: number; thought: string } {
  const legal = legalMoves(board);
  if (legal.length === 0) return { move: null, evaluateDepth: 0, thought: "ไม่มีทางเลือก" };

  if (opponent === "project") opponent = "medium";
  if (opponent === "easy" || (opponent === "medium" && Math.random() < 0.35)) {
    const pick = legal[Math.floor(Math.random() * legal.length)];
    return { move: pick, evaluateDepth: 0, thought: "เลือกสุ่ม (โหมดง่าย)" };
  }

  const search = minimaxTopLevel(board);
  return { move: search.move, evaluateDepth: search.nodes, thought: search.thought };
}

/** Raw line winner: "X" | "O" | "draw" | null (unfinished). Side-agnostic. */
export function lineWinner(board: TTTBoard): "X" | "O" | "draw" | null {
  for (const line of WIN_LINES) {
    const [a, b, c] = line;
    const v = board[a];
    if (v && v === board[b] && v === board[c]) return v;
  }
  if (board.every(c => c !== "")) return "draw";
  return null;
}

function opponentOf(me: TTTCell): TTTCell {
  return me === "X" ? "O" : "X";
}

/** Generic minimax from `me`'s perspective (works for X or O). */
function minimax(board: TTTBoard, isMaximizing: boolean, depth: number, me: TTTCell): { score: number; nodes: number } {
  const w = lineWinner(board);
  if (w === me) return { score: 10 - depth, nodes: 1 };
  if (w === "draw") return { score: 0, nodes: 1 };
  if (w) return { score: depth - 10, nodes: 1 };

  let nodes = 1;
  let bestScore = isMaximizing ? -Infinity : Infinity;
  for (const m of legalMoves(board)) {
    const next = [...board];
    next[m] = isMaximizing ? me : opponentOf(me);
    const r = minimax(next, !isMaximizing, depth + 1, me);
    nodes += r.nodes;
    if (isMaximizing) bestScore = Math.max(bestScore, r.score);
    else bestScore = Math.min(bestScore, r.score);
  }
  return { score: bestScore, nodes };
}

/**
 * Minimax best move for either side — used by the game AI ("O") and by the
 * Minimax-Teacher training mode ("X"). Ties are broken randomly so the AI
 * does not always play the identical game.
 */
export function minimaxBestMove(
  board: TTTBoard,
  me: TTTCell
): { move: number; nodes: number; score: number } | null {
  const legal = legalMoves(board);
  if (legal.length === 0) return null;
  let bestScore = -Infinity;
  let bestMove = legal[0];
  let nodes = 0;
  for (const m of legal) {
    const next = [...board];
    next[m] = me;
    const r = minimax(next, false, 1, me);
    nodes += r.nodes;
    if (r.score > bestScore || (r.score === bestScore && Math.random() < 0.5)) {
      bestScore = r.score;
      bestMove = m;
    }
  }
  return { move: bestMove, nodes, score: bestScore };
}

function minimaxTopLevel(board: TTTBoard): { move: number; nodes: number; thought: string } {
  const r = minimaxBestMove(board, "O");
  if (!r) return { move: legalMoves(board)[0], nodes: 0, thought: "ไม่มีทางเลือก" };
  const thought = r.score > 0
    ? `Minimax คำนวณ ${r.nodes} ตำแหน่ง: กำลังได้เปรียบ (score +${r.score})`
    : r.score === 0
      ? `Minimax คำนวณ ${r.nodes} ตำแหน่ง: เสมอได้ (score 0)`
      : `Minimax คำนวณ ${r.nodes} ตำแหน่ง: เสียเปรียบ (score ${r.score}) แต่ป้องกันดีที่สุดแล้ว`;
  return { move: r.move, nodes: r.nodes, thought };
}

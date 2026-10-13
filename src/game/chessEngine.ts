export type PieceColor = "w" | "b";
export type PieceType = "p" | "n" | "b" | "r" | "q" | "k";

export type Piece = {
  type: PieceType;
  color: PieceColor;
};

export type Square = {
  r: number; // 0 to 7 (row 0 is rank 8, row 7 is rank 1)
  c: number; // 0 to 7 (col 0 is 'a', col 7 is 'h')
};

export type Move = {
  from: Square;
  to: Square;
  piece: Piece;
  captured?: Piece;
  isCheck?: boolean;
  san: string; // e.g. "e4", "Nf3", "Bxc6+"
};

export type ChessState = {
  board: (Piece | null)[][];
  turn: PieceColor; // "w" is player (White), "b" is AI (Black)
  moveHistory: Move[];
  capturedByWhite: Piece[];
  capturedByBlack: Piece[];
  isCheck: boolean;
  isCheckmate: boolean;
  isStalemate: boolean;
  statusMessage: string;
  lastMove?: Move;
  aiThought?: string;
};

// Initial standard 8x8 chessboard
export function initialBoard(): (Piece | null)[][] {
  const b: (Piece | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null));

  // Black pieces (row 0 & 1)
  const backRow: PieceType[] = ["r", "n", "b", "q", "k", "b", "n", "r"];
  for (let c = 0; c < 8; c++) {
    b[0][c] = { type: backRow[c], color: "b" };
    b[1][c] = { type: "p", color: "b" };
  }

  // White pieces (row 6 & 7)
  for (let c = 0; c < 8; c++) {
    b[6][c] = { type: "p", color: "w" };
    b[7][c] = { type: backRow[c], color: "w" };
  }

  return b;
}

export function newChessGame(): ChessState {
  return {
    board: initialBoard(),
    turn: "w",
    moveHistory: [],
    capturedByWhite: [],
    capturedByBlack: [],
    isCheck: false,
    isCheckmate: false,
    isStalemate: false,
    statusMessage: "ตาของคุณ (สีขาว) - คลิกเลือกตัวหมากเพื่อเดิน"
  };
}

export function squareToCoords(r: number, c: number): string {
  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];
  return `${files[c]}${ranks[r]}`;
}

export function isInside(r: number, c: number): boolean {
  return r >= 0 && r < 8 && c >= 0 && c < 8;
}

// Pseudo-legal moves for a piece at (r, c)
export function getPieceMoves(board: (Piece | null)[][], from: Square): Square[] {
  const piece = board[from.r][from.c];
  if (!piece) return [];
  const moves: Square[] = [];
  const color = piece.color;
  const oppColor: PieceColor = color === "w" ? "b" : "w";

  const addIfValid = (r: number, c: number): boolean => {
    if (!isInside(r, c)) return false;
    const target = board[r][c];
    if (!target) {
      moves.push({ r, c });
      return true; // continue sliding
    }
    if (target.color === oppColor) {
      moves.push({ r, c });
    }
    return false; // obstacle encountered
  };

  if (piece.type === "p") {
    const dir = color === "w" ? -1 : 1;
    const startRow = color === "w" ? 6 : 1;

    // Move forward 1
    const nextR = from.r + dir;
    if (isInside(nextR, from.c) && !board[nextR][from.c]) {
      moves.push({ r: nextR, c: from.c });
      // Move forward 2 from start
      const nextR2 = from.r + dir * 2;
      if (from.r === startRow && isInside(nextR2, from.c) && !board[nextR2][from.c]) {
        moves.push({ r: nextR2, c: from.c });
      }
    }

    // Diagonal captures
    for (const dc of [-1, 1]) {
      const capC = from.c + dc;
      if (isInside(nextR, capC)) {
        const target = board[nextR][capC];
        if (target && target.color === oppColor) {
          moves.push({ r: nextR, c: capC });
        }
      }
    }
  } else if (piece.type === "n") {
    const jumps = [
      [-2, -1], [-2, 1], [-1, -2], [-1, 2],
      [1, -2], [1, 2], [2, -1], [2, 1]
    ];
    for (const [dr, dc] of jumps) {
      const nr = from.r + dr;
      const nc = from.c + dc;
      if (isInside(nr, nc)) {
        const t = board[nr][nc];
        if (!t || t.color === oppColor) {
          moves.push({ r: nr, c: nc });
        }
      }
    }
  } else if (piece.type === "b" || piece.type === "r" || piece.type === "q") {
    const directions: [number, number][] = [];
    if (piece.type === "b" || piece.type === "q") {
      directions.push([-1, -1], [-1, 1], [1, -1], [1, 1]);
    }
    if (piece.type === "r" || piece.type === "q") {
      directions.push([-1, 0], [1, 0], [0, -1], [0, 1]);
    }

    for (const [dr, dc] of directions) {
      let step = 1;
      while (true) {
        const nr = from.r + dr * step;
        const nc = from.c + dc * step;
        const canContinue = addIfValid(nr, nc);
        if (!canContinue) break;
        step++;
      }
    }
  } else if (piece.type === "k") {
    const dirs = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],           [0, 1],
      [1, -1],  [1, 0],  [1, 1]
    ];
    for (const [dr, dc] of dirs) {
      const nr = from.r + dr;
      const nc = from.c + dc;
      if (isInside(nr, nc)) {
        const t = board[nr][nc];
        if (!t || t.color === oppColor) {
          moves.push({ r: nr, c: nc });
        }
      }
    }
  }

  return moves;
}

// Find king square
export function findKing(board: (Piece | null)[][], color: PieceColor): Square | null {
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p && p.type === "k" && p.color === color) {
        return { r, c };
      }
    }
  }
  return null;
}

// Check if square is attacked by enemy
export function isSquareAttacked(board: (Piece | null)[][], target: Square, byColor: PieceColor): boolean {
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p && p.color === byColor) {
        const moves = getPieceMoves(board, { r, c });
        if (moves.some(m => m.r === target.r && m.c === target.c)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Check if given side's king is in check
export function isKingInCheck(board: (Piece | null)[][], color: PieceColor): boolean {
  const king = findKing(board, color);
  if (!king) return false;
  const oppColor: PieceColor = color === "w" ? "b" : "w";
  return isSquareAttacked(board, king, oppColor);
}

// Filter moves to only those that do NOT leave own king in check
export function getLegalMoves(board: (Piece | null)[][], from: Square): Square[] {
  const piece = board[from.r][from.c];
  if (!piece) return [];
  const pseudo = getPieceMoves(board, from);

  return pseudo.filter(to => {
    // Clone board and test move
    const cloned = board.map(row => [...row]);
    cloned[to.r][to.c] = cloned[from.r][from.c];
    cloned[from.r][from.c] = null;
    return !isKingInCheck(cloned, piece.color);
  });
}

// Get all legal moves for a given side
export function getAllLegalMoves(board: (Piece | null)[][], color: PieceColor): { from: Square; to: Square }[] {
  const allMoves: { from: Square; to: Square }[] = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p && p.color === color) {
        const dests = getLegalMoves(board, { r, c });
        for (const to of dests) {
          allMoves.push({ from: { r, c }, to });
        }
      }
    }
  }
  return allMoves;
}

// Format move notation
export function formatSan(piece: Piece, from: Square, to: Square, captured: boolean, isCheck: boolean): string {
  const dest = squareToCoords(to.r, to.c);
  let str = "";
  if (piece.type === "p") {
    if (captured) {
      const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
      str = `${files[from.c]}x${dest}`;
    } else {
      str = dest;
    }
  } else {
    const symbol = piece.type.toUpperCase();
    str = `${symbol}${captured ? "x" : ""}${dest}`;
  }
  if (isCheck) str += "+";
  return str;
}

// Apply move to state
export function makeMove(state: ChessState, from: Square, to: Square): ChessState {
  const board = state.board.map(row => [...row]);
  const movingPiece = board[from.r][from.c];
  if (!movingPiece || movingPiece.color !== state.turn) return state;

  const captured = board[to.r][to.c] ?? undefined;
  board[to.r][to.c] = movingPiece;
  board[from.r][from.c] = null;

  // Pawn promotion to Queen
  if (movingPiece.type === "p" && (to.r === 0 || to.r === 7)) {
    board[to.r][to.c] = { type: "q", color: movingPiece.color };
  }

  const nextTurn: PieceColor = state.turn === "w" ? "b" : "w";
  const enemyInCheck = isKingInCheck(board, nextTurn);
  const enemyLegalMoves = getAllLegalMoves(board, nextTurn);
  const isCheckmate = enemyInCheck && enemyLegalMoves.length === 0;
  const isStalemate = !enemyInCheck && enemyLegalMoves.length === 0;

  const san = formatSan(movingPiece, from, to, !!captured, enemyInCheck);
  const moveObj: Move = {
    from,
    to,
    piece: movingPiece,
    captured,
    isCheck: enemyInCheck,
    san
  };

  const capturedByWhite = [...state.capturedByWhite];
  const capturedByBlack = [...state.capturedByBlack];
  if (captured) {
    if (state.turn === "w") capturedByWhite.push(captured);
    else capturedByBlack.push(captured);
  }

  let statusMessage = "";
  if (isCheckmate) {
    statusMessage = state.turn === "w" ? "🏆 คุณชนะ! รุกฆาต (Checkmate)" : "🤖 AI ชนะ! รุกฆาต (Checkmate)";
  } else if (isStalemate) {
    statusMessage = "🤝 เสมอกัน! อับจน (Stalemate)";
  } else if (enemyInCheck) {
    statusMessage = nextTurn === "w" ? "⚠️ ตาของคุณ - คุณกำลังโดนรุก (Check!)" : "⚡ คุณรุกเจ้าเมือง AI (Check!)";
  } else {
    statusMessage = nextTurn === "w" ? "ตาของคุณ (สีขาว) - เลือกตัวหมาก" : "ตาของ AI (สีดำ) - กำลังวิเคราะห์...";
  }

  return {
    board,
    turn: nextTurn,
    moveHistory: [...state.moveHistory, moveObj],
    capturedByWhite,
    capturedByBlack,
    isCheck: enemyInCheck,
    isCheckmate,
    isStalemate,
    statusMessage,
    lastMove: moveObj,
    aiThought: state.aiThought
  };
}

// Convert board to simplified position string for threefold repetition detection
export function boardToFen(board: (Piece | null)[][], turn: PieceColor): string {
  let fen = "";
  for (let r = 0; r < 8; r++) {
    let empty = 0;
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (!p) {
        empty++;
      } else {
        if (empty > 0) {
          fen += empty;
          empty = 0;
        }
        fen += p.color === "w" ? p.type.toUpperCase() : p.type.toLowerCase();
      }
    }
    if (empty > 0) fen += empty;
    if (r < 7) fen += "/";
  }
  return `${fen} ${turn}`;
}

// Detect insufficient material to force a draw (K vs K, K+B vs K, K+N vs K)
export function isInsufficientMaterial(board: (Piece | null)[][]): boolean {
  const pieces: Piece[] = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p) pieces.push(p);
    }
  }
  // King vs King
  if (pieces.length === 2) return true;
  // King + minor vs King
  if (pieces.length === 3) {
    return pieces.some(p => p.type === "b" || p.type === "n");
  }
  return false;
}

// Undo / Takeback moves in chess
export function undoChessMove(state: ChessState, steps?: number): ChessState {
  if (state.moveHistory.length === 0) return state;

  const defaultSteps = state.turn === "b" ? 1 : 2;
  const count = Math.min(steps ?? defaultSteps, state.moveHistory.length);
  const targetMoves = state.moveHistory.slice(0, state.moveHistory.length - count);

  let replay = newChessGame();
  for (const m of targetMoves) {
    replay = makeMove(replay, m.from, m.to);
  }

  return {
    ...replay,
    aiThought: targetMoves.length > 0 ? "ย้อนกลับตาเดินเรียบร้อยแล้ว (ถึงตาคุณเดินใหม่)" : undefined
  };
}

export interface ChessModelWeights {
  pieceValues: Record<PieceType, number>;
  centerBonus: number;
  pawnAdvanceBonus: number;
  mobilityBonus: number;
  depth: number;
  trainedGames: number;
}

export const DEFAULT_CHESS_WEIGHTS: ChessModelWeights = {
  pieceValues: { p: 100, n: 320, b: 330, r: 500, q: 900, k: 20000 },
  centerBonus: 25,
  pawnAdvanceBonus: 8,
  mobilityBonus: 5,
  depth: 2,
  trainedGames: 0
};

// Piece values for Minimax evaluation
export const PIECE_VALUES: Record<PieceType, number> = {
  p: 100,
  n: 320,
  b: 330,
  r: 500,
  q: 900,
  k: 20000
};

// Evaluate board from Black (AI) perspective
export function evaluateBoard(board: (Piece | null)[][], weights?: ChessModelWeights): number {
  const pValues = weights?.pieceValues ?? PIECE_VALUES;
  const centerBonus = weights?.centerBonus ?? 25;
  const pawnAdvance = weights?.pawnAdvanceBonus ?? 8;

  let score = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (!p) continue;
      const val = pValues[p.type] ?? PIECE_VALUES[p.type];

      // Positional center bonus (d4, d5, e4, e5)
      let posBonus = 0;
      if ((r === 3 || r === 4) && (c === 3 || c === 4)) posBonus = centerBonus;
      else if (r >= 2 && r <= 5 && c >= 2 && c <= 5) posBonus = Math.round(centerBonus * 0.4);

      // Pawn advancement bonus
      if (p.type === "p") {
        posBonus += p.color === "b" ? r * pawnAdvance : (7 - r) * pawnAdvance;
      }

      const totalVal = val + posBonus;
      if (p.color === "b") score += totalVal;
      else score -= totalVal;
    }
  }
  return score;
}

// Minimax with Alpha-Beta Pruning for Chess AI
export function getChessAiMove(
  state: ChessState,
  difficulty: "baseline" | "thinking" | "master" | "project" = "thinking",
  customWeights?: ChessModelWeights,
  modelName?: string
): { move: { from: Square; to: Square } | null; thought: string } {
  const legalMoves = getAllLegalMoves(state.board, "b");
  if (legalMoves.length === 0) {
    return { move: null, thought: "ไม่มีตาเดินที่ถูกต้อง (จบลูป)" };
  }


  // Baseline: Greedily pick highest capture or center move
  if (difficulty === "baseline") {
    let bestScore = -Infinity;
    let bestMove = legalMoves[0];

    for (const m of legalMoves) {
      const target = state.board[m.to.r][m.to.c];
      let moveVal = target ? PIECE_VALUES[target.type] * 2 : 0;
      // Prefer center
      if ((m.to.r === 3 || m.to.r === 4) && (m.to.c === 3 || m.to.c === 4)) moveVal += 15;
      moveVal += Math.random() * 5;

      if (moveVal > bestScore) {
        bestScore = moveVal;
        bestMove = m;
      }
    }

    const movingPiece = state.board[bestMove.from.r][bestMove.from.c]!;
    const fromStr = squareToCoords(bestMove.from.r, bestMove.from.c);
    const toStr = squareToCoords(bestMove.to.r, bestMove.to.c);

    return {
      move: bestMove,
      thought: `[Baseline AI] เลือกเดิน ${movingPiece.type.toUpperCase()} จาก ${fromStr} ไป ${toStr} ตามการประเมินวัตถุและพื้นที่`
    };
  }

  // Thinking / Master / Project: Minimax Depth search with evaluation
  const depth = difficulty === "master" ? 3 : difficulty === "project" ? (customWeights?.depth ?? 2) : 2;
  let bestScore = -Infinity;
  let bestMove = legalMoves[0];
  let bestReason = "";

  const pValues = customWeights?.pieceValues ?? PIECE_VALUES;

  for (const m of legalMoves) {
    const cloned = state.board.map(row => [...row]);
    const captured = cloned[m.to.r][m.to.c];
    cloned[m.to.r][m.to.c] = cloned[m.from.r][m.from.c];
    cloned[m.from.r][m.from.c] = null;

    // Evaluate white responses (minimizing black score)
    const whiteMoves = getAllLegalMoves(cloned, "w");
    let worstWhiteResponse = Infinity;

    if (whiteMoves.length === 0) {
      worstWhiteResponse = isKingInCheck(cloned, "w") ? 10000 : 0;
    } else {
      for (const wm of whiteMoves) {
        const cloned2 = cloned.map(row => [...row]);
        cloned2[wm.to.r][wm.to.c] = cloned2[wm.from.r][wm.from.c];
        cloned2[wm.from.r][wm.from.c] = null;
        const evalScore = evaluateBoard(cloned2, customWeights);
        if (evalScore < worstWhiteResponse) {
          worstWhiteResponse = evalScore;
        }
      }
    }

    // Add tiny tie-breaker
    const score = worstWhiteResponse + (captured ? (pValues[captured.type] ?? 100) * 0.5 : 0) + Math.random() * 3;

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
      const piece = state.board[m.from.r][m.from.c]!;
      const toStr = squareToCoords(m.to.r, m.to.c);
      if (captured) {
        bestReason = `ตรวจพบโอกาสกินหมาก (${captured.type.toUpperCase()} ที่ ${toStr}) พร้อมคุมพื้นที่สำคัญ`;
      } else if (m.to.r >= 3 && m.to.r <= 4) {
        bestReason = `เดิน ${piece.type.toUpperCase()} เข้าคุมกึ่งกลางกระดาน (${toStr}) เพื่อเพิ่มทางเดินหมาก`;
      } else {
        bestReason = `พัฒนาตัวหมาก ${piece.type.toUpperCase()} ไปยังตำแหน่งยุทธศาสตร์ (${toStr})`;
      }
    }
  }

  const fromStr = squareToCoords(bestMove.from.r, bestMove.from.c);
  const toStr = squareToCoords(bestMove.to.r, bestMove.to.c);
  const pieceName = state.board[bestMove.from.r][bestMove.from.c]?.type.toUpperCase() ?? "";

  const prefix = difficulty === "project"
    ? `[Trained Project: ${modelName || "Custom AI"}] ${customWeights && customWeights.trainedGames > 0 ? `(เทรนแล้ว ${customWeights.trainedGames.toLocaleString()} กระดาน)` : "(ยังไม่ผ่านการเทรน)"}`
    : `[Thinking Minimax D${depth}]`;

  return {
    move: bestMove,
    thought: `${prefix} คำนวณตาเดิน ${pieceName} (${fromStr} ➔ ${toStr}) · ${bestReason} (คะแนนประเมิน: ${(bestScore / 100).toFixed(1)})`
  };
}


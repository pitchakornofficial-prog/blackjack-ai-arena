import { describe, expect, it } from "vitest";
import {
  newChessGame, getLegalMoves, makeMove, getChessAiMove,
  isKingInCheck, squareToCoords, undoChessMove
} from "../src/game/chessEngine";

describe("Chess Engine Initialization & Moves", () => {
  it("initializes a valid 8x8 standard chess board", () => {
    const game = newChessGame();
    expect(game.board).toHaveLength(8);
    expect(game.board[0]).toHaveLength(8);
    expect(game.turn).toBe("w");
    expect(game.board[7][4]?.type).toBe("k"); // White king on e1
    expect(game.board[0][4]?.type).toBe("k"); // Black king on e8
  });

  it("calculates initial pawn moves (1 and 2 steps)", () => {
    const game = newChessGame();
    // e2 pawn is at row 6, col 4
    const moves = getLegalMoves(game.board, { r: 6, c: 4 });
    expect(moves).toHaveLength(2);
    // Can move to e3 (row 5) or e4 (row 4)
    expect(moves.map(m => squareToCoords(m.r, m.c))).toContain("e3");
    expect(moves.map(m => squareToCoords(m.r, m.c))).toContain("e4");
  });

  it("calculates initial knight moves", () => {
    const game = newChessGame();
    // b1 knight is at row 7, col 1
    const moves = getLegalMoves(game.board, { r: 7, c: 1 });
    expect(moves).toHaveLength(2);
    expect(moves.map(m => squareToCoords(m.r, m.c))).toContain("a3");
    expect(moves.map(m => squareToCoords(m.r, m.c))).toContain("c3");
  });

  it("executes a move and alternates turn to black", () => {
    const game = newChessGame();
    // Play 1. e4 (from 6,4 to 4,4)
    const afterE4 = makeMove(game, { r: 6, c: 4 }, { r: 4, c: 4 });
    expect(afterE4.turn).toBe("b");
    expect(afterE4.board[4][4]?.type).toBe("p");
    expect(afterE4.board[6][4]).toBeNull();
    expect(afterE4.moveHistory).toHaveLength(1);
    expect(afterE4.moveHistory[0].san).toBe("e4");
  });

  it("generates an AI response move for black with thinking explanation", () => {
    const game = newChessGame();
    const afterE4 = makeMove(game, { r: 6, c: 4 }, { r: 4, c: 4 });
    const aiResult = getChessAiMove(afterE4, "thinking");
    expect(aiResult.move).not.toBeNull();
    expect(aiResult.thought.length).toBeGreaterThan(10);

    if (aiResult.move) {
      const afterAi = makeMove(afterE4, aiResult.move.from, aiResult.move.to);
      expect(afterAi.turn).toBe("w");
      expect(afterAi.moveHistory).toHaveLength(2);
    }
  });

  it("detects check on king", () => {
    const game = newChessGame();
    // Clear pieces to set up a check: White rook attacking Black king
    const emptyBoard = Array(8).fill(null).map(() => Array(8).fill(null));
    emptyBoard[0][4] = { type: "k", color: "b" }; // Black king on e8
    emptyBoard[7][4] = { type: "r", color: "w" }; // White rook on e1
    emptyBoard[7][0] = { type: "k", color: "w" }; // White king on a1

    expect(isKingInCheck(emptyBoard, "b")).toBe(true);
    expect(isKingInCheck(emptyBoard, "w")).toBe(false);
  });

  it("undos moves correctly (takeback)", () => {
    const game = newChessGame();
    // 1. e4 (White)
    const afterE4 = makeMove(game, { r: 6, c: 4 }, { r: 4, c: 4 });
    // 1... e5 (Black)
    const afterE5 = makeMove(afterE4, { r: 1, c: 4 }, { r: 3, c: 4 });
    expect(afterE5.moveHistory).toHaveLength(2);

    // Undo 2 moves (back to White's initial turn)
    const undid = undoChessMove(afterE5);
    expect(undid.moveHistory).toHaveLength(0);
    expect(undid.turn).toBe("w");
    expect(undid.board[6][4]?.type).toBe("p"); // Pawn back on e2
    expect(undid.board[4][4]).toBeNull();
  });
});

import { describe, it, expect } from "vitest";
import {
  newTTTGame, applyPlayerMove, applyAiMove,
  winnerOf, legalMoves, chooseAiMove, TTTBoard
} from "../src/game/tttEngine";

function b(cells: ("" | "X" | "O")[]): TTTBoard {
  return cells as TTTBoard;
}

describe("TTT engine basics", () => {
  it("new game has empty board and player's turn", () => {
    const s = newTTTGame();
    expect(s.board.every(c => c === "")).toBe(true);
    expect(s.phase).toBe("player");
    expect(s.winner).toBeNull();
  });

  it("detects row win for player", () => {
    const { winner, line } = winnerOf(b(["X", "X", "X", "O", "O", "", "", "", ""]));
    expect(winner).toBe("player");
    expect(line).toEqual([0, 1, 2]);
  });

  it("detects column win for ai", () => {
    const { winner, line } = winnerOf(b(["O", "X", "X", "O", "X", "", "O", "", ""]));
    expect(winner).toBe("ai");
    expect(line).toEqual([0, 3, 6]);
  });

  it("detects diagonal win", () => {
    const { winner } = winnerOf(b(["X", "O", "O", "O", "X", "", "", "", "X"]));
    expect(winner).toBe("player");
  });

  it("full board without a line is a draw", () => {
    const { winner } = winnerOf(b(["X", "O", "X", "X", "O", "O", "O", "X", "X"]));
    expect(winner).toBe("draw");
  });

  it("legalMoves returns only empty cells", () => {
    expect(legalMoves(b(["X", "", "O", "", "X", "", "O", "", ""]))).toEqual([1, 3, 5, 7, 8]);
  });
});

describe("TTT flow", () => {
  it("player move switches to AI, AI answers and switches back", () => {
    let s = applyPlayerMove(newTTTGame(), 4);
    expect(s.board[4]).toBe("X");
    expect(s.phase).toBe("ai");
    s = applyAiMove(s, "unbeatable");
    expect(s.board.filter(c => c === "O").length).toBe(1);
    expect(s.phase).toBe("player");
  });

  it("winning move ends the round with over phase", () => {
    let s = newTTTGame();
    // จัดกระดานให้ X กำลังจะชนะแถวบน
    s = { ...s, board: b(["X", "X", "", "O", "O", "", "", "", ""]), phase: "player", currentTurn: "player" };
    s = applyPlayerMove(s, 2);
    expect(s.winner).toBe("player");
    expect(s.phase).toBe("over");
    expect(s.winLine).toEqual([0, 1, 2]);
  });

  it("ignores clicks on occupied cells or when it is not player's turn", () => {
    let s = newTTTGame();
    s = applyPlayerMove(s, 4);          // X ที่ 4 → ตา AI
    const before = s;
    s = applyPlayerMove(s, 5);          // คลิกตอนเป็นตา AI → ปฏิเสธ
    expect(s).toBe(before);
    s = { ...before, phase: "player", currentTurn: "player" };
    const s2 = applyPlayerMove(s, 4);   // คลิกช่องที่มี X แล้ว → ปฏิเสธ
    expect(s2.board[4]).toBe("X");
    expect(s2.board[5]).toBe("");
  });
});

describe("AI strength", () => {
  it("minimax takes the immediate winning move", () => {
    const { move } = chooseAiMove(b(["O", "O", "", "X", "X", "", "", "", ""]), "unbeatable");
    expect(move).toBe(2);
  });

  it("minimax blocks an immediate human win", () => {
    const { move } = chooseAiMove(b(["X", "X", "", "O", "", "", "", "", ""]), "unbeatable");
    expect(move).toBe(2);
  });

  it("perfect AI never loses in 200 random human games", () => {
    for (let g = 0; g < 200; g++) {
      let s = newTTTGame();
      let safety = 0;
      while (s.phase !== "over" && safety++ < 20) {
        if (s.phase === "player") {
          const moves = legalMoves(s.board);
          const pick = moves[Math.floor(Math.random() * moves.length)];
          s = applyPlayerMove(s, pick);
        } else {
          s = applyAiMove(s, "unbeatable");
        }
      }
      if (s.phase === "over") {
        // ห้ามแพ้ — ชนะหรือเสมอเท่านั้น
        expect(s.winner === "ai" || s.winner === "draw").toBe(true);
      }
    }
  });
});

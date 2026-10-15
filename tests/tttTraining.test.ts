import { describe, expect, it } from "vitest";
import {
  applyProjectAiMove, getProjectMove, initialTTTProjects, knownStatesCount,
  scoresFor, trainTTTBatch
} from "../src/game/tttTraining";
import { legalMoves, lineWinner, minimaxBestMove, newTTTGame, applyPlayerMove, TTTBoard } from "../src/game/tttEngine";

const ALGOS = ["selfplay-rl", "minimax-teacher", "random-drill"] as const;

describe("trainTTTBatch — real self-play", () => {
  it.each(ALGOS)("runs %s batch games with metrics that sum to 100%", algo => {
    const r = trainTTTBatch({}, 200, 0.1, 0.9, 0.3, algo);
    expect(r.batchGames).toBe(200);
    expect(r.winRate + r.lossRate + r.drawRate).toBeCloseTo(100, 5);
    expect(r.winRate).toBeGreaterThanOrEqual(0);
    expect(r.lossRate).toBeGreaterThanOrEqual(0);
    expect(r.drawRate).toBeGreaterThanOrEqual(0);
    expect(r.avgReward).toBeGreaterThanOrEqual(-1);
    expect(r.avgReward).toBeLessThanOrEqual(1);
    expect(r.updatedEpsilon).toBeCloseTo(0.3 * 0.98, 5);
  });

  it("learns something: Q-table gains known states after batches", () => {
    let table: Record<string, number[]> = {};
    for (let i = 0; i < 5; i++) {
      table = trainTTTBatch(table, 100, 0.15, 0.9, 0.3, "selfplay-rl").updatedQTable;
    }
    expect(knownStatesCount(table)).toBeGreaterThan(5);
    // มีสถานะบางสถานะที่ค่าไม่เป็นศูนย์
    const nonZero = Object.values(table).some(row => row.some(v => Math.abs(v) > 1e-6));
    expect(nonZero).toBe(true);
  });

  it("random-drill: trained model crushes a random opponent in evaluation", () => {
    // เทรน 2,000 เกมกับคู่สุ่ม (4 batch × 500) แล้ววัดผลในโหมด Greedy
    let table: Record<string, number[]> = {};
    for (let b = 0; b < 4; b++) {
      table = trainTTTBatch(table, 500, 0.3, 0.9, 0.3, "random-drill").updatedQTable;
    }
    let oWins = 0;
    let xWins = 0;
    for (let i = 0; i < 200; i++) {
      const board: TTTBoard = Array(9).fill("") as TTTBoard;
      let w: "X" | "O" | "draw" | null = null;
      while (true) {
        w = lineWinner(board);
        if (w) break;
        const legal = legalMoves(board);
        if (legal.length === 0) break;
        // X คือคู่แข่งสุ่ม
        board[legal[Math.floor(Math.random() * legal.length)]] = "X";
        const w2 = lineWinner(board);
        if (w2) break;
        const legal2 = legalMoves(board);
        if (legal2.length === 0) break;
        const g = getProjectMove(board, table, 1); // fallback สุ่มเมื่อไม่รู้จักสถานะ
        if (g.move === null) break;
        board[g.move] = "O";
      }
      const final = lineWinner(board);
      if (final === "O") oWins++;
      else if (final === "X") xWins++;
    }
    console.log(`eval: O=${oWins} X=${xWins} draw=${200 - oWins - xWins}`);
    expect(oWins).toBeGreaterThan(xWins);
    expect(oWins).toBeGreaterThan(75);
  });

  it("minimax-teacher: model (O) should rarely beat a perfect teacher", () => {
    const r = trainTTTBatch({}, 100, 0.15, 0.9, 0.3, "minimax-teacher");
    // ครูแพ้ไม่ได้ → ที่พังของ O ควรต่ำมาก (ไม่เกิน ~10%)
    expect(r.winRate).toBeLessThan(15);
  });

});

describe("getProjectMove", () => {
  it("returns a legal move", () => {
    const g = getProjectMove(
      ["X", "", "", "", "O", "", "", "", ""] as TTTBoard,
      {}
    );
    expect(g.move).not.toBeNull();
    expect(g.move!).toBeGreaterThanOrEqual(0);
    expect(g.move!).toBeLessThanOrEqual(8);
  });

  it("plays the best learned cell when the state is known", () => {
    // บอร์ด: 0 ว่างเป็นตา O และ Q-Table บอกว่าช่อง 0 ดีที่สุด
    const boardState: TTTBoard = ["", "X", "O", "X", "O", "", "", "", ""];
    const stateKey = `O|${boardState.join(",")}`;
    const qTable = { [stateKey]: [9, -5, -5, 0, 0, 0, 0, 0, 0] };
    const g = getProjectMove(boardState, qTable);
    expect(g.usedModel).toBe(true);
    expect(g.move).toBe(0);
  });

  it("falls back to minimax/random when the state is unknown", () => {
    const g = getProjectMove(Array(9).fill("") as any, {});
    expect(g.move).not.toBeNull();
    expect(g.usedModel).toBe(false);
  });
});

describe("applyProjectAiMove", () => {
  it("refuses to move on the player's turn", () => {
    const s = newTTTGame();
    const next = applyProjectAiMove(s, {});
    expect(next).toBe(s);
  });

  it("completes the AI move when it is the AI's turn", () => {
    const s0 = applyPlayerMove(newTTTGame(), 4); // กลาง
    const next = applyProjectAiMove(s0, {});
    expect(next.board[4]).toBe("X");
    expect(next.phase).toBe("player");
    expect(next.aiThought.length).toBeGreaterThan(0);
  });
});

describe("minimaxBestMove (X side) — teacher correctness", () => {
  it("blocks an immediate O win threat", () => {
    const board = ["O", "O", "", "", "X", "", "", "", ""] as TTTBoard;
    expect(minimaxBestMove(board, "X")?.move).toBe(2);
  });

  it("takes the winning cell", () => {
    const board = ["X", "X", "", "", "O", "", "", "", "O"] as TTTBoard;
    expect(minimaxBestMove(board, "X")?.move).toBe(2);
  });

  it("cannot run when the board is full", () => {
    expect(minimaxBestMove(["X", "O", "X", "X", "O", "O", "O", "X", "X"] as TTTBoard, "O")).toBeNull();
  });
});

describe("initialTTTProjects", () => {
  it("has two seed projects with valid algorithms", () => {
    expect(initialTTTProjects.length).toBe(2);
    for (const p of initialTTTProjects) {
      expect(ALGOS).toContain(p.algorithm as any);
      expect(scoresFor(p.qTable, "O", newTTTGame().board)).toBeNull();
    }
  });
});

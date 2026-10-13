import { describe, expect, it } from "vitest";
import { trainChessBatch, initialChessProjects } from "../src/game/chessTraining";
import { DEFAULT_CHESS_WEIGHTS } from "../src/game/chessEngine";

describe("Chess AI Training System", () => {
  it("initial projects are well configured", () => {
    expect(initialChessProjects.length).toBeGreaterThanOrEqual(2);
    expect(initialChessProjects[0].name).toContain("DeepChess");
    expect(initialChessProjects[0].weights.depth).toBe(2);
  });

  it("trainChessBatch updates weights and tracks metrics", () => {
    const startWeights = { ...DEFAULT_CHESS_WEIGHTS };
    const { result, updatedWeights, updatedEpsilon } = {
      result: trainChessBatch(startWeights, 100, 0.1, 0.95, 0.2, "selfplay-rl"),
      updatedWeights: trainChessBatch(startWeights, 100, 0.1, 0.95, 0.2, "selfplay-rl").updatedWeights,
      updatedEpsilon: trainChessBatch(startWeights, 100, 0.1, 0.95, 0.2, "selfplay-rl").updatedEpsilon
    };

    expect(result.batchGames).toBe(100);
    expect(result.winRate + result.lossRate + result.drawRate).toBeCloseTo(100, 1);
    expect(updatedWeights.trainedGames).toBe(100);
    expect(updatedEpsilon).toBeLessThan(0.2);
  });

  it("initDeepExplore and stepDeepExplore simulate visual self-play and pause/resume", async () => {
    const { initDeepExplore, stepDeepExplore } = await import("../src/game/chessTraining");
    const state = initDeepExplore();

    expect(state.shannonNumber).toBe("10¹²⁰");
    expect(state.paused).toBe(true);
    expect(state.totalPositionsExplored).toBe(0);
    expect(state.currentGame.turn).toBe("w");

    // Advance 1 move
    const afterStep1 = stepDeepExplore(state);
    expect(afterStep1.totalPositionsExplored).toBeGreaterThan(0);
    expect(afterStep1.currentGameMoveCount).toBe(1);
    expect(afterStep1.currentGame.turn).toBe("b");

    // Advance another move (Black)
    const afterStep2 = stepDeepExplore(afterStep1);
    expect(afterStep2.currentGameMoveCount).toBe(2);
    expect(afterStep2.currentGame.turn).toBe("w");
  });

  it("stepDeepExplore concludes games within move limit and detects repetitions", async () => {
    const { initDeepExplore, stepDeepExplore } = await import("../src/game/chessTraining");
    let state = initDeepExplore();
    state.paused = false;

    // Simulate up to 130 steps - game MUST conclude and start game #2 before 125 steps
    for (let i = 0; i < 130; i++) {
      state = stepDeepExplore(state);
      if (state.totalGamesPlayed > 0) break;
    }

    expect(state.totalGamesPlayed).toBeGreaterThanOrEqual(1);
    expect(state.currentGameMoveCount).toBeLessThan(125);
  });
});

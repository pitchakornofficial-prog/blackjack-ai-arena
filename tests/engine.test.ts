import { describe, expect, it } from "vitest";
import {
  Card, scoreHand, createShoe, newGame, legalActions, startRound,
  applyAction, applyAiAction
} from "../src/game/engine";
import { trainBatch, createQTable, encodeState, getBestAction } from "../src/game/trainer";
import { getAiDecision } from "../src/game/aiAgent";

const cards = (...items: [Card["rank"], Card["suit"]][]): Card[] =>
  items.map(([rank, suit], i) => ({ rank, suit, id: `test-${i}` }));

describe("Blackjack hand scoring", () => {
  it("counts Ace as 11 when possible", () => {
    expect(scoreHand(cards(["A", "♠"], ["6", "♥"]))).toMatchObject({ total: 17, soft: true, bust: false, natural: false });
  });
  it("downgrades Ace from 11 to 1 when needed", () => {
    expect(scoreHand(cards(["A", "♠"], ["6", "♥"], ["10", "♣"]))).toMatchObject({ total: 17, soft: false, bust: false });
  });
  it("counts two Aces as 12", () => {
    expect(scoreHand(cards(["A", "♠"], ["A", "♥"])).total).toBe(12);
  });
  it("recognizes a natural blackjack", () => {
    expect(scoreHand(cards(["A", "♠"], ["K", "♥"])).natural).toBe(true);
  });
  it("does not call a three-card 21 natural", () => {
    expect(scoreHand(cards(["A", "♠"], ["5", "♥"], ["5", "♣"])).natural).toBe(false);
  });
  it("detects bust", () => {
    expect(scoreHand(cards(["10", "♠"], ["8", "♥"], ["5", "♣"])).bust).toBe(true);
  });
});

describe("Deck and action legality", () => {
  it("creates six decks with 312 unique cards", () => {
    const shoe = createShoe(6, () => 0.5);
    expect(shoe).toHaveLength(312);
    expect(new Set(shoe.map(c => c.id)).size).toBe(312);
  });
  it("does not expose actions before a round starts", () => {
    expect(legalActions(newGame())).toEqual([]);
  });
});

describe("Arena Duel Alternating Turns", () => {
  it("initializes duel mode with both player and AI", () => {
    const game = newGame(() => 0.5, "play", "duel");
    expect(game.playMode).toBe("duel");
    expect(game.currentTurn).toBe("player");
  });

  it("deals cards to player, AI, and dealer in duel mode", () => {
    const game = startRound(newGame(() => 0.1, "play", "duel"), 100);
    expect(game.playerHands.length).toBe(1);
    expect(game.aiHand).toBeDefined();
    expect(game.dealer.length).toBe(2);
  });

  it("alternates turn from player to AI when player takes an action", () => {
    // Construct a non-natural state
    let state = newGame(() => 0.2, "play", "duel");
    state = startRound(state, 50);
    if (state.phase === "player") {
      expect(state.currentTurn).toBe("player");
      expect(legalActions(state)).toContain("hit");
      // Player hits
      const afterPlayerHit = applyAction(state, "hit");
      if (!afterPlayerHit.playerHands[0].busted) {
        expect(afterPlayerHit.currentTurn).toBe("ai");
      }
    }
  });

  it("alternates turn from AI back to player after AI acts", () => {
    let state = newGame(() => 0.3, "play", "duel");
    state = startRound(state, 50);
    if (state.phase === "player") {
      // Simulate AI turn
      state.currentTurn = "ai";
      const afterAi = applyAiAction(state, "hit", "AI decides to hit");
      expect(afterAi.currentTurn).toBe("player");
      expect(afterAi.roundMessage).toContain("สลับมาที่ตาของคุณแล้ว");
    }
  });
});

describe("Real Q-Learning Training Simulator", () => {
  it("encodes state and tracks Q-values properly", () => {
    const qTable = createQTable();
    const key = encodeState(16, 10, false);
    expect(key).toBe("16_10_0");
    const action = getBestAction(qTable, key, false);
    expect(["hit", "stand"]).toContain(action);
  });

  it("trains a batch of real episodes and returns valid metrics", () => {
    const qTable = createQTable();
    const { result, updatedEpsilon } = trainBatch(qTable, 100, 0.1, 0.95, 0.2, "qlearning");

    expect(result.episodes).toBe(100);
    expect(result.wins + result.losses + result.pushes).toBe(100);
    expect(result.winRate).toBeGreaterThanOrEqual(0);
    expect(result.winRate).toBeLessThanOrEqual(100);
    expect(typeof result.avgReward).toBe("number");
    expect(updatedEpsilon).toBeLessThanOrEqual(0.2);
    expect(Object.keys(qTable).length).toBeGreaterThan(0);
  });
});

describe("AI Agent Decisions", () => {
  it("provides reasonable basic decisions and thought explanations", () => {
    const state = startRound(newGame(() => 0.4, "play", "duel"), 50);
    if (state.aiHand) {
      const decision = getAiDecision(state, "thinking");
      expect(["hit", "stand", "double"]).toContain(decision.action);
      expect(decision.thought.length).toBeGreaterThan(5);
    }
  });
});

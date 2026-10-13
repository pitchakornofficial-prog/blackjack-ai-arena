import { describe, expect, it } from "vitest";
import { Card, scoreHand, createShoe, newGame, legalActions } from "../src/game/engine";

const cards = (...items: [Card["rank"], Card["suit"]][]): Card[] =>
  items.map(([rank, suit], i) => ({ rank, suit, id: `test-${i}` }));

describe("Blackjack hand scoring", () => {
  it("counts Ace as 11 when possible", () => {
    expect(scoreHand(cards(["A","♠"], ["6","♥"]))).toMatchObject({ total: 17, soft: true, bust: false, natural: false });
  });
  it("downgrades Ace from 11 to 1 when needed", () => {
    expect(scoreHand(cards(["A","♠"], ["6","♥"], ["10","♣"]))).toMatchObject({ total: 17, soft: false, bust: false });
  });
  it("counts two Aces as 12", () => {
    expect(scoreHand(cards(["A","♠"], ["A","♥"])).total).toBe(12);
  });
  it("recognizes a natural blackjack", () => {
    expect(scoreHand(cards(["A","♠"], ["K","♥"])).natural).toBe(true);
  });
  it("does not call a three-card 21 natural", () => {
    expect(scoreHand(cards(["A","♠"], ["5","♥"], ["5","♣"])).natural).toBe(false);
  });
  it("detects bust", () => {
    expect(scoreHand(cards(["10","♠"], ["8","♥"], ["5","♣"])).bust).toBe(true);
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

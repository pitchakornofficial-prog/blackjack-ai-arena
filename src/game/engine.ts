export type Suit = "♠" | "♥" | "♦" | "♣";
export type Rank = "A" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "J" | "Q" | "K";
export type Card = { rank: Rank; suit: Suit; id: string };
export type Action = "hit" | "stand" | "double" | "split";
export type Phase = "ready" | "player" | "dealer" | "settlement" | "over";
export type Hand = { cards: Card[]; bet: number; stood: boolean; busted: boolean; doubled: boolean; fromSplit: boolean; splitAces: boolean; natural: boolean };
export type GameState = {
  shoe: Card[];
  playerHands: Hand[];
  activeHand: number;
  dealer: Card[];
  phase: Phase;
  bankroll: number;
  initialBet: number;
  roundMessage: string;
  mode: "play" | "training";
  lastAction?: Action;
  lastAdvice?: string;
  round: number;
};

const suits: Suit[] = ["♠", "♥", "♦", "♣"];
const ranks: Rank[] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];

export function createShoe(decks = 6, random: () => number = Math.random): Card[] {
  const cards: Card[] = [];
  for (let d = 0; d < decks; d++) {
    for (const suit of suits) for (const rank of ranks) {
      cards.push({ rank, suit, id: `${d}-${suit}-${rank}` });
    }
  }
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }
  return cards;
}

export function scoreHand(cards: Card[]): { total: number; soft: boolean; bust: boolean; natural: boolean } {
  let total = 0;
  let aces = 0;
  for (const card of cards) {
    if (card.rank === "A") { total += 11; aces++; }
    else if (["K", "Q", "J", "10"].includes(card.rank)) total += 10;
    else total += Number(card.rank);
  }
  while (total > 21 && aces > 0) { total -= 10; aces--; }
  const soft = aces > 0 && total <= 21;
  return { total, soft, bust: total > 21, natural: cards.length === 2 && total === 21 };
}

function hand(cards: Card[], bet: number, fromSplit = false): Hand {
  const score = scoreHand(cards);
  return { cards, bet, stood: false, busted: score.bust, doubled: false, fromSplit, splitAces: false, natural: score.natural && !fromSplit };
}

function draw(state: GameState): Card {
  const card = state.shoe.pop();
  if (!card) throw new Error("Shoe is empty");
  return card;
}

export function newGame(random: () => number = Math.random, mode: "play" | "training" = "play"): GameState {
  return {
    shoe: createShoe(6, random), playerHands: [], activeHand: 0, dealer: [],
    phase: "ready", bankroll: 1000, initialBet: 100,
    roundMessage: "พร้อมเริ่มเกม", mode, round: 0
  };
}

export function startRound(state: GameState, bet = state.initialBet): GameState {
  if (state.phase !== "ready" && state.phase !== "over") return state;
  if (bet < 10 || bet > state.bankroll) return { ...state, roundMessage: "เดิมพันต้องอย่างน้อย 10 เครดิตและไม่เกินเครดิตที่มี" };
  let next: GameState = {
    ...state, shoe: state.shoe.length < 60 ? createShoe() : [...state.shoe],
    playerHands: [], dealer: [], activeHand: 0, phase: "player",
    bankroll: state.bankroll - bet, initialBet: bet, roundMessage: "ตาคุณแล้ว",
    lastAction: undefined, lastAdvice: undefined, round: state.round + 1
  };
  const playerCards = [draw(next), draw(next)];
  const dealerCards = [draw(next), draw(next)];
  next.playerHands = [hand(playerCards, bet)];
  next.dealer = dealerCards;

  const playerNatural = scoreHand(playerCards).natural;
  const dealerNatural = scoreHand(dealerCards).natural;
  if (playerNatural || dealerNatural) {
    next.phase = "settlement";
    next = settle(next, true);
  }
  return next;
}

export function legalActions(state: GameState): Action[] {
  if (state.phase !== "player") return [];
  const current = state.playerHands[state.activeHand];
  if (!current || current.stood || current.busted) return [];
  const score = scoreHand(current.cards);
  if (score.bust || score.natural) return [];
  const actions: Action[] = ["hit", "stand"];
  if (current.cards.length === 2 && state.bankroll >= current.bet && !current.splitAces) actions.push("double");
  const sameRank = current.cards.length === 2 && current.cards[0].rank === current.cards[1].rank;
  if (sameRank && state.playerHands.length < 4 && state.bankroll >= current.bet && !current.splitAces) actions.push("split");
  return actions;
}

export function applyAction(state: GameState, action: Action, random?: () => number): GameState {
  if (!legalActions(state).includes(action)) return { ...state, roundMessage: "Action นี้ใช้ไม่ได้ในสถานะปัจจุบัน" };
  const next = { ...state, shoe: [...state.shoe], playerHands: state.playerHands.map(h => ({ ...h, cards: [...h.cards] })), dealer: [...state.dealer], lastAction: action };
  const current = next.playerHands[next.activeHand];
  if (action === "hit") {
    current.cards.push(draw(next));
    const score = scoreHand(current.cards);
    current.busted = score.bust;
    if (score.bust) current.stood = true;
  } else if (action === "stand") {
    current.stood = true;
  } else if (action === "double") {
    next.bankroll -= current.bet;
    current.bet *= 2;
    current.doubled = true;
    current.cards.push(draw(next));
    current.busted = scoreHand(current.cards).bust;
    current.stood = true;
  } else if (action === "split") {
    const [first, second] = current.cards;
    next.bankroll -= current.bet;
    const splitAces = first.rank === "A";
    const left = hand([first, draw(next)], current.bet, true);
    const right = hand([second, draw(next)], current.bet, true);
    left.splitAces = splitAces;
    right.splitAces = splitAces;
    if (splitAces) { left.stood = true; right.stood = true; }
    next.playerHands.splice(next.activeHand, 1, left, right);
  }
  return advance(next, random);
}

function advance(state: GameState, random?: () => number): GameState {
  while (state.activeHand < state.playerHands.length) {
    const h = state.playerHands[state.activeHand];
    if (!h.stood && !h.busted) break;
    state.activeHand++;
  }
  if (state.activeHand >= state.playerHands.length) {
    state.phase = "dealer";
    return playDealer(state, random);
  }
  state.phase = "player";
  state.roundMessage = `มือ ${state.activeHand + 1} จาก ${state.playerHands.length}`;
  return state;
}

function playDealer(state: GameState, random?: () => number): GameState {
  const next = { ...state, dealer: [...state.dealer], shoe: [...state.shoe], phase: "settlement" as Phase };
  const playerCanWin = next.playerHands.some(h => !scoreHand(h.cards).bust);
  if (playerCanWin) {
    while (true) {
      const s = scoreHand(next.dealer);
      if (s.total >= 17 || s.bust) break;
      next.dealer.push(draw(next));
    }
  }
  return settle(next, false);
}

function settle(state: GameState, peeked: boolean): GameState {
  const dealerScore = scoreHand(state.dealer);
  let bankroll = state.bankroll;
  const outcomes = state.playerHands.map(h => {
    const playerScore = scoreHand(h.cards);
    if (playerScore.bust) return "แพ้";
    if (playerScore.natural && !h.fromSplit) {
      if (dealerScore.natural) { bankroll += h.bet; return "เสมอ"; }
      bankroll += h.bet * 2.5;
      return "Blackjack!";
    }
    if (dealerScore.natural) return "แพ้";
    if (dealerScore.bust || playerScore.total > dealerScore.total) { bankroll += h.bet * 2; return "ชนะ"; }
    if (playerScore.total === dealerScore.total) { bankroll += h.bet; return "เสมอ"; }
    return "แพ้";
  });
  const allNaturalHandled = peeked;
  return {
    ...state, bankroll, phase: "over",
    roundMessage: allNaturalHandled ? outcomes[0] : outcomes.map((o, i) => `มือ ${i + 1}: ${o}`).join(" · ")
  };
}

export function basicStrategy(state: GameState): Action {
  const h = state.playerHands[state.activeHand];
  const p = scoreHand(h.cards);
  const up = state.dealer[0];
  const dealer = up.rank === "A" ? 11 : ["K", "Q", "J", "10"].includes(up.rank) ? 10 : Number(up.rank);
  const canDouble = legalActions(state).includes("double");
  const canSplit = legalActions(state).includes("split");
  if (canSplit && ["A", "8"].includes(h.cards[0].rank)) return "split";
  if (p.total >= 17) return "stand";
  if (p.total <= 11) return canDouble && p.total >= 9 && dealer >= 3 && dealer <= 6 ? "double" : "hit";
  if (p.soft) {
    if (p.total >= 19) return "stand";
    if (p.total === 18 && dealer >= 2 && dealer <= 8) return canDouble && dealer >= 3 && dealer <= 6 ? "double" : "stand";
    return canDouble && p.total >= 13 && p.total <= 17 && dealer >= 4 && dealer <= 6 ? "double" : "hit";
  }
  if (p.total >= 13 && dealer <= 6) return "stand";
  if (p.total === 12 && dealer >= 4 && dealer <= 6) return "stand";
  if (p.total === 12 && dealer >= 2 && dealer <= 3) return "hit";
  if (p.total === 16 && dealer >= 9) return "hit";
  if (canDouble && p.total === 10 && dealer <= 9) return "double";
  if (canDouble && p.total === 9 && dealer >= 3 && dealer <= 6) return "double";
  return "hit";
}

export function explainAction(action: Action): string {
  return ({
    hit: "จั่วไพ่เพิ่มเพื่อพยายามเข้าใกล้ 21",
    stand: "หยุดจั่วและให้เจ้ามือเล่นตามกติกา",
    double: "เพิ่มเดิมพันเป็นสองเท่า รับไพ่เพิ่มหนึ่งใบ แล้วหยุด",
    split: "แยกไพ่คู่เป็นสองมือและเล่นแยกกัน"
  })[action];
}

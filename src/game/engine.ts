export type Suit = "♠" | "♥" | "♦" | "♣";
export type Rank = "A" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "J" | "Q" | "K";
export type Card = { rank: Rank; suit: Suit; id: string };
export type Action = "hit" | "stand" | "double" | "split";
export type Phase = "ready" | "player" | "dealer" | "settlement" | "over";
export type Turn = "player" | "ai" | "dealer" | "settlement" | "over";
export type Hand = {
  cards: Card[];
  bet: number;
  stood: boolean;
  busted: boolean;
  doubled: boolean;
  fromSplit: boolean;
  splitAces: boolean;
  natural: boolean;
};

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
  // Arena Duel fields
  playMode: "duel" | "solo";
  aiHand?: Hand;
  aiBankroll: number;
  currentTurn: Turn;
  aiLastAction?: string;
  aiThought?: string;
  duelMessage?: string;
  duelWinner?: "player" | "ai" | "dealer" | "tie";
};

const suits: Suit[] = ["♠", "♥", "♦", "♣"];
const ranks: Rank[] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];

export function createShoe(decks = 6, random: () => number = Math.random): Card[] {
  const cards: Card[] = [];
  for (let d = 0; d < decks; d++) {
    for (const suit of suits) {
      for (const rank of ranks) {
        cards.push({ rank, suit, id: `${d}-${suit}-${rank}` });
      }
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
    if (card.rank === "A") {
      total += 11;
      aces++;
    } else if (["K", "Q", "J", "10"].includes(card.rank)) {
      total += 10;
    } else {
      total += Number(card.rank);
    }
  }
  while (total > 21 && aces > 0) {
    total -= 10;
    aces--;
  }
  const soft = aces > 0 && total <= 21;
  return { total, soft, bust: total > 21, natural: cards.length === 2 && total === 21 };
}

export function createHand(cards: Card[], bet: number, fromSplit = false): Hand {
  const score = scoreHand(cards);
  return {
    cards,
    bet,
    stood: false,
    busted: score.bust,
    doubled: false,
    fromSplit,
    splitAces: false,
    natural: score.natural && !fromSplit
  };
}

function draw(state: GameState): Card {
  const card = state.shoe.pop();
  if (!card) throw new Error("Shoe is empty");
  return card;
}

export function newGame(
  random: () => number = Math.random,
  mode: "play" | "training" = "play",
  playMode: "duel" | "solo" = "duel"
): GameState {
  return {
    shoe: createShoe(6, random),
    playerHands: [],
    activeHand: 0,
    dealer: [],
    phase: "ready",
    bankroll: 1000,
    initialBet: 100,
    roundMessage: playMode === "duel" ? "พร้อมเริ่มการดวล กด DEAL CARDS เพื่อเริ่มรอบ (สลับกันเล่นกับ AI)" : "พร้อมเริ่มเกม",
    mode,
    round: 0,
    playMode,
    aiBankroll: 1000,
    currentTurn: "player"
  };
}

export function startRound(state: GameState, bet = state.initialBet): GameState {
  if (state.phase !== "ready" && state.phase !== "over") return state;
  if (bet < 10 || bet > state.bankroll) {
    return { ...state, roundMessage: "เดิมพันต้องอย่างน้อย 10 เครดิตและไม่เกินเครดิตที่มี" };
  }

  let next: GameState = {
    ...state,
    shoe: state.shoe.length < 60 ? createShoe() : [...state.shoe],
    playerHands: [],
    aiHand: undefined,
    dealer: [],
    activeHand: 0,
    phase: "player",
    currentTurn: "player",
    bankroll: state.bankroll - bet,
    initialBet: bet,
    roundMessage: state.playMode === "duel" ? "ตาของคุณแล้ว: เลือกคำสั่ง (จะสลับกับตาของ AI)" : "ตาคุณแล้ว",
    lastAction: undefined,
    lastAdvice: undefined,
    aiLastAction: undefined,
    aiThought: undefined,
    duelMessage: undefined,
    duelWinner: undefined,
    round: state.round + 1
  };

  const playerCards = [draw(next), draw(next)];
  next.playerHands = [createHand(playerCards, bet)];

  if (state.playMode === "duel") {
    const aiCards = [draw(next), draw(next)];
    next.aiHand = createHand(aiCards, bet);
    next.aiBankroll = Math.max(0, next.aiBankroll - bet);
  }

  const dealerCards = [draw(next), draw(next)];
  next.dealer = dealerCards;

  const playerNatural = scoreHand(playerCards).natural;
  const dealerNatural = scoreHand(dealerCards).natural;
  const aiNatural = next.aiHand ? scoreHand(next.aiHand.cards).natural : false;

  if (playerNatural || dealerNatural || (state.playMode === "duel" && aiNatural)) {
    next.phase = "settlement";
    next.currentTurn = "settlement";
    next = settle(next, true);
  }

  return next;
}

export function legalActions(state: GameState): Action[] {
  if (state.phase !== "player" || state.currentTurn !== "player") return [];
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
  if (!legalActions(state).includes(action)) {
    return { ...state, roundMessage: "Action นี้ใช้ไม่ได้ในสถานะปัจจุบัน" };
  }

  const next: GameState = {
    ...state,
    shoe: [...state.shoe],
    playerHands: state.playerHands.map(h => ({ ...h, cards: [...h.cards] })),
    dealer: [...state.dealer],
    lastAction: action
  };

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
    const left = createHand([first, draw(next)], current.bet, true);
    const right = createHand([second, draw(next)], current.bet, true);
    left.splitAces = splitAces;
    right.splitAces = splitAces;
    if (splitAces) {
      left.stood = true;
      right.stood = true;
    }
    next.playerHands.splice(next.activeHand, 1, left, right);
  }

  if (next.playMode === "duel") {
    return advanceDuelAfterPlayer(next, random);
  }

  return advance(next, random);
}

function advanceDuelAfterPlayer(state: GameState, random?: () => number): GameState {
  const currentHand = state.playerHands[state.activeHand];
  const playerFinished = !currentHand || currentHand.stood || currentHand.busted;
  const aiHand = state.aiHand;
  const aiFinished = !aiHand || aiHand.stood || aiHand.busted;

  if (!aiFinished) {
    state.currentTurn = "ai";
    state.roundMessage = "ตาของ AI คู่แข่ง (กำลังวิเคราะห์ตาเดิน...)";
    return state;
  }

  // AI is already finished
  if (playerFinished) {
    // Check if more player hands (e.g. split)
    let nextIdx = state.activeHand + 1;
    while (nextIdx < state.playerHands.length) {
      const h = state.playerHands[nextIdx];
      if (!h.stood && !h.busted) break;
      nextIdx++;
    }
    if (nextIdx < state.playerHands.length) {
      state.activeHand = nextIdx;
      state.currentTurn = "player";
      state.roundMessage = `มือที่ ${nextIdx + 1} ของคุณ`;
      return state;
    }
    // Both done -> dealer plays
    state.phase = "dealer";
    state.currentTurn = "dealer";
    return playDealer(state, random);
  }

  // Player not finished and AI finished
  state.currentTurn = "player";
  state.roundMessage = "AI จบตาแล้ว · ตาของคุณต่อ";
  return state;
}

export function applyAiAction(
  state: GameState,
  action: "hit" | "stand" | "double",
  thought?: string,
  random?: () => number
): GameState {
  if (state.playMode !== "duel" || !state.aiHand) return state;

  const next: GameState = {
    ...state,
    shoe: [...state.shoe],
    aiHand: { ...state.aiHand, cards: [...state.aiHand.cards] },
    aiLastAction: action,
    aiThought: thought
  };

  const ai = next.aiHand!;

  if (action === "hit") {
    ai.cards.push(draw(next));
    const score = scoreHand(ai.cards);
    ai.busted = score.bust;
    if (score.bust) ai.stood = true;
  } else if (action === "stand") {
    ai.stood = true;
  } else if (action === "double") {
    next.aiBankroll -= ai.bet;
    ai.bet *= 2;
    ai.doubled = true;
    ai.cards.push(draw(next));
    ai.busted = scoreHand(ai.cards).bust;
    ai.stood = true;
  }

  const currentHand = next.playerHands[next.activeHand];
  const playerFinished = !currentHand || currentHand.stood || currentHand.busted;
  const aiFinished = ai.stood || ai.busted;

  if (!playerFinished) {
    next.currentTurn = "player";
    const actionTh = action === "hit" ? "HIT" : action === "stand" ? "STAND" : "DOUBLE";
    next.roundMessage = `AI เลือก ${actionTh} -> สลับมาที่ตาของคุณแล้ว!`;
    return next;
  }

  // Player is finished
  if (!aiFinished) {
    next.currentTurn = "ai";
    next.roundMessage = "คุณจบตาแล้ว · AI กำลังเล่นต่อ...";
    return next;
  }

  // Both finished -> dealer plays
  next.phase = "dealer";
  next.currentTurn = "dealer";
  return playDealer(next, random);
}

function advance(state: GameState, random?: () => number): GameState {
  while (state.activeHand < state.playerHands.length) {
    const h = state.playerHands[state.activeHand];
    if (!h.stood && !h.busted) break;
    state.activeHand++;
  }
  if (state.activeHand >= state.playerHands.length) {
    state.phase = "dealer";
    state.currentTurn = "dealer";
    return playDealer(state, random);
  }
  state.phase = "player";
  state.currentTurn = "player";
  state.roundMessage = `มือ ${state.activeHand + 1} จาก ${state.playerHands.length}`;
  return state;
}

export function playDealer(state: GameState, random?: () => number): GameState {
  const next: GameState = {
    ...state,
    dealer: [...state.dealer],
    shoe: [...state.shoe],
    phase: "settlement" as Phase,
    currentTurn: "settlement" as Turn
  };

  const playerCanWin = next.playerHands.some(h => !scoreHand(h.cards).bust);
  const aiCanWin = next.aiHand ? !scoreHand(next.aiHand.cards).bust : false;

  if (playerCanWin || aiCanWin) {
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
  let aiBankroll = state.aiBankroll;

  // Settle player hands
  const outcomes = state.playerHands.map(h => {
    const playerScore = scoreHand(h.cards);
    if (playerScore.bust) return "แพ้ (Bust)";
    if (playerScore.natural && !h.fromSplit) {
      if (dealerScore.natural) {
        bankroll += h.bet;
        return "เสมอ (Blackjack)";
      }
      bankroll += h.bet * 2.5;
      return "ชนะ (Natural Blackjack!)";
    }
    if (dealerScore.natural) return "แพ้ (เจ้ามือ Blackjack)";
    if (dealerScore.bust || playerScore.total > dealerScore.total) {
      bankroll += h.bet * 2;
      return "ชนะ";
    }
    if (playerScore.total === dealerScore.total) {
      bankroll += h.bet;
      return "เสมอ";
    }
    return "แพ้";
  });

  // Settle AI hand if in duel
  let aiOutcome = "";
  let duelWinner: "player" | "ai" | "dealer" | "tie" = "tie";
  let duelMessage = "";

  if (state.playMode === "duel" && state.aiHand) {
    const aiHand = state.aiHand;
    const aiScore = scoreHand(aiHand.cards);
    const playerScore = scoreHand(state.playerHands[0]?.cards ?? []);

    if (aiScore.bust) {
      aiOutcome = "แพ้ (Bust)";
    } else if (aiScore.natural) {
      if (dealerScore.natural) {
        aiBankroll += aiHand.bet;
        aiOutcome = "เสมอ (Blackjack)";
      } else {
        aiBankroll += aiHand.bet * 2.5;
        aiOutcome = "ชนะ (Blackjack!)";
      }
    } else if (dealerScore.natural) {
      aiOutcome = "แพ้";
    } else if (dealerScore.bust || aiScore.total > dealerScore.total) {
      aiBankroll += aiHand.bet * 2;
      aiOutcome = "ชนะ";
    } else if (aiScore.total === dealerScore.total) {
      aiBankroll += aiHand.bet;
      aiOutcome = "เสมอ";
    } else {
      aiOutcome = "แพ้";
    }

    // Determine Duel winner between Player and AI
    const pBust = playerScore.bust;
    const aBust = aiScore.bust;
    if (pBust && aBust) {
      duelWinner = "dealer";
      duelMessage = "ทั้งคุณและ AI แพ้ Bust ทั้งคู่ เจ้ามือกวาดโต๊ะ!";
    } else if (pBust && !aBust) {
      duelWinner = "ai";
      duelMessage = `AI ชนะการดวล! (คุณ Bust · AI ได้ ${aiScore.total} แต้ม)`;
    } else if (!pBust && aBust) {
      duelWinner = "player";
      duelMessage = `คุณชนะการดวล! (คุณได้ ${playerScore.total} แต้ม · AI Bust)`;
    } else if (playerScore.total > aiScore.total) {
      duelWinner = "player";
      duelMessage = `คุณชนะการดวล! (${playerScore.total} แต้ม vs AI ${aiScore.total} แต้ม)`;
    } else if (aiScore.total > playerScore.total) {
      duelWinner = "ai";
      duelMessage = `AI ชนะการดวล! (AI ${aiScore.total} แต้ม vs คุณ ${playerScore.total} แต้ม)`;
    } else {
      duelWinner = "tie";
      duelMessage = `เสมอกันในการดวล! (ทั้งคู่ได้ ${playerScore.total} แต้ม)`;
    }
  }

  const pSummary = outcomes.join(" · ");
  let finalRoundMessage = pSummary;
  if (state.playMode === "duel" && duelMessage) {
    finalRoundMessage = `${duelMessage} [คุณ: ${pSummary} | AI: ${aiOutcome}]`;
  }

  return {
    ...state,
    bankroll,
    aiBankroll,
    phase: "over",
    currentTurn: "over",
    roundMessage: finalRoundMessage,
    duelMessage,
    duelWinner
  };
}

export function basicStrategy(state: GameState): Action {
  const h = state.playerHands[state.activeHand];
  if (!h) return "stand";
  const p = scoreHand(h.cards);
  const up = state.dealer[0];
  const dealer = up ? (up.rank === "A" ? 11 : ["K", "Q", "J", "10"].includes(up.rank) ? 10 : Number(up.rank)) : 10;
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

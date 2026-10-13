# Blackjack AI Arena — Prototype v0.2

A responsive monochrome Blackjack web app with separate workspace screens.

## Run locally
Requires Node.js 20+.
```bash
npm install
npm run dev
```
Open the local URL printed by Vite.

## Test and build
```bash
npm test
npm run build
```

## Pages included
- Overview dashboard
- Play Game: choose Baseline AI, Thinking AI, or Project AI
- AI Training: choose an agent project, opponent type, and training duration (1, 5, 10, 30, 60 minutes)
- AI Projects: create and view training projects
- How to Play & Rules
- Play Assist / Basic Strategy hint

## Important prototype notes
- AI training progress and simulated-hand counts are UI demonstrations only. No Reinforcement Learning model is trained yet.
- Projects are held in React state and do not persist after a browser refresh.
- Thinking AI and Project AI are integration placeholders; only the basic local game and rough strategy helper are present.
- The current Basic Strategy helper is an initial approximation, not a validated complete chart.
- The game engine needs more thorough testing for split-hand sequencing, bankroll settlement and dealer peek edge cases before production use.
- Simulated credits only; no real-money gambling or payment integration.

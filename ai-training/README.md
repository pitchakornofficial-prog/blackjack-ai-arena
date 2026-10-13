# AI training plan

The web app currently includes a small Basic Strategy baseline for interactive hints. It is intentionally not presented as a fully validated optimal strategy table.

Next training milestone:
1. Implement a Python environment matching `src/game/engine.ts` exactly.
2. Add action masking for hit, stand, double, and split.
3. Establish a Basic Strategy benchmark under the same rules.
4. Train Q-learning on a simplified state representation first.
5. Evaluate on held-out random seeds and report expected return, variance, and confidence intervals.

Do not compare agents trained under different deck, payout, or dealer-soft-17 rules.

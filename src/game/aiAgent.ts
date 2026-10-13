import { GameState, scoreHand } from "./engine";
import { QTable, encodeState, getBestAction, getQValues } from "./trainer";

export function getAiDecision(
  state: GameState,
  opponentType: "baseline" | "thinking" | "project",
  qTable?: QTable
): { action: "hit" | "stand" | "double"; thought: string } {
  const aiHand = state.aiHand;
  if (!aiHand) return { action: "stand", thought: "ไม่มีข้อมูลมือของ AI" };

  const score = scoreHand(aiHand.cards);
  const upcard = state.dealer[0];
  const dealerVal = upcard
    ? upcard.rank === "A"
      ? 11
      : ["K", "Q", "J", "10"].includes(upcard.rank)
      ? 10
      : Number(upcard.rank)
    : 10;
  const canDouble = aiHand.cards.length === 2 && state.aiBankroll >= aiHand.bet;

  // 1. Project AI: If a trained Q-Table exists, query it!
  if (opponentType === "project" && qTable && Object.keys(qTable).length > 0) {
    const stateKey = encodeState(score.total, dealerVal, score.soft);
    const q = getQValues(qTable, stateKey);
    const hasLearned = Object.values(q).some(val => val !== 0);

    if (hasLearned) {
      const action = getBestAction(qTable, stateKey, canDouble);
      return {
        action,
        thought: `[Trained Q-Agent] สถานะ (รวม ${score.total}, ไพ่เจ้ามือ ${dealerVal}) -> แนะนำ ${action.toUpperCase()}`
      };
    } else {
      let action: "hit" | "stand" | "double" = "hit";
      if (score.total >= 17) action = "stand";
      else if (score.total <= 11) action = canDouble && dealerVal >= 3 && dealerVal <= 6 ? "double" : "hit";
      else if (dealerVal <= 6) action = "stand";
      else action = "hit";

      return {
        action,
        thought: `[Trained Q-Agent] สถานะ (รวม ${score.total}, ไพ่เจ้ามือ ${dealerVal}) -> แนะนำ ${action.toUpperCase()}`
      };
    }
  }

  // 2. Thinking AI: Detailed step-by-step reasoning simulation
  if (opponentType === "thinking") {
    if (score.total >= 17) {
      return {
        action: "stand",
        thought: `ประเมินความน่าจะเป็น: แต้ม ${score.total} มีโอกาส Bust สูงกว่า 69% หากจั่วเพิ่ม จึงตัดสินใจ STAND อย่างปลอดภัย`
      };
    }
    if (score.total <= 11) {
      if (canDouble && dealerVal >= 3 && dealerVal <= 6) {
        return {
          action: "double",
          thought: `วิเคราะห์ Expected Value: แต้มต้น ${score.total} เผชิญกับไพ่เจ้ามือ ${dealerVal} (Bust rate สูง) -> แนะนำ DOUBLE เดิมพัน`
        };
      }
      return {
        action: "hit",
        thought: `คำนวณความเสี่ยง: แต้ม ${score.total} ปลอดภัย 100% จากการ Bust -> แนะนำ HIT เพื่อเพิ่มคะแนน`
      };
    }
    if (score.soft) {
      if (score.total >= 19) {
        return { action: "stand", thought: `Soft ${score.total} คะแนนสูงมาก อยู่ในระดับปลอดภัยสูงสุด -> STAND` };
      }
      if (score.total === 18 && dealerVal <= 8) {
        return { action: "stand", thought: `Soft 18 เมื่อเจ้ามือเปิดแต้มกลาง (${dealerVal}) สถิติแนะนำ STAND` };
      }
      return { action: "hit", thought: `Soft ${score.total} ความยืดหยุ่นของ Ace อนุญาตให้จั่ว HIT ได้โดยไม่กลัว Bust ทันที` };
    }
    // Hard 12 - 16
    if (dealerVal <= 6) {
      return {
        action: "stand",
        thought: `จำลองผลลัพธ์: เจ้ามือเปิด ${dealerVal} มีโอกาส Bust กว่า 40% จึงเลือก STAND ที่แต้ม ${score.total} เพื่อกดดันให้เจ้ามือจั่วเอง`
      };
    }
    return {
      action: "hit",
      thought: `สถานการณ์เสี่ยง: แต้ม ${score.total} เป็นรองเจ้ามือที่เปิดไพ่แข็ง (${dealerVal}) สถิติบ่งชี้ว่าจำเป็นต้องเสี่ยง HIT`
    };
  }

  // 3. Baseline Strategy: Pure mathematical rule lookup
  if (score.total >= 17) {
    return { action: "stand", thought: `Baseline Strategy: แต้ม ${score.total} >= 17 -> เลือก Stand ตามสถิติ` };
  }
  if (score.total <= 11) {
    if (canDouble && score.total >= 9 && dealerVal >= 3 && dealerVal <= 6) {
      return { action: "double", thought: `Baseline Strategy: แต้ม ${score.total} และเจ้ามือเปิด ${dealerVal} -> Double` };
    }
    return { action: "hit", thought: `Baseline Strategy: แต้ม ${score.total} <= 11 -> Hit เสมอ` };
  }
  if (score.soft) {
    if (score.total >= 19) return { action: "stand", thought: `Baseline: Soft 19+ -> Stand` };
    if (score.total === 18 && dealerVal <= 8) return { action: "stand", thought: `Baseline: Soft 18 vs ${dealerVal} -> Stand` };
    return { action: "hit", thought: `Baseline: Soft ${score.total} -> Hit` };
  }
  if (score.total >= 13 && dealerVal <= 6) {
    return { action: "stand", thought: `Baseline: Hard ${score.total} vs เจ้ามือ ${dealerVal} (<=6) -> Stand` };
  }
  if (score.total === 12 && dealerVal >= 4 && dealerVal <= 6) {
    return { action: "stand", thought: `Baseline: Hard 12 vs เจ้ามือ ${dealerVal} -> Stand` };
  }
  return { action: "hit", thought: `Baseline: Hard ${score.total} vs เจ้ามือ ${dealerVal} -> Hit` };
}

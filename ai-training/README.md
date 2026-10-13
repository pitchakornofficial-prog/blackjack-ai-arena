# AI Training Engine — implementation notes

## Engine Selection (v0.5)
ผู้ใช้งานสามารถเลือกรูปแบบเอนจินการประมวลผลได้อย่างใดอย่างหนึ่ง (Mutually Exclusive):

1. **Local RL Engine (CPU / GPU)**:
   - เหมาะสำหรับการฝึก Reinforcement Learning แบบ Tabular Q-Learning หรือ Deep Q-Network (DQN)
   - มีระบบจำลองการเล่น Blackjack จริงในเบราว์เซอร์ เพื่อคำนวณ Q-Values และสถิติ
   - ตัวเลือกระบบประมวลผล: Auto / CPU / GPU
   - อัปเดตไฮเปอร์พารามิเตอร์: Learning rate (α), Discount factor (γ), Exploration rate (ε)

2. **Cloud AI API (Google Gemini)**:
   - เหมาะสำหรับการใช้ LLM ในการวิเคราะห์กลยุทธ์ อธิบายการตัดสินใจ หรือจำลองคู่แข่งอัจฉริยะ
   - ไม่ต้องใช้ชิป GPU ในเครื่องในการคำนวณ
   - รองรับการเลือกโมเดล (Gemini 2.5 Flash, Gemini 2.5 Pro, ฯลฯ) พร้อมช่องกรอก API Key และปุ่มทดสอบการเชื่อมต่อ

## Real Simulation & Metrics Chart
- ระหว่างที่การฝึกดำเนินไป ระบบจะรัน `trainBatch` จำลองมือ Blackjack จริงในเบราว์เซอร์
- กราฟ SVG จะคำนวณและพล็อตจุดตามข้อมูลจริง (Real Data Points) เช่น:
  - อัตราชนะ (Win Rate %)
  - รางวัลเฉลี่ยต่อมือ (Average Reward)
  - ค่าความน่าจะเป็นในการสำรวจ (Exploration ε)
- สามารถสลับดูกราฟแต่ละมิติได้แบบ Real-time พร้อมแสดงสถิติจำนวนมือที่จำลองไปแล้วจริง

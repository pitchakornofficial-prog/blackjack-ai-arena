# Blackjack AI Arena

A monochrome React + TypeScript + Vite prototype for Blackjack play, strategy assistance, and AI project management.

## Run the web UI
```bash
npm install
npm run dev
```

## Checks
```bash
npm test
npm run build
```

## Current capabilities
- **Arena Duel Mode (สลับเทิร์นกับ AI)**: ผู้เล่นและ AI คู่แข่ง (Baseline / Thinking / Project AI) นั่งคู่กันบนโต๊ะและผลัดกันเล่นคนละตา (Alternating Turns) พร้อมคำอธิบายเหตุผลของ AI แบบ Real-time
- **Unified Engine Selection (เลือกเอนจินประมวลผล)**: เลือกระหว่าง Local RL Engine (CPU/GPU ในเครื่อง) หรือ Cloud AI API (Google Gemini) อย่างใดอย่างหนึ่งตามต้องการ
- **Real-Time Simulation & Dynamic Chart (กราฟจริง)**: เครื่องจำลองการฝึกเล่น Blackjack จริงในเบราว์เซอร์ คำนวณค่า Q-Table และพล็อตกราฟอัตราชนะ (Win Rate %), รางวัลเฉลี่ย (Average Reward) และ Exploration Rate (ε) สดๆ ระหว่างฝึก
- **Accurate Duration Timer**: เลือกระยะเวลาฝึก 1, 5, 10, 30, 60 นาที ได้อย่างแม่นยำ (เลือก 1 นาทีจะตั้งเวลา 60 วินาทีเป๊ะ และซิงค์กับโปรเจกต์)
- **Separate Views**: Overview, Play Arena, AI Training, AI Projects, Rules, and Play Assist.

## v0.5 Updates
1. **แก้ปัญหาตั้งเวลาฝึก 1 นาทีแต่กลายเป็น 10 นาที**: ปรับแก้ state synchronization ให้การเลือกระยะเวลาฝึกซิงค์กับทั้งตัวจับเวลาและโปรเจกต์อย่างถูกต้อง
2. **รวมการเลือกเอนจิน Local RL (CPU/GPU) กับ Cloud AI API**: รวมเป็นตัวเลือกแบบ Mutually Exclusive ให้ผู้ใช้งานเลือกโหมดใดโหมดหนึ่งได้ชัดเจน
3. **ระบบกราฟข้อมูลจริง (Real Metrics Chart)**: เปลี่ยนจากกราฟตัวอย่างแบบจำลอง มาเป็นกราฟ Dynamic SVG ที่คำนวณและวาดจุดตามการจำลองการฝึกเล่นจริงในเบราว์เซอร์
4. **โหมดผลัดกันเล่นใน Arena (Alternating Turns)**: เพิ่มระบบโต๊ะดวลที่ผู้เล่นและ AI สลับกันเล่นคนละตา แก้ปัญหาที่ผู้เล่นเคยกด Hit ได้ฝ่ายเดียวอย่างต่อเนื่อง

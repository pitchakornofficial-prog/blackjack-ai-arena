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
- **Local Training (ฝึกในเครื่อง)**: ฝึก Blackjack และ Chess AI ในเครื่อง โดยเกมหมากรุกจะเล่นเกมจริงด้วย legal moves บน Web Worker; ตัวอัปเดตโมเดลหลักใช้ CPU ส่วน GPU/WebGL เป็นตัวเร่งงานประเมินเสริม ไม่ใช่การฝึกโมเดลทั้งหมดบน GPU วิธีเทรนถูกล็อกไว้กับโปรเจกต์หลังสร้าง และไม่มีการเรียก Cloud AI API
- **Real-Time Simulation & Dynamic Chart (กราฟจริง)**: เครื่องจำลองการฝึกเล่น Blackjack จริงในเบราว์เซอร์ คำนวณค่า Q-Table และพล็อตกราฟอัตราชนะ (Win Rate %), รางวัลเฉลี่ย (Average Reward) และ Exploration Rate (ε) สดๆ ระหว่างฝึก
- **Accurate Duration Timer**: เลือกระยะเวลาฝึก 1, 5, 10, 15, 30 หรือ 60 นาที; เวลาและความคืบหน้าซิงค์กับโปรเจกต์ที่เลือก
- **Automatic Training Saves**: แยกข้อมูลฝึกตามเกมและบันทึกโปรเจกต์ Blackjack/Chess/Tic-Tac-Toe, สถานะ, เวลา, โมเดล และกราฟลงใน browser storage อัตโนมัติ รวมถึงเมื่อหยุดฝึกก่อนครบเวลา
- **Training Project Management**: แก้ไขชื่อและเวลาเป้าหมาย ลบโปรเจกต์ และฝึกต่อจากโมเดลเดิมได้ทุกเกม พร้อมเลือกโปรเจกต์ AI ที่จะใช้เล่นใน Arena
- **Separate Views**: Overview, Play Arena, AI Training, AI Projects, Rules, and Play Assist.

## v0.5 Updates
1. **แก้ปัญหาตั้งเวลาฝึก 1 นาทีแต่กลายเป็น 10 นาที**: ปรับแก้ state synchronization ให้การเลือกระยะเวลาฝึกซิงค์กับทั้งตัวจับเวลาและโปรเจกต์อย่างถูกต้อง
2. **ฝึก Blackjack และ Chess ในเครื่อง**: เลือก CPU หรือ GPU ที่รองรับและวิธีฝึกขณะสร้างโปรเจกต์; การตั้งค่ารูปแบบการฝึกถูกล็อกหลังสร้าง ข้อมูลแต่ละเกมแยกจากกันและบันทึกอัตโนมัติ รวมถึงเมื่อหยุดก่อนครบเวลา
3. **ระบบกราฟข้อมูลจริง (Real Metrics Chart)**: เปลี่ยนจากกราฟตัวอย่างแบบจำลอง มาเป็นกราฟ Dynamic SVG ที่คำนวณและวาดจุดตามการจำลองการฝึกเล่นจริงในเบราว์เซอร์
4. **โหมดผลัดกันเล่นใน Arena (Alternating Turns)**: เพิ่มระบบโต๊ะดวลที่ผู้เล่นและ AI สลับกันเล่นคนละตา แก้ปัญหาที่ผู้เล่นเคยกด Hit ได้ฝ่ายเดียวอย่างต่อเนื่อง

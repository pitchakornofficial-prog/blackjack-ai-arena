import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft, ArrowRight, BrainCircuit, BookOpen, CircleHelp, Clock3, Code2,
  Cpu, FolderKanban, Gauge, House, Info, Play, Plus, RotateCcw, Settings2,
  ShieldCheck, Sparkles, Swords, Trophy, WandSparkles, X
} from "lucide-react";
import {
  Action, Card as PlayingCard, GameState, applyAction, basicStrategy,
  explainAction, legalActions, newGame, scoreHand, startRound
} from "./game/engine";

type Page = "home" | "play" | "training" | "projects" | "rules" | "help";
type Opponent = "baseline" | "thinking" | "project";
type TrainingProject = { id: number; name: string; minutes: number; status: "Ready" | "Training" | "Complete"; progress: number; games: number; created: string };

function Card({ card, hidden = false }: { card?: PlayingCard; hidden?: boolean }) {
  if (hidden || !card) return <div className="playing-card face-down"><span>♠</span><small>ARENA</small></div>;
  const red = card.suit === "♥" || card.suit === "♦";
  return <div className={`playing-card ${red ? "red-suit" : ""}`}><strong>{card.rank}</strong><span>{card.suit}</span><small>{card.rank}</small></div>;
}

const initialProjects: TrainingProject[] = [
  { id: 1, name: "Baseline Challenger", minutes: 10, status: "Ready", progress: 0, games: 0, created: "ตัวอย่างโปรเจกต์" }
];

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [game, setGame] = useState<GameState>(() => newGame());
  const [bet, setBet] = useState(100);
  const [opponent, setOpponent] = useState<Opponent>("baseline");
  const [showHelp, setShowHelp] = useState(false);
  const [projects, setProjects] = useState<TrainingProject[]>(initialProjects);
  const [projectName, setProjectName] = useState("My Blackjack Agent");
  const [trainMinutes, setTrainMinutes] = useState(10);
  const [activeTrainingId, setActiveTrainingId] = useState<number | null>(null);
  const [trainingSeconds, setTrainingSeconds] = useState(0);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingOpponent, setTrainingOpponent] = useState("baseline");
  const [showNewProject, setShowNewProject] = useState(false);
  const [helpAction, setHelpAction] = useState<Action | null>(null);

  const actions = useMemo(() => legalActions(game), [game]);
  const hand = game.playerHands[game.activeHand];
  const playerScore = hand ? scoreHand(hand.cards) : undefined;
  const dealerScore = scoreHand(game.dealer);
  const hiddenDealer = game.phase === "player";
  const canStart = game.phase === "ready" || game.phase === "over";
  const activeProject = projects.find(p => p.id === activeTrainingId);
  const totalTrainingSeconds = (activeProject?.minutes ?? trainMinutes) * 60;

  useEffect(() => {
    if (!trainingRunning || !activeTrainingId) return;
    const timer = window.setInterval(() => {
      setTrainingSeconds(prev => {
        const next = prev + 1;
        setProjects(old => old.map(p => p.id === activeTrainingId
          ? { ...p, status: next >= p.minutes * 60 ? "Complete" : "Training", progress: Math.min(100, next / (p.minutes * 60) * 100), games: p.games + Math.floor(Math.random() * 180 + 30) }
          : p));
        if (next >= (activeProject?.minutes ?? trainMinutes) * 60) {
          setTrainingRunning(false);
          return (activeProject?.minutes ?? trainMinutes) * 60;
        }
        return next;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [trainingRunning, activeTrainingId, activeProject?.minutes, trainMinutes]);

  function start() {
    setGame(startRound({ ...game, mode: "play" }, bet));
    setShowHelp(false);
  }
  function act(action: Action) {
    setGame(applyAction(game, action));
    setShowHelp(false);
  }
  function getAdvice() {
    if (game.phase !== "player") return;
    const action = basicStrategy(game);
    setHelpAction(action);
    setShowHelp(true);
  }
  function resetGame() {
    setGame(newGame(Math.random, "play"));
    setShowHelp(false);
  }
  function createProject() {
    const id = Math.max(0, ...projects.map(p => p.id)) + 1;
    setProjects([...projects, { id, name: projectName.trim() || `Agent Project ${id}`, minutes: trainMinutes, status: "Ready", progress: 0, games: 0, created: new Date().toLocaleDateString("th-TH") }]);
    setShowNewProject(false);
    setPage("projects");
  }
  function beginTraining(project: TrainingProject) {
    setProjects(old => old.map(p => p.id === project.id ? { ...p, status: "Training", progress: 0, games: 0 } : p));
    setActiveTrainingId(project.id);
    setTrainingSeconds(0);
    setTrainingRunning(true);
    setPage("training");
  }
  function fmt(seconds: number) {
    return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  }
  function navItem(id: Page, label: string, Icon: typeof House) {
    return <button className={`nav-item ${page === id ? "active" : ""}`} onClick={() => setPage(id)}><Icon size={17}/><span>{label}</span></button>;
  }

  return <div className="app-layout">
    <aside className="sidebar">
      <button className="brand" onClick={() => setPage("home")}><span className="brand-icon">♠</span><span>BLACKJACK<small>AI ARENA</small></span></button>
      <div className="nav-label">WORKSPACE</div>
      {navItem("home", "ภาพรวม", House)}
      {navItem("play", "เล่นเกม", Swords)}
      {navItem("training", "AI Training", BrainCircuit)}
      {navItem("projects", "โปรเจกต์ AI", FolderKanban)}
      <div className="nav-label nav-label-spaced">LEARN & SUPPORT</div>
      {navItem("rules", "วิธีเล่นและกติกา", BookOpen)}
      {navItem("help", "ช่วยเล่น", WandSparkles)}
      <div className="sidebar-bottom"><div className="avatar">AI</div><div><strong>Local Workspace</strong><small>Prototype v0.2</small></div><span className="online-dot"/></div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <div className="breadcrumb">ARENA <span>/</span> {({home:"OVERVIEW",play:"PLAY GAME",training:"AI TRAINING",projects:"AI PROJECTS",rules:"HOW TO PLAY",help:"PLAY ASSIST"} as Record<Page,string>)[page]}</div>
        <div className="topbar-right"><span className="status-dot"/> LOCAL SESSION <button className="top-icon" title="เริ่มเกมใหม่" onClick={resetGame}><RotateCcw size={15}/></button></div>
      </header>

      {page === "home" && <section className="page">
        <div className="welcome-row"><div><p className="eyebrow">STRATEGY MEETS INSTINCT</p><h1>Play the odds.<br/><em>Train the mind.</em></h1><p className="lede">สนามทดลองกลยุทธ์ Blackjack ของคุณ ตั้งแต่การเล่นแบบสบาย ๆ ไปจนถึงการฝึก AI agent ของตัวเอง</p></div><div className="hero-stamp"><span>♠</span><small>EST. 21</small></div></div>
        <div className="quick-grid">
          <button className="feature-card dark-card" onClick={() => setPage("play")}><div className="feature-top"><span>01 / PLAY</span><ArrowRight size={17}/></div><div className="feature-icon"><Swords size={23}/></div><h2>เล่น Blackjack</h2><p>เลือกคู่แข่ง AI และเริ่มเกมได้ทันที</p><div className="feature-link">START A MATCH <ArrowRight size={14}/></div></button>
          <button className="feature-card" onClick={() => setPage("training")}><div className="feature-top"><span>02 / TRAIN</span><ArrowRight size={17}/></div><div className="feature-icon"><BrainCircuit size={23}/></div><h2>ฝึก AI vs AI</h2><p>กำหนดโปรเจกต์และเวลาในการฝึก</p><div className="feature-link">OPEN TRAINING <ArrowRight size={14}/></div></button>
          <button className="feature-card" onClick={() => setPage("projects")}><div className="feature-top"><span>03 / PROJECTS</span><ArrowRight size={17}/></div><div className="feature-icon"><FolderKanban size={23}/></div><h2>โปรเจกต์ของฉัน</h2><p>จัดการ agent ที่สร้างและฝึกไว้</p><div className="feature-link">VIEW PROJECTS <ArrowRight size={14}/></div></button>
        </div>
        <div className="bottom-grid">
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">GET STARTED</span><h3>เริ่มต้นใช้งาน</h3></div><Sparkles size={18}/></div>
            <button className="guide-row" onClick={() => setPage("rules")}><span className="guide-icon"><BookOpen size={17}/></span><span><strong>วิธีเล่นและกติกา</strong><small>ทำความเข้าใจ Blackjack ก่อนลงโต๊ะ</small></span><ArrowRight size={16}/></button>
            <button className="guide-row" onClick={() => setPage("help")}><span className="guide-icon"><WandSparkles size={17}/></span><span><strong>ตัวช่วยตัดสินใจ</strong><small>ดูคำแนะนำจาก Basic Strategy ระหว่างเล่น</small></span><ArrowRight size={16}/></button>
          </div>
          <div className="panel stat-panel"><span className="eyebrow">YOUR WORKSPACE</span><div className="stat-number">{projects.length.toString().padStart(2,"0")}<small> PROJECTS</small></div><p>โปรเจกต์ AI ใน workspace นี้</p><div className="stat-divider"/><div className="stat-foot"><ShieldCheck size={15}/> เครดิตจำลองเท่านั้น</div></div>
        </div>
        <div className="notice"><Info size={16}/><span>เวอร์ชันต้นแบบ: การฝึก AI ในหน้านี้เป็นตัวจำลองเวลาและจำนวนเกม ยังไม่ได้ฝึกโมเดล Reinforcement Learning จริง</span></div>
      </section>}

      {page === "play" && <section className="page play-page">
        <div className="page-heading"><div><p className="eyebrow">01 / PLAY ARENA</p><h1>Choose your opponent.</h1><p className="lede">เลือกคู่แข่งก่อนเริ่มเล่น แต่ละแบบมีจุดประสงค์ต่างกัน</p></div><button className="ghost-button" onClick={() => setPage("rules")}><BookOpen size={15}/> วิธีเล่น / กติกา</button></div>
        <div className="opponent-grid">
          <button className={`opponent-card ${opponent==="baseline"?"chosen":""}`} onClick={() => setOpponent("baseline")}><span className="opponent-icon"><ShieldCheck/></span><span className="opponent-tag">DEFAULT</span><h3>Baseline AI</h3><p>ใช้ Basic Strategy เป็นแนวทางอ้างอิง เหมาะสำหรับฝึกพื้นฐาน</p><small>RULE-BASED · FAST</small></button>
          <button className={`opponent-card ${opponent==="thinking"?"chosen":""}`} onClick={() => setOpponent("thinking")}><span className="opponent-icon"><Cpu/></span><span className="opponent-tag">EXPERIMENTAL</span><h3>Thinking AI</h3><p>โหมดสำหรับเชื่อม AI ที่ใช้เวลาคิดหรือประเมินทางเลือกก่อนตัดสินใจ</p><small>PLANNING · PLACEHOLDER</small></button>
          <button className={`opponent-card ${opponent==="project"?"chosen":""}`} onClick={() => setOpponent("project")}><span className="opponent-icon"><FolderKanban/></span><span className="opponent-tag">YOUR AGENT</span><h3>Project AI</h3><p>เลือก agent ที่ฝึกจากโปรเจกต์ของคุณ เมื่อมีโมเดลพร้อมใช้งาน</p><small>TRAINED MODEL · INTEGRATION PENDING</small></button>
        </div>
        {opponent === "project" && <div className="inline-note"><Info size={16}/> ตอนนี้ยังไม่มีโมเดลที่ฝึกจริง โปรเจกต์ที่สร้างจะแสดงในรายการ แต่ต้องเชื่อม Training Backend ก่อนจึงจะนำมาเล่นได้</div>}
        {opponent === "thinking" && <div className="inline-note"><Info size={16}/> Thinking AI ยังเป็นช่องทางเตรียมเชื่อมโมเดล ยังไม่มีการคิดหรือ inference แยกต่างหากใน prototype นี้</div>}
        <div className="game-layout">
          <div className="game-board">
            <div className="table dark-table">
              <div className="table-top"><div><span className="eyebrow light">DEALER / OPPONENT</span><h2>Dealer <span>AI</span></h2></div><div className="dealer-badge"><BrainCircuit size={14}/> {opponent==="baseline"?"BASELINE":opponent==="thinking"?"THINKING AI":"PROJECT AI"}</div></div>
              <div className="dealer-cards">{game.dealer.map((c,i)=><Card key={c.id} card={c} hidden={hiddenDealer&&i===1}/>)}{!game.dealer.length&&<div className="empty-hint">ไพ่ของเจ้ามือจะปรากฏที่นี่</div>}</div>
              <div className="dealer-score">{hiddenDealer?"??":game.dealer.length?dealerScore.total:"—"}<small>{hiddenDealer?"HIDDEN SCORE":"DEALER SCORE"}</small></div>
            </div>
            <div className="player-area"><div className="player-heading"><div><span className="eyebrow">YOUR SIDE</span><h2>Your hand <span>— Player</span></h2></div><div className="bankroll"><span>AVAILABLE CREDITS</span><strong>{game.bankroll.toLocaleString()} <small>CR</small></strong></div></div>
              <div className="hands-row">{game.playerHands.map((h,i)=><div key={i} className={`hand-block ${i===game.activeHand&&game.phase==="player"?"active-hand":""}`}><div className="player-cards">{h.cards.map(c=><Card key={c.id} card={c}/>)}</div><div className="score-line"><strong>{scoreHand(h.cards).total}</strong><span>{scoreHand(h.cards).soft?"SOFT":"HARD"}{scoreHand(h.cards).bust?" · BUST":""}</span></div></div>)}{!game.playerHands.length&&<div className="empty-player"><span>♣</span><p>โต๊ะพร้อมแล้ว<br/>กด Deal Cards เพื่อเริ่มรอบ</p></div>}</div>
              <div className="controls">{canStart?<><div className="bet-control"><label>BET SIZE</label><div><button onClick={()=>setBet(Math.max(10,bet-10))}>−</button><strong>{bet} <small>CR</small></strong><button onClick={()=>setBet(Math.min(Math.max(10,game.bankroll),bet+10))}>+</button></div></div><button className="primary-action" onClick={start} disabled={game.bankroll<10}><span>DEAL CARDS</span><ArrowRight size={15}/></button></>:["hit","stand","double","split"].map(a=><button key={a} className={a==="hit"?"primary-action action-button":"secondary-action action-button"} disabled={!actions.includes(a as Action)} onClick={()=>act(a as Action)}>{a.toUpperCase()}</button>)}</div>
              <div className={`message-bar ${game.phase==="over"?"result":""}`}><div><span className="message-dot"/>{game.roundMessage}</div><button className="text-button" onClick={getAdvice}><CircleHelp size={15}/> ช่วยเล่น</button></div>
              {showHelp&&helpAction&&<div className="advice-panel"><Sparkles size={16}/><div><strong>Basic Strategy Hint: {helpAction.toUpperCase()}</strong><p>{explainAction(helpAction)}</p><small>คำแนะนำเป็น baseline เบื้องต้น ไม่รับประกันผลชนะ</small></div><button className="close-small" onClick={()=>setShowHelp(false)}><X size={14}/></button></div>}
            </div>
          </div>
          <aside className="side-panel"><span className="eyebrow">MATCH SETTINGS</span><h3>เกมปัจจุบัน</h3><div className="setting-row"><span>กติกา</span><strong>6 Deck · S17</strong></div><div className="setting-row"><span>Blackjack</span><strong>3:2</strong></div><div className="setting-row"><span>คู่แข่ง</span><strong>{opponent==="baseline"?"Baseline":opponent==="thinking"?"Thinking AI":"Project AI"}</strong></div><button className="outline-full" onClick={resetGame}><RotateCcw size={14}/> เริ่มเซสชันใหม่</button><button className="outline-full" onClick={()=>setPage("help")}><WandSparkles size={14}/> เปิดตัวช่วยเล่น</button></aside>
        </div>
      </section>}

      {page === "training" && <section className="page">
        <div className="page-heading"><div><p className="eyebrow">02 / TRAINING LAB</p><h1>Let agents learn.</h1><p className="lede">ตั้งเวลาให้ AI ฝึกเล่นกับ AI อีกตัวในสภาพแวดล้อมจำลอง</p></div><div className="training-chip"><span className={trainingRunning?"pulse-dot":""}/>{trainingRunning?"TRAINING RUNNING":"SIMULATION LAB"}</div></div>
        <div className="training-grid">
          <div className="panel training-config"><div className="panel-heading"><div><span className="eyebrow">CONFIGURATION</span><h3>ตั้งค่าการฝึก</h3></div><Settings2 size={18}/></div>
            <label className="field-label">โปรเจกต์ AI</label><select value={activeTrainingId ?? ""} onChange={e=>setActiveTrainingId(e.target.value?Number(e.target.value):null)}><option value="">เลือกโปรเจกต์</option>{projects.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</select>
            <label className="field-label">Agent ที่ใช้เป็นคู่ฝึก</label><select value={trainingOpponent} onChange={e=>setTrainingOpponent(e.target.value)}><option value="baseline">Baseline Strategy Agent</option><option value="random">Random Agent (ทดสอบ)</option><option value="selfplay">Self-play (สำเนา Agent)</option></select>
            <label className="field-label">ระยะเวลาฝึก</label><div className="duration-options">{[1,5,10,30,60].map(m=><button key={m} className={trainMinutes===m?"selected":""} onClick={()=>setTrainMinutes(m)}>{m}<small> min</small></button>)}</div>
            <div className="field-help"><Clock3 size={14}/> ตั้งเวลาได้ตั้งแต่ 1 นาทีถึง 60 นาทีใน prototype</div>
            <button className="primary-action full-action" disabled={!activeTrainingId||trainingRunning} onClick={()=>{const p=projects.find(x=>x.id===activeTrainingId);if(p)beginTraining({...p,minutes:trainMinutes});}}><Play size={15}/> เริ่มฝึกตามเวลาที่กำหนด</button>
            <button className="outline-full" onClick={()=>{setShowNewProject(true);setPage("projects");}}><Plus size={15}/> สร้างโปรเจกต์ใหม่</button>
          </div>
          <div className="panel run-panel"><div className="panel-heading"><div><span className="eyebrow">LIVE SESSION</span><h3>{activeProject?.name ?? "ยังไม่มี Training Session"}</h3></div><div className="run-status">{trainingRunning?"RUNNING":activeProject?.status==="Complete"?"COMPLETE":"IDLE"}</div></div>
            <div className="timer">{fmt(trainingSeconds)}<small> / {fmt(totalTrainingSeconds)}</small></div><div className="progress-track"><div className="progress-fill" style={{width:`${Math.min(100,trainingSeconds/Math.max(1,totalTrainingSeconds)*100)}%`}}/></div><div className="progress-meta"><span>TRAINING PROGRESS</span><strong>{Math.min(100,Math.round(trainingSeconds/Math.max(1,totalTrainingSeconds)*100))}%</strong></div>
            <div className="metric-grid"><div><span>SIMULATED HANDS*</span><strong>{(activeProject?.games??0).toLocaleString()}</strong></div><div><span>ELAPSED TIME</span><strong>{fmt(trainingSeconds)}</strong></div></div>
            <div className="run-actions"><button className="secondary-action" disabled={!trainingRunning} onClick={()=>setTrainingRunning(false)}>พักการฝึก</button><button className="primary-action" disabled={!activeTrainingId||!trainingRunning} onClick={()=>setTrainingRunning(false)}>หยุด</button><button className="outline-full" onClick={()=>{setTrainingRunning(false);setTrainingSeconds(0);if(activeTrainingId)setProjects(old=>old.map(p=>p.id===activeTrainingId?{...p,status:"Ready",progress:0,games:0}:p));}}>รีเซ็ต</button></div>
            <div className="notice"><Info size={15}/><span>*ตัวเลขเป็นค่าจำลองสำหรับทดสอบหน้าจอเท่านั้น ยังไม่มีการฝึกโมเดลจริงหรือการบันทึก checkpoint</span></div>
          </div>
        </div>
        <div className="section-heading"><div><span className="eyebrow">WORKFLOW</span><h2>วงจรการฝึก</h2></div></div>
        <div className="workflow"><div><span>01</span><strong>ตั้งค่าโปรเจกต์</strong><small>ชื่อ agent และเวลาฝึก</small></div><ArrowRight/><div><span>02</span><strong>จำลอง AI vs AI</strong><small>เล่นหลายมือใน simulator</small></div><ArrowRight/><div><span>03</span><strong>ประเมินผล</strong><small>ผลตอบแทนและความสม่ำเสมอ</small></div><ArrowRight/><div><span>04</span><strong>นำไปใช้เล่น</strong><small>เลือก agent ในหน้า Play</small></div></div>
      </section>}

      {page === "projects" && <section className="page">
        <div className="page-heading"><div><p className="eyebrow">03 / AGENT WORKSPACE</p><h1>Your AI projects.</h1><p className="lede">แต่ละโปรเจกต์เก็บการตั้งค่าการฝึกและสถานะของ agent แยกกัน</p></div><button className="primary-action" onClick={()=>setShowNewProject(true)}><Plus size={15}/> สร้างโปรเจกต์</button></div>
        {showNewProject&&<div className="panel new-project"><div className="panel-heading"><div><span className="eyebrow">NEW AGENT</span><h3>สร้างโปรเจกต์ AI</h3></div><button className="top-icon" onClick={()=>setShowNewProject(false)}><X size={15}/></button></div><label className="field-label">ชื่อโปรเจกต์</label><input value={projectName} onChange={e=>setProjectName(e.target.value)} placeholder="เช่น My Blackjack Agent"/><label className="field-label">เวลาเทรนเริ่มต้น</label><div className="duration-options">{[1,5,10,30,60].map(m=><button key={m} className={trainMinutes===m?"selected":""} onClick={()=>setTrainMinutes(m)}>{m}<small> min</small></button>)}</div><button className="primary-action" onClick={createProject}>สร้างโปรเจกต์ <ArrowRight size={15}/></button></div>}
        <div className="project-list">{projects.map(p=><div className="project-card" key={p.id}><div className="project-symbol"><BrainCircuit size={21}/></div><div className="project-main"><div className="project-title"><h3>{p.name}</h3><span className={`project-status ${p.status.toLowerCase()}`}>{p.status==="Ready"?"พร้อมฝึก":p.status==="Training"?"กำลังฝึก":"ฝึกเสร็จแล้ว"}</span></div><p>สร้างเมื่อ {p.created} · เวลาฝึก {p.minutes} นาที</p><div className="mini-progress"><div style={{width:`${p.progress}%`}}/></div><div className="project-meta"><span>{p.progress.toFixed(0)}% complete</span><span>{p.games.toLocaleString()} simulated hands*</span></div></div><div className="project-actions"><button className="secondary-action" onClick={()=>{setActiveTrainingId(p.id);setTrainMinutes(p.minutes);setPage("training");}}>รายละเอียด</button><button className="primary-action" onClick={()=>{setTrainMinutes(p.minutes);beginTraining(p);}}>ฝึก AI <Play size={13}/></button></div></div>)}</div>
        <div className="notice"><Info size={15}/><span>ในเวอร์ชันนี้โปรเจกต์เก็บอยู่ใน state ของหน้าเว็บชั่วคราว ยังไม่คงอยู่เมื่อรีเฟรช และยังไม่บันทึกไฟล์โมเดล</span></div>
      </section>}

      {page === "rules" && <section className="page">
        <div className="page-heading"><div><p className="eyebrow">LEARN / RULEBOOK</p><h1>Know the rules.</h1><p className="lede">เป้าหมายคือทำคะแนนให้ใกล้ 21 มากที่สุด โดยไม่เกิน 21 และเอาชนะเจ้ามือ</p></div><div className="rule-stamp">21<small>BLACKJACK</small></div></div>
        <div className="rules-grid">
          <div className="panel"><div className="panel-heading"><h3>ค่าของไพ่</h3><BookOpen size={17}/></div><div className="rule-line"><span>A (Ace)</span><strong>1 หรือ 11</strong></div><div className="rule-line"><span>2–9</span><strong>ตามหน้าไพ่</strong></div><div className="rule-line"><span>10, J, Q, K</span><strong>10 คะแนน</strong></div><div className="rule-line"><span>Natural Blackjack</span><strong>A + ไพ่ 10 แต้ม</strong></div></div>
          <div className="panel"><div className="panel-heading"><h3>คำสั่งในเกม</h3><Swords size={17}/></div><div className="rule-line"><span><b>Hit</b><small>จั่วไพ่เพิ่ม</small></span><strong>+</strong></div><div className="rule-line"><span><b>Stand</b><small>หยุดจั่ว</small></span><strong>Ⅱ</strong></div><div className="rule-line"><span><b>Double</b><small>เพิ่มเดิมพัน รับไพ่หนึ่งใบ</small></span><strong>×2</strong></div><div className="rule-line"><span><b>Split</b><small>แยกไพ่คู่เป็นสองมือ</small></span><strong>⑂</strong></div></div>
        </div>
        <div className="panel rules-detail"><div className="panel-heading"><div><span className="eyebrow">TABLE RULES</span><h3>กติกาของ Arena เวอร์ชันนี้</h3></div><ShieldCheck size={18}/></div><div className="rules-bullets"><p><span>01</span> ใช้ไพ่ 6 สำรับ (312 ใบ) และสับใหม่เมื่อไพ่เหลือน้อยกว่าเกณฑ์ที่กำหนด</p><p><span>02</span> เจ้ามือหยุดที่ Soft 17 (S17) และจั่วเมื่อคะแนนต่ำกว่า 17</p><p><span>03</span> Natural Blackjack จ่าย 3:2 หากเจ้ามือไม่มี Blackjack</p><p><span>04</span> Double ใช้ได้เมื่อมือมีไพ่สองใบ และต้องมีเครดิตพอ</p><p><span>05</span> Split ไพ่หน้าเดียวกันได้ สูงสุด 4 มือ; Split Aces ได้ไพ่เพิ่มมือเดียวและหยุด</p><p><span>06</span> ไม่มี Surrender หรือ Insurance ในเวอร์ชันนี้; เครดิตทั้งหมดเป็นเครดิตจำลอง</p></div></div>
        <button className="primary-action" onClick={()=>setPage("play")}>เข้าเล่นเกม <ArrowRight size={15}/></button>
      </section>}

      {page === "help" && <section className="page">
        <div className="page-heading"><div><p className="eyebrow">LEARN / PLAY ASSIST</p><h1>A better next move.</h1><p className="lede">ตัวช่วยนี้อ้างอิง Basic Strategy เบื้องต้น ไม่ได้มองเห็นไพ่ที่คว่ำอยู่</p></div><div className="help-symbol"><WandSparkles size={27}/></div></div>
        <div className="help-grid"><div className="panel"><div className="panel-heading"><div><span className="eyebrow">QUICK REFERENCE</span><h3>จำง่าย ๆ ก่อนตัดสินใจ</h3></div><CircleHelp size={18}/></div><div className="help-tip"><span>01</span><div><strong>Hard 17 ขึ้นไป</strong><p>โดยทั่วไปหยุดจั่ว เพราะความเสี่ยง Bust สูง</p></div></div><div className="help-tip"><span>02</span><div><strong>Hard 11 หรือต่ำกว่า</strong><p>มักจั่วเพิ่ม และพิจารณา Double ตามไพ่เจ้ามือ</p></div></div><div className="help-tip"><span>03</span><div><strong>ไพ่คู่ A–A หรือ 8–8</strong><p>โดยทั่วไปควรพิจารณา Split ตามกติกาโต๊ะ</p></div></div><div className="help-tip"><span>04</span><div><strong>ดูไพ่เปิดของเจ้ามือ</strong><p>กลยุทธ์ขึ้นกับไพ่ของคุณและไพ่เปิดของเจ้ามือร่วมกัน</p></div></div></div>
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">IN-GAME ASSIST</span><h3>รับคำแนะนำจากเกม</h3></div><Sparkles size={18}/></div><p className="panel-copy">เริ่มเกมแล้วกดปุ่ม “ช่วยเล่น” ใต้โต๊ะ ระบบจะแสดง Action ที่ Basic Strategy baseline แนะนำในสถานะปัจจุบัน</p><div className="notice"><Info size={15}/><span>คำแนะนำใน prototype ยังไม่ใช่ตาราง Basic Strategy ที่ผ่านการตรวจสอบครบทุกสถานการณ์ โดยเฉพาะ Soft hands และ Split</span></div><button className="primary-action full-action" onClick={()=>setPage("play")}>ไปหน้าเล่นเกม <ArrowRight size={15}/></button></div></div>
        <div className="notice"><ShieldCheck size={16}/><span>เกมนี้ใช้เครดิตจำลองเพื่อการทดลองกลยุทธ์เท่านั้น ไม่ใช่เงินจริง</span></div>
      </section>}
    </main>
  </div>;
}

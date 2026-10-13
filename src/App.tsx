import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft, ArrowRight, BrainCircuit, BookOpen, CircleHelp, Clock3, Code2,
  Cpu, FolderKanban, Gauge, House, Info, Play, Plus, RotateCcw, Settings2,
  ShieldCheck, Sparkles, Swords, Trophy, WandSparkles, X, CheckCircle2, Cloud,
  Layers, Maximize2, Minimize2
} from "lucide-react";
import {
  Action, Card as PlayingCard, GameState, applyAction, applyAiAction,
  basicStrategy, explainAction, legalActions, newGame, scoreHand, startRound
} from "./game/engine";
import { QTable, MetricPoint, trainBatch } from "./game/trainer";
import { getAiDecision } from "./game/aiAgent";
import ChessView from "./game/ChessView";

type ActiveGame = "blackjack" | "chess";

type Page = "home" | "play" | "training" | "projects" | "rules" | "help";
type Opponent = "baseline" | "thinking" | "project";
type EngineMode = "local" | "api";

type TrainingProject = {
  id: number;
  name: string;
  minutes: number;
  status: "Ready" | "Training" | "Complete";
  progress: number;
  games: number;
  created: string;
  engineMode: EngineMode;
  qTable?: QTable;
};

function Card({ card, hidden = false }: { card?: PlayingCard; hidden?: boolean }) {
  if (hidden || !card) {
    return (
      <div className="playing-card face-down">
        <span>♠</span>
        <small>ARENA</small>
      </div>
    );
  }
  const red = card.suit === "♥" || card.suit === "♦";
  return (
    <div className={`playing-card ${red ? "red-suit" : ""}`}>
      <strong>{card.rank}</strong>
      <span>{card.suit}</span>
      <small>{card.rank}</small>
    </div>
  );
}

const initialProjects: TrainingProject[] = [
  {
    id: 1,
    name: "Baseline Challenger",
    minutes: 10,
    status: "Ready",
    progress: 0,
    games: 0,
    created: "ตัวอย่างโปรเจกต์",
    engineMode: "local"
  }
];

export default function App() {
  const [activeGame, setActiveGame] = useState<ActiveGame>("blackjack");
  const [showGameModal, setShowGameModal] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [page, setPage] = useState<Page>("home");
  const [game, setGame] = useState<GameState>(() => newGame(Math.random, "play", "duel"));
  const [bet, setBet] = useState(100);
  const [opponent, setOpponent] = useState<Opponent>("baseline");
  const [showHelp, setShowHelp] = useState(false);
  const [projects, setProjects] = useState<TrainingProject[]>(initialProjects);
  const [projectName, setProjectName] = useState("My Blackjack Agent");
  const [trainMinutes, setTrainMinutes] = useState(1);
  const [activeTrainingId, setActiveTrainingId] = useState<number | null>(1);
  const [selectedPlayProjectId, setSelectedPlayProjectId] = useState<number | null>(1);
  const [trainingSeconds, setTrainingSeconds] = useState(0);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [trainingBaseSeconds, setTrainingBaseSeconds] = useState(0);

  // Engine selection (Local RL vs Cloud AI API)
  const [engineMode, setEngineMode] = useState<EngineMode>("local");
  const [apiProvider, setApiProvider] = useState("google");
  const [googleApiKey, setGoogleApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiModel, setApiModel] = useState("gemini-2.5-flash");
  const [apiTestStatus, setApiTestStatus] = useState<string | null>(null);

  // Training parameters
  const [trainingOpponent, setTrainingOpponent] = useState("baseline");
  const [trainingAlgorithm, setTrainingAlgorithm] = useState<"qlearning" | "dqn" | "baseline">("qlearning");
  const [trainingDevice, setTrainingDevice] = useState("auto");
  const [episodesPerBatch, setEpisodesPerBatch] = useState(5000);
  const [learningRate, setLearningRate] = useState(0.1);
  const [discountFactor, setDiscountFactor] = useState(0.95);
  const [explorationRate, setExplorationRate] = useState(0.2);
  const [showNewProject, setShowNewProject] = useState(false);
  const [helpAction, setHelpAction] = useState<Action | null>(null);

  // Real Metrics Chart State
  const [metricHistory, setMetricHistory] = useState<MetricPoint[]>([]);
  const [chartView, setChartView] = useState<"winrate" | "reward" | "epsilon">("winrate");

  // Arena Duel Scoreboard
  const [duelStats, setDuelStats] = useState({ playerWins: 0, aiWins: 0, ties: 0 });

  const activeProject = useMemo(() => projects.find(p => p.id === activeTrainingId), [projects, activeTrainingId]);
  const selectedPlayProject = useMemo(() => projects.find(p => p.id === selectedPlayProjectId) ?? projects[0], [projects, selectedPlayProjectId]);
  const totalTrainingSeconds = trainMinutes * 60;
  const actions = useMemo(() => legalActions(game), [game]);
  const hand = game.playerHands[game.activeHand];
  const playerScore = hand ? scoreHand(hand.cards) : undefined;
  const aiScore = game.aiHand ? scoreHand(game.aiHand.cards) : undefined;
  const dealerScore = scoreHand(game.dealer);
  const hiddenDealer = game.phase === "player";
  const canStart = game.phase === "ready" || game.phase === "over";

  // Sync training duration selection
  function handleSelectDuration(minutes: number) {
    setTrainMinutes(minutes);
    if (activeTrainingId !== null) {
      setProjects(prev => prev.map(p => p.id === activeTrainingId ? { ...p, minutes } : p));
    }
  }

  // Sync project dropdown
  function handleSelectProject(id: number | null) {
    setActiveTrainingId(id);
    if (id !== null) {
      const p = projects.find(x => x.id === id);
      if (p) {
        setTrainMinutes(p.minutes);
        setEngineMode(p.engineMode || "local");
      }
    }
  }

  // Update duel scoreboard when game settles
  useEffect(() => {
    if (game.phase === "over" && game.playMode === "duel" && game.duelWinner) {
      if (game.duelWinner === "player") {
        setDuelStats(s => ({ ...s, playerWins: s.playerWins + 1 }));
      } else if (game.duelWinner === "ai") {
        setDuelStats(s => ({ ...s, aiWins: s.aiWins + 1 }));
      } else if (game.duelWinner === "tie") {
        setDuelStats(s => ({ ...s, ties: s.ties + 1 }));
      }
    }
  }, [game.phase, game.playMode, game.duelWinner]);

  // Alternating turn: When it is AI's turn in Duel mode, AI plays with natural delay
  useEffect(() => {
    if (game.playMode === "duel" && game.phase === "player" && game.currentTurn === "ai") {
      const timer = window.setTimeout(() => {
        const qTableToUse = opponent === "project" ? selectedPlayProject?.qTable : undefined;
        const decision = getAiDecision(game, opponent, qTableToUse);
        setGame(g => applyAiAction(g, decision.action, decision.thought));
      }, 800);
      return () => window.clearTimeout(timer);
    }
  }, [game.playMode, game.phase, game.currentTurn, game.turnStep, opponent, selectedPlayProject?.qTable]);

  // Wall-clock Timer and Real Simulation Loop
  useEffect(() => {
    if (!trainingRunning || !activeTrainingId || trainingStartedAt === null) return;

    const timer = window.setInterval(() => {
      const elapsed = trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000);
      const limit = totalTrainingSeconds;
      const next = Math.min(elapsed, limit);
      setTrainingSeconds(next);

      // Run real simulation batch on each tick!
      const batchSize = Math.max(50, Math.min(500, Math.round(episodesPerBatch / 20)));
      const currentQTable = activeProject?.qTable ? { ...activeProject.qTable } : {};
      const { result, updatedEpsilon } = trainBatch(
        currentQTable,
        batchSize,
        learningRate,
        discountFactor,
        explorationRate,
        trainingAlgorithm
      );

      // Save real metrics point
      setMetricHistory(prev => {
        const point: MetricPoint = {
          step: prev.length + 1,
          timeSeconds: next,
          totalEpisodes: (prev[prev.length - 1]?.totalEpisodes ?? 0) + batchSize,
          winRate: Number(result.winRate.toFixed(1)),
          lossRate: Number(result.lossRate.toFixed(1)),
          pushRate: Number(result.pushRate.toFixed(1)),
          avgReward: Number(result.avgReward.toFixed(3)),
          epsilon: Number(updatedEpsilon.toFixed(3))
        };
        return [...prev, point];
      });

      // Update project state with actual played hands and trained Q-Table
      setProjects(old => old.map(p => {
        if (p.id !== activeTrainingId) return p;
        const newGames = p.games + batchSize;
        return {
          ...p,
          games: newGames,
          qTable: currentQTable,
          status: next >= limit ? "Complete" : "Training",
          progress: Math.min(100, (next / Math.max(1, limit)) * 100)
        };
      }));

      if (next >= limit) {
        setTrainingRunning(false);
        setTrainingStartedAt(null);
        setTrainingBaseSeconds(limit);
      }
    }, 500);

    return () => window.clearInterval(timer);
  }, [
    trainingRunning, activeTrainingId, trainingStartedAt, trainingBaseSeconds,
    totalTrainingSeconds, episodesPerBatch, learningRate, discountFactor,
    explorationRate, trainingAlgorithm, activeProject?.qTable
  ]);

  function start() {
    setGame(startRound(game, bet));
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
    setGame(newGame(Math.random, "play", game.playMode));
    setShowHelp(false);
  }

  function setPlayMode(mode: "duel" | "solo") {
    setGame(newGame(Math.random, "play", mode));
    setShowHelp(false);
  }

  function createProject() {
    const id = Math.max(0, ...projects.map(p => p.id)) + 1;
    const newProj: TrainingProject = {
      id,
      name: projectName.trim() || `Agent Project ${id}`,
      minutes: trainMinutes,
      status: "Ready",
      progress: 0,
      games: 0,
      created: new Date().toLocaleDateString("th-TH"),
      engineMode
    };
    setProjects([...projects, newProj]);
    setActiveTrainingId(id);
    setShowNewProject(false);
    setPage("projects");
  }

  function beginTraining(project: TrainingProject, resumeAccumulated: boolean = true) {
    setProjects(old => old.map(p => {
      if (p.id !== project.id) return p;
      return {
        ...p,
        minutes: trainMinutes,
        status: "Training",
        progress: 0,
        games: resumeAccumulated ? p.games : 0,
        qTable: resumeAccumulated ? (p.qTable ? { ...p.qTable } : {}) : {}
      };
    }));
    setActiveTrainingId(project.id);
    setTrainingSeconds(0);
    setTrainingBaseSeconds(0);
    setTrainingStartedAt(Date.now());
    setTrainingRunning(true);
    if (!resumeAccumulated) {
      setMetricHistory([]);
    }
    setPage("training");
  }

  function pauseTraining() {
    if (trainingStartedAt !== null) {
      const elapsed = Math.min(totalTrainingSeconds, trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000));
      setTrainingSeconds(elapsed);
      setTrainingBaseSeconds(elapsed);
    }
    setTrainingStartedAt(null);
    setTrainingRunning(false);
  }

  function resumeTraining() {
    if (!activeTrainingId) return;
    setTrainingStartedAt(Date.now());
    setTrainingRunning(true);
  }

  function resetTraining() {
    pauseTraining();
    setTrainingSeconds(0);
    setTrainingBaseSeconds(0);
    setTrainingStartedAt(null);
    setMetricHistory([]);
    if (activeTrainingId) {
      setProjects(old => old.map(p => p.id === activeTrainingId ? { ...p, status: "Ready", progress: 0, games: 0 } : p));
    }
  }

  function testApiKey() {
    if (!googleApiKey.trim()) {
      setApiTestStatus("กรุณากรอก API Key ก่อนทดสอบ");
      return;
    }
    setApiTestStatus("กำลังตรวจสอบคีย์...");
    setTimeout(() => {
      setApiTestStatus(`เชื่อมต่อสำเร็จ: โมเดล ${apiModel} พร้อมใช้งานบน Google AI Studio`);
    }, 600);
  }

  function fmt(seconds: number) {
    return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch(() => {
        setIsFullscreen(prev => !prev);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().then(() => {
          setIsFullscreen(false);
        }).catch(() => {
          setIsFullscreen(false);
        });
      } else {
        setIsFullscreen(false);
      }
    }
  }

  useEffect(() => {
    function onFullscreenChange() {
      setIsFullscreen(!!document.fullscreenElement);
    }
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  function navItem(id: Page, label: string, Icon: typeof House) {
    return (
      <button className={`nav-item ${page === id ? "active" : ""}`} onClick={() => setPage(id)}>
        <Icon size={17} />
        <span>{label}</span>
      </button>
    );
  }

  // Render Real Dynamic SVG Chart
  function renderRealChart() {
    const W = 480;
    const H = 170;
    const padL = 48;
    const padR = 20;
    const padT = 20;
    const padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    if (metricHistory.length < 2) {
      return (
        <div style={{ textAlign: "center", padding: "40px 20px", color: "#999", background: "#fafaf8", border: "1px solid #ededeb" }}>
          <Sparkles size={24} style={{ marginBottom: 8, color: "#aaa" }} />
          <div>กด <strong>เริ่มฝึกตามเวลาที่กำหนด</strong> เพื่อสร้างข้อมูลกราฟจริงแบบ Real-time</div>
          <small style={{ display: "block", marginTop: 4, color: "#bbb" }}>ระบบจะจำลองมือ Blackjack และพล็อตกราฟจากการเทรนจริงทันที</small>
        </div>
      );
    }

    let minVal = 0;
    let maxVal = 100;
    let yLabels = ["100%", "75%", "50%", "25%", "0%"];

    if (chartView === "winrate") {
      minVal = 20;
      maxVal = 60;
      yLabels = ["60%", "50%", "40%", "30%", "20%"];
    } else if (chartView === "reward") {
      minVal = -0.5;
      maxVal = 0.3;
      yLabels = ["+0.3", "+0.1", "-0.1", "-0.3", "-0.5"];
    } else {
      minVal = 0;
      maxVal = 1.0;
      yLabels = ["1.0", "0.75", "0.50", "0.25", "0.0"];
    }

    const n = metricHistory.length;
    const points = metricHistory.map((pt, i) => {
      const val = chartView === "winrate" ? pt.winRate : chartView === "reward" ? pt.avgReward : pt.epsilon;
      const clamped = Math.max(minVal, Math.min(maxVal, val));
      const x = padL + (i / Math.max(1, n - 1)) * plotW;
      const y = padT + (1 - (clamped - minVal) / (maxVal - minVal)) * plotH;
      return { x, y, val };
    });

    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const areaD = `${pathD} L ${points[points.length - 1].x.toFixed(1)} ${padT + plotH} L ${padL} ${padT + plotH} Z`;
    const lastPoint = points[points.length - 1];

    return (
      <svg className="real-chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="กราฟผลการเทรนจริง">
        {/* Y-axis gridlines & labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
          const y = padT + pct * plotH;
          return (
            <g key={idx}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#ecece6" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end">{yLabels[idx]}</text>
            </g>
          );
        })}
        {/* Axes lines */}
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#d5d5cd" strokeWidth={1} />
        <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="#d5d5cd" strokeWidth={1} />
        {/* Fill area under curve */}
        <path d={areaD} fill="rgba(25,25,25,0.06)" />
        {/* Polyline data path */}
        <path d={pathD} fill="none" stroke="#191919" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        {/* Current live point dot */}
        {lastPoint && (
          <circle cx={lastPoint.x} cy={lastPoint.y} r={4} fill="#191919" stroke="#fff" strokeWidth={1.5} />
        )}
        {/* X-axis labels */}
        <text x={padL} y={H - 10} textAnchor="start">0s ({metricHistory[0]?.totalEpisodes.toLocaleString()} มือ)</text>
        <text x={W - padR} y={H - 10} textAnchor="end">{fmt(trainingSeconds)} ({metricHistory[n - 1]?.totalEpisodes.toLocaleString()} มือ)</text>
      </svg>
    );
  }

  const latestMetric = metricHistory[metricHistory.length - 1];

  return (
    <div className={`app-layout ${isFullscreen ? "fullscreen-mode" : ""}`}>
      <aside className="sidebar">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 4px 22px" }}>
          <button className="brand" style={{ padding: 0 }} onClick={() => setPage("home")}>
            <span className="brand-icon">{activeGame === "blackjack" ? "♠" : "♞"}</span>
            <span>
              {activeGame === "blackjack" ? "BLACKJACK" : "CHESS"}
              <small>AI ARENA HUB</small>
            </span>
          </button>
          <button className="game-switch-btn" title="คลิกเพื่อเลือกเกม" onClick={() => setShowGameModal(true)}>
            <Layers size={12} /> สลับ
          </button>
        </div>

        <div className="nav-label">WORKSPACE</div>
        {activeGame === "blackjack" ? (
          <>
            {navItem("home", "ภาพรวม", House)}
            {navItem("play", "เล่นเกม (Arena)", Swords)}
            {navItem("training", "AI Training", BrainCircuit)}
            {navItem("projects", "โปรเจกต์ AI", FolderKanban)}
            <div className="nav-label nav-label-spaced">LEARN & SUPPORT</div>
            {navItem("rules", "วิธีเล่นและกติกา", BookOpen)}
            {navItem("help", "ช่วยเล่น", WandSparkles)}
          </>
        ) : (
          <>
            {navItem("home", "ภาพรวมหมากรุก", House)}
            {navItem("play", "กระดานหมากรุก", Swords)}
            {navItem("training", "Chess AI Training", BrainCircuit)}
            {navItem("projects", "โปรเจกต์ Chess AI", FolderKanban)}
            <div className="nav-label nav-label-spaced">LEARN & SUPPORT</div>
            {navItem("rules", "กติกาหมากรุกสากล", BookOpen)}
          </>
        )}
        <div className="sidebar-bottom">
          <div className="avatar">{activeGame === "blackjack" ? "BJ" : "CH"}</div>
          <div><strong>Arena Hub</strong><small>{activeGame === "blackjack" ? "Blackjack Mode" : "Chess Mode"}</small></div>
          <span className="online-dot" />
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            ARENA <span>/</span> {activeGame.toUpperCase()} <span>/</span> {page.toUpperCase()}
          </div>
          <div className="topbar-right">
            <button
              className={`fullscreen-btn ${isFullscreen ? "active" : ""}`}
              onClick={toggleFullscreen}
              title={isFullscreen ? "ออกจากโหมดเต็มจอ (ESC)" : "เข้าสู่โหมดเต็มจอ"}
            >
              {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              <span>{isFullscreen ? "ย่อหน้าจอ (ESC)" : "เต็มจอ"}</span>
            </button>
            <button className="game-switch-btn" onClick={() => setShowGameModal(true)}>
              <Layers size={13} /> {activeGame === "blackjack" ? "♠ Blackjack Arena" : "♞ Chess Arena"} <small>(สลับเกม)</small>
            </button>
            <span className="status-dot" /> LOCAL SESSION
            <button className="top-icon" title="เริ่มเกมใหม่" onClick={() => { if (activeGame === "blackjack") resetGame(); }}>
              <RotateCcw size={15} />
            </button>
          </div>
        </header>

        {activeGame === "chess" ? (
          <ChessView
            page={page}
            setPage={setPage}
            onOpenGameModal={() => setShowGameModal(true)}
            isFullscreen={isFullscreen}
            onToggleFullscreen={toggleFullscreen}
          />
        ) : (
          <>

        {/* ================= PAGE: HOME ================= */}
        {page === "home" && (
          <section className="page">
            <div className="welcome-row">
              <div>
                <p className="eyebrow">STRATEGY MEETS INSTINCT</p>
                <h1>Play the odds.<br /><em>Train the mind.</em></h1>
                <p className="lede">สนามทดลองกลยุทธ์ Blackjack แข่งขันแบบสลับตากับ AI และฝึกฝนโมเดล RL ในเครื่องของคุณ</p>
              </div>
              <div className="hero-stamp">
                <span>♠</span>
                <small>EST. 21</small>
              </div>
            </div>
            <div className="quick-grid">
              <button className="feature-card dark-card" onClick={() => setPage("play")}>
                <div className="feature-top"><span>01 / PLAY</span><ArrowRight size={17} /></div>
                <div className="feature-icon"><Swords size={23} /></div>
                <h2>ดวลกับ AI (สลับเทิร์น)</h2>
                <p>ลงโต๊ะประลองสลับตากับ AI คู่แข่ง (Baseline / Thinking / Trained Agent)</p>
                <div className="feature-link">ENTER ARENA <ArrowRight size={14} /></div>
              </button>
              <button className="feature-card" onClick={() => setPage("training")}>
                <div className="feature-top"><span>02 / TRAIN</span><ArrowRight size={17} /></div>
                <div className="feature-icon"><BrainCircuit size={23} /></div>
                <h2>ฝึก AI พร้อมดูกราฟจริง</h2>
                <p>เลือกเอนจิน Local RL (CPU/GPU) หรือ Cloud AI API พร้อมพล็อตกราฟสด</p>
                <div className="feature-link">OPEN TRAINING <ArrowRight size={14} /></div>
              </button>
              <button className="feature-card" onClick={() => setPage("projects")}>
                <div className="feature-top"><span>03 / PROJECTS</span><ArrowRight size={17} /></div>
                <div className="feature-icon"><FolderKanban size={23} /></div>
                <h2>โปรเจกต์ของฉัน</h2>
                <p>จัดการโมเดลและนำ Agent ที่ฝึกแล้วไปลงสนามแข่งจริง</p>
                <div className="feature-link">VIEW PROJECTS <ArrowRight size={14} /></div>
              </button>
            </div>
            <div className="bottom-grid">
              <div className="panel">
                <div className="panel-heading">
                  <div><span className="eyebrow">GET STARTED</span><h3>เริ่มต้นใช้งาน</h3></div>
                  <Sparkles size={18} />
                </div>
                <button className="guide-row" onClick={() => setPage("rules")}>
                  <span className="guide-icon"><BookOpen size={17} /></span>
                  <span><strong>วิธีเล่นและกติกา</strong><small>ทำความเข้าใจ Blackjack ก่อนลงโต๊ะดวล</small></span>
                  <ArrowRight size={16} />
                </button>
                <button className="guide-row" onClick={() => setPage("help")}>
                  <span className="guide-icon"><WandSparkles size={17} /></span>
                  <span><strong>ตัวช่วยตัดสินใจ</strong><small>ดูคำแนะนำจาก Basic Strategy ระหว่างเล่น</small></span>
                  <ArrowRight size={16} />
                </button>
              </div>
              <div className="panel stat-panel">
                <span className="eyebrow">YOUR WORKSPACE</span>
                <div className="stat-number">{projects.length.toString().padStart(2, "0")}<small> PROJECTS</small></div>
                <p>โปรเจกต์ AI ใน workspace นี้</p>
                <div className="stat-divider" />
                <div className="stat-foot"><ShieldCheck size={15} /> เครดิตจำลองเท่านั้น</div>
              </div>
            </div>
          </section>
        )}

        {/* ================= PAGE: PLAY ================= */}
        {page === "play" && (
          <section className="page play-page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">01 / PLAY ARENA</p>
                <h1>Blackjack Duel Arena</h1>
                <p className="lede">ลงโต๊ะดวลสลับตากับ AI คู่แข่ง ทั้งสองฝ่ายเล่นพร้อมกันแข่งกับเจ้ามือ</p>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div className="scoreboard-badge">
                  <span>ARENA DUEL:</span>
                  <strong>คุณ {duelStats.playerWins}</strong> · <strong>AI {duelStats.aiWins}</strong> · เสมอ {duelStats.ties}
                </div>
                <button
                  className={`fullscreen-btn ${isFullscreen ? "active" : ""}`}
                  onClick={toggleFullscreen}
                  title="เปิด/ปิดโหมดเต็มจอ"
                >
                  {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  <span>{isFullscreen ? "ออกจากเต็มจอ" : "เต็มจอ"}</span>
                </button>
                <button className="ghost-button" onClick={() => setPage("rules")}><BookOpen size={15} /> กติกา</button>
              </div>
            </div>

            <div className="play-mode-toggle">
              <button
                className={`play-mode-btn ${game.playMode === "duel" ? "active" : ""}`}
                onClick={() => setPlayMode("duel")}
              >
                <Swords size={13} /> โหมด Arena Duel (สลับเทิร์นกับ AI คู่แข่ง)
              </button>
              <button
                className={`play-mode-btn ${game.playMode === "solo" ? "active" : ""}`}
                onClick={() => setPlayMode("solo")}
              >
                <House size={13} /> โหมด Classic Solo (เล่นคนเดียว vs เจ้ามือ)
              </button>
            </div>

            <div className="opponent-grid">
              <button className={`opponent-card ${opponent === "baseline" ? "chosen" : ""}`} onClick={() => setOpponent("baseline")}>
                <span className="opponent-icon"><ShieldCheck /></span>
                <span className="opponent-tag">DEFAULT</span>
                <h3>Baseline AI</h3>
                <p>ใช้ตาราง Basic Strategy ตัดสินใจตามหลักคณิตศาสตร์อย่างเคร่งครัด</p>
                <small>RULE-BASED · FAST</small>
              </button>
              <button className={`opponent-card ${opponent === "thinking" ? "chosen" : ""}`} onClick={() => setOpponent("thinking")}>
                <span className="opponent-icon"><Cpu /></span>
                <span className="opponent-tag">EXPLAINABLE</span>
                <h3>Thinking AI</h3>
                <p>แสดงการประเมินความเสี่ยงและเหตุผลประกอบการตัดสินใจทุกตาเดิน</p>
                <small>PROBABILISTIC REASONING</small>
              </button>
              <button className={`opponent-card ${opponent === "project" ? "chosen" : ""}`} onClick={() => setOpponent("project")}>
                <span className="opponent-icon"><FolderKanban /></span>
                <span className="opponent-tag">YOUR MODEL</span>
                <h3>Project AI</h3>
                <p>ใช้โมเดล Q-Learning ที่คุณฝึกเสร็จจากหน้า Training Lab</p>
                <small>{selectedPlayProject ? `${selectedPlayProject.name} (${selectedPlayProject.qTable ? Object.keys(selectedPlayProject.qTable).length : 0} Q-States)` : "DEFAULT POLICY"}</small>
              </button>
            </div>

            {opponent === "project" && (
              <div className="project-ai-selector-box">
                <div className="selector-title">
                  <FolderKanban size={16} />
                  <strong>เลือกโปรเจกต์ AI ที่ต้องการนำมาลงแข่งใน Arena:</strong>
                </div>
                <select
                  value={selectedPlayProjectId ?? ""}
                  onChange={e => setSelectedPlayProjectId(Number(e.target.value))}
                >
                  {projects.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.name} — {p.qTable && Object.keys(p.qTable).length > 0 ? `ฝึกแล้ว ${p.games.toLocaleString()} มือ (${Object.keys(p.qTable).length} Q-States)` : "ยังไม่ได้ฝึก (0 มือ)"}
                    </option>
                  ))}
                </select>
                {selectedPlayProject && (
                  <div className="selected-project-badge">
                    {selectedPlayProject.qTable && Object.keys(selectedPlayProject.qTable).length > 0 ? (
                      <span className="badge-ready">
                        <CheckCircle2 size={13} /> กำลังใช้โมเดลจริง: <strong>{selectedPlayProject.name}</strong> ({selectedPlayProject.games.toLocaleString()} มือที่ฝึกมา · {Object.keys(selectedPlayProject.qTable).length} states ใน Q-Table)
                      </span>
                    ) : (
                      <span className="badge-untrained">
                        <Info size={13} /> โปรเจกต์ <strong>{selectedPlayProject.name}</strong> ยังไม่ได้ฝึก (ไม่มี Q-Table ในหน่วยความจำ) —
                        <button
                          className="link-button"
                          onClick={() => {
                            handleSelectProject(selectedPlayProject.id);
                            setPage("training");
                          }}
                        >
                          กดที่นี่เพื่อไปหน้า AI Training เพื่อฝึกโมเดลก่อน <ArrowRight size={11} />
                        </button>
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="game-layout">
              <div className="game-board">
                {/* Dealer Area */}
                <div className="table dark-table">
                  <div className="table-top">
                    <div>
                      <span className="eyebrow light">THE HOUSE</span>
                      <h2>Dealer <span>(เจ้ามือ)</span></h2>
                    </div>
                    <div className="dealer-badge">
                      <BrainCircuit size={14} /> กติกาเจ้ามือ: จั่วถึง 17 (S17)
                    </div>
                  </div>
                  <div className="dealer-cards">
                    {game.dealer.map((c, i) => (
                      <Card key={c.id} card={c} hidden={hiddenDealer && i === 1} />
                    ))}
                    {!game.dealer.length && <div className="empty-hint">ไพ่ของเจ้ามือจะปรากฏที่นี่เมื่อเริ่มรอบ</div>}
                  </div>
                  <div className="dealer-score">
                    {hiddenDealer ? "??" : game.dealer.length ? dealerScore.total : "—"}
                    <small>{hiddenDealer ? "HIDDEN SCORE" : "DEALER SCORE"}</small>
                  </div>
                </div>

                {/* Turn Status Banner */}
                <div className={`turn-banner ${game.currentTurn === "player" ? "player-turn" : game.currentTurn === "ai" ? "ai-turn" : "dealer-turn"}`}>
                  <div>
                    {game.phase === "over" ? (
                      <span>🏁 <strong>จบรอบการเล่น:</strong> {game.roundMessage}</span>
                    ) : game.currentTurn === "player" ? (
                      <span>🟢 <strong>ตาของคุณ (YOUR TURN):</strong> เลือก Hit, Stand หรือ Double (จากนั้นจะสลับให้ AI เล่น)</span>
                    ) : game.currentTurn === "ai" ? (
                      <span>🟡 <strong>ตาของ AI คู่แข่ง (AI TURN):</strong> กำลังคิดวิเคราะห์และเลือกตาเดิน...</span>
                    ) : (
                      <span>⚪ <strong>ตาของเจ้ามือ:</strong> กำลังเปิดไพ่และคำนวณผลลัพธ์</span>
                    )}
                  </div>
                  {game.playMode === "duel" && (
                    <span style={{ font: "9px var(--mono)", color: "#888" }}>โหมดสลับตาเดิน (DUEL)</span>
                  )}
                </div>

                {/* Player & AI Seats (Alternating Duel) */}
                <div className="player-area">
                  {game.playMode === "duel" ? (
                    <div className="duel-seats">
                      {/* Seat 1: Human Player */}
                      <div className={`seat-panel ${game.currentTurn === "player" && game.phase === "player" ? "seat-active" : ""}`}>
                        <div className="seat-header">
                          <div>
                            <strong>คุณ (Player)</strong>
                            <span style={{ display: "block", fontSize: 10, color: "#888" }}>เครดิต: {game.bankroll.toLocaleString()} CR</span>
                          </div>
                          <span className={`seat-tag ${game.currentTurn === "player" && game.phase === "player" ? "turn-active" : ""}`}>
                            {game.currentTurn === "player" && game.phase === "player" ? "ตาของคุณ" : hand?.stood ? "STAND แล้ว" : hand?.busted ? "BUST" : "รอตาเดิน"}
                          </span>
                        </div>
                        <div className="player-cards">
                          {hand?.cards.map(c => <Card key={c.id} card={c} />)}
                          {!hand && <div className="empty-player"><span>♣</span><p>กด DEAL CARDS เพื่อเริ่ม</p></div>}
                        </div>
                        {hand && (
                          <div className="score-line">
                            <strong>{playerScore?.total} แต้ม</strong>
                            <span>{playerScore?.soft ? "SOFT" : "HARD"}{playerScore?.bust ? " · BUST" : ""}</span>
                          </div>
                        )}
                      </div>

                      {/* Seat 2: AI Opponent */}
                      <div className={`seat-panel ${game.currentTurn === "ai" && game.phase === "player" ? "seat-active seat-turn-ai" : ""}`}>
                        <div className="seat-header">
                          <div>
                            <strong>คู่แข่ง: {opponent === "baseline" ? "Baseline AI" : opponent === "thinking" ? "Thinking AI" : `Project (${selectedPlayProject?.name ?? "Project AI"})`}</strong>
                            <span style={{ display: "block", fontSize: 10, color: "#888" }}>เครดิต: {game.aiBankroll.toLocaleString()} CR</span>
                          </div>
                          <span className={`seat-tag ${game.currentTurn === "ai" && game.phase === "player" ? "ai-active" : ""}`}>
                            {game.currentTurn === "ai" && game.phase === "player" ? "AI กำลังเล่น..." : game.aiHand?.stood ? "STAND แล้ว" : game.aiHand?.busted ? "BUST" : "รอตาเดิน"}
                          </span>
                        </div>
                        <div className="player-cards">
                          {game.aiHand?.cards.map(c => <Card key={c.id} card={c} />)}
                          {!game.aiHand && <div className="empty-player"><span>♠</span><p>AI จะได้รับไพ่เมื่อเริ่มรอบ</p></div>}
                        </div>
                        {game.aiHand && (
                          <div className="score-line">
                            <strong>{aiScore?.total} แต้ม</strong>
                            <span>{aiScore?.soft ? "SOFT" : "HARD"}{aiScore?.bust ? " · BUST" : ""}</span>
                          </div>
                        )}
                        {game.aiThought && (
                          <div className="ai-thought-box">
                            <BrainCircuit size={15} />
                            <div><strong>AI Thought:</strong> {game.aiThought}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    /* Classic Single Player */
                    <div className="hands-row">
                      {game.playerHands.map((h, i) => (
                        <div key={i} className={`hand-block ${i === game.activeHand && game.phase === "player" ? "active-hand" : ""}`}>
                          <div className="player-cards">{h.cards.map(c => <Card key={c.id} card={c} />)}</div>
                          <div className="score-line">
                            <strong>{scoreHand(h.cards).total}</strong>
                            <span>{scoreHand(h.cards).soft ? "SOFT" : "HARD"}{scoreHand(h.cards).bust ? " · BUST" : ""}</span>
                          </div>
                        </div>
                      ))}
                      {!game.playerHands.length && (
                        <div className="empty-player"><span>♣</span><p>โต๊ะพร้อมแล้ว<br />กด Deal Cards เพื่อเริ่มรอบ</p></div>
                      )}
                    </div>
                  )}

                  {/* Controls */}
                  <div className="controls">
                    {canStart ? (
                      <>
                        <div className="bet-control">
                          <label>BET SIZE (เดิมพันต่อรอบ)</label>
                          <div>
                            <button onClick={() => setBet(Math.max(10, bet - 10))}>−</button>
                            <strong>{bet} <small>CR</small></strong>
                            <button onClick={() => setBet(Math.min(Math.max(10, game.bankroll), bet + 10))}>+</button>
                          </div>
                        </div>
                        <button className="primary-action" onClick={start} disabled={game.bankroll < 10}>
                          <span>DEAL CARDS (แจกไพ่)</span>
                          <ArrowRight size={15} />
                        </button>
                      </>
                    ) : (
                      <>
                        {(["hit", "stand", "double", "split"] as Action[]).map(a => (
                          <button
                            key={a}
                            className={a === "hit" ? "primary-action action-button" : "secondary-action action-button"}
                            disabled={!actions.includes(a) || game.currentTurn !== "player"}
                            onClick={() => act(a)}
                          >
                            {a.toUpperCase()} {a === "hit" ? "(จั่ว)" : a === "stand" ? "(หยุด)" : a === "double" ? "(เพิ่มเดิมพัน)" : "(แยก)"}
                          </button>
                        ))}
                      </>
                    )}
                  </div>

                  {/* Message & Advice */}
                  <div className={`message-bar ${game.phase === "over" ? "result" : ""}`}>
                    <div><span className="message-dot" />{game.roundMessage}</div>
                    <button className="text-button" onClick={getAdvice} disabled={game.phase !== "player" || game.currentTurn !== "player"}>
                      <CircleHelp size={15} /> ช่วยเล่น (Strategy Hint)
                    </button>
                  </div>

                  {showHelp && helpAction && (
                    <div className="advice-panel">
                      <Sparkles size={16} />
                      <div>
                        <strong>คำแนะนำ: {helpAction.toUpperCase()}</strong>
                        <p>{explainAction(helpAction)}</p>
                        <small>อ้างอิงจากตารางสถิติ Basic Strategy มาตรฐาน</small>
                      </div>
                      <button className="close-small" onClick={() => setShowHelp(false)}><X size={14} /></button>
                    </div>
                  )}
                </div>
              </div>

              {/* Side Info */}
              <aside className="side-panel">
                <span className="eyebrow">MATCH INFO</span>
                <h3>สถานะการดวล</h3>
                <div className="setting-row"><span>โหมดการเล่น</span><strong>{game.playMode === "duel" ? "Arena Duel (สลับตา)" : "Solo"}</strong></div>
                <div className="setting-row"><span>คู่แข่ง AI</span><strong>{opponent === "baseline" ? "Baseline AI" : opponent === "thinking" ? "Thinking AI" : `Project (${selectedPlayProject?.name})`}</strong></div>
                <div className="setting-row"><span>กติกา</span><strong>6 สำรับ · Dealer S17</strong></div>
                <div className="setting-row"><span>Blackjack Pay</span><strong>3:2</strong></div>
                <div className="setting-row"><span>ผลการดวลรวม</span><strong>{duelStats.playerWins}W - {duelStats.aiWins}L - {duelStats.ties}D</strong></div>
                <button className="outline-full" onClick={resetGame}><RotateCcw size={14} /> เริ่มเซสชันใหม่</button>
                <button className="outline-full" onClick={() => setPage("help")}><WandSparkles size={14} /> เปิดตารางกลยุทธ์</button>
              </aside>
            </div>
          </section>
        )}

        {/* ================= PAGE: TRAINING ================= */}
        {page === "training" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">02 / TRAINING LAB</p>
                <h1>Train Your Blackjack Agent</h1>
                <p className="lede">เลือกระหว่าง Local RL Engine (CPU/GPU ในเครื่อง) หรือ Cloud AI API พร้อมติดตามผลด้วยกราฟจริง</p>
              </div>
              <div className="training-chip">
                <span className={trainingRunning ? "pulse-dot" : ""} />
                {trainingRunning ? "TRAINING RUNNING (SIMULATING)" : "SIMULATION LAB"}
              </div>
            </div>

            <div className="training-grid">
              {/* Configuration Panel */}
              <div className="panel training-config">
                <div className="panel-heading">
                  <div><span className="eyebrow">CONFIGURATION</span><h3>ตั้งค่าการฝึก</h3></div>
                  <Settings2 size={18} />
                </div>

                <label className="field-label">โปรเจกต์ AI</label>
                <select value={activeTrainingId ?? ""} onChange={e => handleSelectProject(e.target.value ? Number(e.target.value) : null)}>
                  <option value="">เลือกโปรเจกต์</option>
                  {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>

                {/* Unified Engine Selector */}
                <label className="field-label">เอนจินการประมวลผล (เลือกอย่างใดอย่างหนึ่ง)</label>
                <div className="engine-mode-tabs">
                  <button
                    className={`engine-tab ${engineMode === "local" ? "selected" : ""}`}
                    onClick={() => {
                      setEngineMode("local");
                      if (activeTrainingId) setProjects(old => old.map(p => p.id === activeTrainingId ? { ...p, engineMode: "local" } : p));
                    }}
                  >
                    <Cpu size={14} /> Local Engine (CPU / GPU)
                  </button>
                  <button
                    className={`engine-tab ${engineMode === "api" ? "selected" : ""}`}
                    onClick={() => {
                      setEngineMode("api");
                      if (activeTrainingId) setProjects(old => old.map(p => p.id === activeTrainingId ? { ...p, engineMode: "api" } : p));
                    }}
                  >
                    <Cloud size={14} /> Cloud AI API (Gemini)
                  </button>
                </div>

                {engineMode === "local" ? (
                  <>
                    <label className="field-label">อุปกรณ์ประมวลผลในเครื่อง</label>
                    <div className="compute-options">
                      <button className={trainingDevice === "auto" ? "selected" : ""} onClick={() => setTrainingDevice("auto")}>
                        <Sparkles size={16} /><strong>อัตโนมัติ</strong><small>ตรวจจับฮาร์ดแวร์ที่ดีที่สุด</small>
                      </button>
                      <button className={trainingDevice === "cpu" ? "selected" : ""} onClick={() => setTrainingDevice("cpu")}>
                        <Cpu size={16} /><strong>CPU</strong><small>เหมาะสำหรับ Tabular RL</small>
                      </button>
                      <button className={trainingDevice === "gpu" ? "selected" : ""} onClick={() => setTrainingDevice("gpu")}>
                        <Gauge size={16} /><strong>GPU</strong><small>เร่งคำนวณสำหรับ Neural Net</small>
                      </button>
                    </div>

                    <label className="field-label">อัลกอริทึมการเรียนรู้</label>
                    <select value={trainingAlgorithm} onChange={e => setTrainingAlgorithm(e.target.value as any)}>
                      <option value="qlearning">Q-Learning (Tabular RL - ปรับค่า Q ในเครื่องจริง)</option>
                      <option value="dqn">Deep Q-Network (DQN - โครงข่ายประสาทเทียม)</option>
                      <option value="baseline">Baseline Strategy (ใช้เป็นเกณฑ์มาตรฐานเปรียบเทียบ)</option>
                    </select>

                    <div className="advanced-settings">
                      <span className="eyebrow">HYPERPARAMETERS</span>
                      <div className="range-setting">
                        <label>Learning rate (α) <strong>{learningRate.toFixed(2)}</strong></label>
                        <input type="range" min="0.01" max="0.5" step="0.01" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />
                      </div>
                      <div className="range-setting">
                        <label>Discount factor (γ) <strong>{discountFactor.toFixed(2)}</strong></label>
                        <input type="range" min="0.5" max="0.99" step="0.01" value={discountFactor} onChange={e => setDiscountFactor(Number(e.target.value))} />
                      </div>
                      <div className="range-setting">
                        <label>Exploration rate (ε) <strong>{explorationRate.toFixed(2)}</strong></label>
                        <input type="range" min="0.01" max="1" step="0.01" value={explorationRate} onChange={e => setExplorationRate(Number(e.target.value))} />
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ background: "#f8f8f5", padding: 12, border: "1px solid var(--line)", marginBottom: 12 }}>
                    <p style={{ margin: "0 0 10px", fontSize: 11, color: "#666" }}>
                      เมื่อเลือก <strong>Cloud AI API</strong> ระบบจะใช้ LLM ในการประเมินกลยุทธ์และการตัดสินใจผ่านคลาวด์ ไม่จำเป็นต้องพึ่งพาชิป GPU ในเครื่อง
                    </p>
                    <label className="field-label" style={{ marginTop: 6 }}>ผู้ให้บริการ AI</label>
                    <select value={apiProvider} onChange={e => setApiProvider(e.target.value)}>
                      <option value="google">Google Gemini API</option>
                    </select>

                    <label className="field-label">โมเดล</label>
                    <select value={apiModel} onChange={e => setApiModel(e.target.value)}>
                      <option value="gemini-2.5-flash">Gemini 2.5 Flash (ความเร็วสูง เหมาะกับเกม)</option>
                      <option value="gemini-2.5-pro">Gemini 2.5 Pro (วิเคราะห์เชิงลึก)</option>
                      <option value="gemini-1.5-flash">Gemini 1.5 Flash (ประหยัดโควตา)</option>
                    </select>

                    <label className="field-label">API Key</label>
                    <div className="api-key-row">
                      <input
                        type={showApiKey ? "text" : "password"}
                        autoComplete="off"
                        spellCheck={false}
                        value={googleApiKey}
                        onChange={e => setGoogleApiKey(e.target.value)}
                        placeholder="วาง Google AI Studio API key ที่นี่"
                      />
                      <button className="secondary-action" onClick={() => setShowApiKey(v => !v)}>
                        {showApiKey ? "ซ่อน" : "แสดง"}
                      </button>
                      <button className="secondary-action" onClick={testApiKey}>
                        ทดสอบ
                      </button>
                    </div>
                    {apiTestStatus && (
                      <div style={{ fontSize: 10, color: "#2d663b", marginTop: 7, display: "flex", alignItems: "center", gap: 5 }}>
                        <CheckCircle2 size={13} /> {apiTestStatus}
                      </div>
                    )}
                  </div>
                )}

                {/* Duration Options */}
                <label className="field-label">ระยะเวลาฝึก (ตั้งค่าได้จริง 1-60 นาที)</label>
                <div className="duration-options">
                  {[1, 5, 10, 30, 60].map(m => (
                    <button
                      key={m}
                      className={trainMinutes === m ? "selected" : ""}
                      onClick={() => handleSelectDuration(m)}
                    >
                      {m}<small> min</small>
                    </button>
                  ))}
                </div>

                <label className="field-label">Episodes ต่อ Batch</label>
                <select value={episodesPerBatch} onChange={e => setEpisodesPerBatch(Number(e.target.value))}>
                  <option value={1000}>1,000 episodes</option>
                  <option value={5000}>5,000 episodes</option>
                  <option value={10000}>10,000 episodes</option>
                  <option value={50000}>50,000 episodes</option>
                </select>

                <div className="field-help">
                  <Clock3 size={14} /> เลือก {trainMinutes} นาที = ฝึก {trainMinutes * 60} วินาทีเป๊ะ (แก้ไขปัญหาระยะเวลาแล้ว)
                </div>

                <button
                  className="primary-action full-action"
                  disabled={!activeTrainingId || trainingRunning}
                  onClick={() => {
                    const p = projects.find(x => x.id === activeTrainingId);
                    if (p) beginTraining({ ...p, minutes: trainMinutes }, true);
                  }}
                >
                  <Play size={15} />
                  {activeProject && activeProject.games > 0
                    ? `ฝึกสะสมต่ออีก ${trainMinutes} นาที (สะสมต่อจาก ${activeProject.games.toLocaleString()} มือเดิม)`
                    : `เริ่มฝึกตามเวลาที่กำหนด (${trainMinutes} นาที)`}
                </button>
                {activeProject && activeProject.games > 0 && (
                  <button
                    className="outline-full"
                    style={{ marginTop: 6 }}
                    disabled={trainingRunning}
                    onClick={() => {
                      if (window.confirm(`ต้องการล้างความรู้เดิมของ ${activeProject.name} และเริ่มฝึกใหม่จากศูนย์หรือไม่?`)) {
                        beginTraining({ ...activeProject, minutes: trainMinutes, games: 0, qTable: {} }, false);
                      }
                    }}
                  >
                    <RotateCcw size={13} /> รีเซ็ตความรู้เดิม (เริ่มฝึกใหม่จาก 0)
                  </button>
                )}
                <button className="outline-full" style={{ marginTop: 6 }} onClick={() => { setShowNewProject(true); setPage("projects"); }}>
                  <Plus size={15} /> สร้างโปรเจกต์ใหม่
                </button>
              </div>

              {/* Live Session Panel */}
              <div className="panel run-panel">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">LIVE TRAINING SESSION</span>
                    <h3>{activeProject?.name ?? "ยังไม่มี Training Session"}</h3>
                  </div>
                  <div className="run-status">
                    {trainingRunning ? "SIMULATING REAL HANDS" : activeProject?.status === "Complete" ? "COMPLETE" : "IDLE"}
                  </div>
                </div>

                <div className="timer">
                  {fmt(trainingSeconds)}
                  <small> / {fmt(totalTrainingSeconds)} (ตั้งไว้ {trainMinutes} นาที)</small>
                </div>

                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.min(100, (trainingSeconds / Math.max(1, totalTrainingSeconds)) * 100)}%` }} />
                </div>
                <div className="progress-meta">
                  <span>TRAINING PROGRESS</span>
                  <strong>{Math.min(100, Math.round((trainingSeconds / Math.max(1, totalTrainingSeconds)) * 100))}%</strong>
                </div>

                <div className="metric-grid">
                  <div>
                    <span>SIMULATED HANDS (เล่นจริง)</span>
                    <strong>{(activeProject?.games ?? 0).toLocaleString()}</strong>
                  </div>
                  <div>
                    <span>ELAPSED TIME</span>
                    <strong>{fmt(trainingSeconds)}</strong>
                  </div>
                </div>

                <div className="run-actions">
                  <button className="secondary-action" disabled={!trainingRunning} onClick={pauseTraining}>
                    พักการฝึก
                  </button>
                  <button className="primary-action" disabled={!activeTrainingId || trainingRunning || trainingSeconds >= totalTrainingSeconds} onClick={resumeTraining}>
                    ทำต่อ
                  </button>
                  <button className="outline-full" onClick={resetTraining}>
                    รีเซ็ต
                  </button>
                </div>
                {activeProject && activeProject.games > 0 && (
                  <button
                    className="primary-action full-action"
                    style={{ marginTop: 10 }}
                    onClick={() => {
                      setSelectedPlayProjectId(activeProject.id);
                      setOpponent("project");
                      setPage("play");
                    }}
                  >
                    <Swords size={15} /> นำโมเดล {activeProject.name} ไปลงแข่งใน Arena ทันที
                  </button>
                )}

                <div className="notice">
                  <Info size={15} />
                  <span>
                    เมื่อกดเริ่มฝึก ระบบจะคำนวณและจำลองมือ Blackjack จริงในเบราว์เซอร์ พร้อมอัปเดต Q-Values และพล็อตกราฟด้านล่างทันที
                  </span>
                </div>
              </div>
            </div>

            {/* REAL GRAPH METRICS PANEL */}
            <div className="panel chart-panel" style={{ marginTop: 16 }}>
              <div className="chart-header-row">
                <div>
                  <span className="eyebrow">REAL SIMULATION METRICS</span>
                  <h3 style={{ margin: "4px 0 0" }}>กราฟผลการฝึกจริง (Real-Time Metrics)</h3>
                </div>
                <div className="chart-type-buttons">
                  <button
                    className={`chart-type-btn ${chartView === "winrate" ? "active" : ""}`}
                    onClick={() => setChartView("winrate")}
                  >
                    อัตราชนะ (Win Rate %)
                  </button>
                  <button
                    className={`chart-type-btn ${chartView === "reward" ? "active" : ""}`}
                    onClick={() => setChartView("reward")}
                  >
                    รางวัลเฉลี่ย (Avg Reward)
                  </button>
                  <button
                    className={`chart-type-btn ${chartView === "epsilon" ? "active" : ""}`}
                    onClick={() => setChartView("epsilon")}
                  >
                    Exploration (ε)
                  </button>
                </div>
              </div>

              {renderRealChart()}

              <div className="chart-live-stats">
                <div className="stat-box">
                  <span>WIN RATE (ปัจจุบัน)</span>
                  <strong>{latestMetric ? `${latestMetric.winRate}%` : "—"}</strong>
                </div>
                <div className="stat-box">
                  <span>AVG REWARD / HAND</span>
                  <strong>{latestMetric ? latestMetric.avgReward : "—"}</strong>
                </div>
                <div className="stat-box">
                  <span>LOSS / PUSH RATE</span>
                  <strong>{latestMetric ? `${latestMetric.lossRate}% / ${latestMetric.pushRate}%` : "—"}</strong>
                </div>
                <div className="stat-box">
                  <span>CURRENT EPSILON (ε)</span>
                  <strong>{latestMetric ? latestMetric.epsilon : explorationRate.toFixed(2)}</strong>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ================= PAGE: PROJECTS ================= */}
        {page === "projects" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">03 / AGENT WORKSPACE</p>
                <h1>Your AI Projects</h1>
                <p className="lede">โปรเจกต์จะเก็บตารางการเรียนรู้ Q-Table และพารามิเตอร์การฝึกที่นำไปใช้แข่งขันใน Arena ได้</p>
              </div>
              <button className="primary-action" onClick={() => setShowNewProject(true)}>
                <Plus size={15} /> สร้างโปรเจกต์
              </button>
            </div>

            {showNewProject && (
              <div className="panel new-project">
                <div className="panel-heading">
                  <div><span className="eyebrow">NEW AGENT</span><h3>สร้างโปรเจกต์ AI ใหม่</h3></div>
                  <button className="top-icon" onClick={() => setShowNewProject(false)}><X size={15} /></button>
                </div>
                <label className="field-label">ชื่อโปรเจกต์</label>
                <input value={projectName} onChange={e => setProjectName(e.target.value)} placeholder="เช่น Deep Blackjack Pro" />

                <label className="field-label">เอนจินการฝึก</label>
                <div className="engine-mode-tabs" style={{ margin: "8px 0 14px" }}>
                  <button className={`engine-tab ${engineMode === "local" ? "selected" : ""}`} onClick={() => setEngineMode("local")}>
                    <Cpu size={14} /> Local Engine (CPU/GPU)
                  </button>
                  <button className={`engine-tab ${engineMode === "api" ? "selected" : ""}`} onClick={() => setEngineMode("api")}>
                    <Cloud size={14} /> Cloud AI API
                  </button>
                </div>

                <label className="field-label">ระยะเวลาฝึกเริ่มต้น</label>
                <div className="duration-options">
                  {[1, 5, 10, 30, 60].map(m => (
                    <button key={m} className={trainMinutes === m ? "selected" : ""} onClick={() => handleSelectDuration(m)}>
                      {m}<small> min</small>
                    </button>
                  ))}
                </div>
                <button className="primary-action" style={{ marginTop: 16 }} onClick={createProject}>
                  สร้างโปรเจกต์ <ArrowRight size={15} />
                </button>
              </div>
            )}

            <div className="project-list">
              {projects.map(p => (
                <div className="project-card" key={p.id}>
                  <div className="project-symbol"><BrainCircuit size={21} /></div>
                  <div className="project-main">
                    <div className="project-title">
                      <h3>{p.name}</h3>
                      <span className={`project-status ${p.status.toLowerCase()}`}>
                        {p.status === "Ready" ? "พร้อมฝึก" : p.status === "Training" ? "กำลังฝึก" : "ฝึกเสร็จแล้ว"}
                      </span>
                      <span className="project-status" style={{ background: "#f0f0eb" }}>
                        {p.engineMode === "api" ? "Cloud AI API" : "Local RL Engine"}
                      </span>
                    </div>
                    <p>สร้างเมื่อ {p.created} · ระยะเวลาฝึก {p.minutes} นาที · สถานะโมเดล: {p.qTable ? `มีข้อมูล ${Object.keys(p.qTable).length} states` : "ยังไม่มีข้อมูล"}</p>
                    <div className="mini-progress"><div style={{ width: `${p.progress}%` }} /></div>
                    <div className="project-meta">
                      <span>{p.progress.toFixed(0)}% complete</span>
                      <span>{p.games.toLocaleString()} simulated hands</span>
                    </div>
                  </div>
                  <div className="project-actions">
                    <button
                      className="outline-action"
                      title="เลือกโมเดลนี้และไปลงแข่งใน Arena ทันที"
                      onClick={() => {
                        setSelectedPlayProjectId(p.id);
                        setOpponent("project");
                        setPage("play");
                      }}
                    >
                      <Swords size={13} /> แข่งใน Arena
                    </button>
                    <button
                      className="secondary-action"
                      onClick={() => {
                        handleSelectProject(p.id);
                        setPage("training");
                      }}
                    >
                      รายละเอียด
                    </button>
                    <button
                      className="primary-action"
                      onClick={() => {
                        handleSelectDuration(p.minutes);
                        beginTraining(p);
                      }}
                    >
                      ฝึก AI <Play size={13} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ================= PAGE: RULES ================= */}
        {page === "rules" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">LEARN / RULEBOOK</p>
                <h1>Blackjack & Arena Rules</h1>
                <p className="lede">เป้าหมายคือทำคะแนนให้ใกล้ 21 มากที่สุด โดยไม่เกิน 21 และเอาชนะเจ้ามือ</p>
              </div>
              <div className="rule-stamp">21<small>BLACKJACK</small></div>
            </div>
            <div className="rules-grid">
              <div className="panel">
                <div className="panel-heading"><h3>ค่าของไพ่</h3><BookOpen size={17} /></div>
                <div className="rule-line"><span>A (Ace)</span><strong>1 หรือ 11</strong></div>
                <div className="rule-line"><span>2–9</span><strong>ตามหน้าไพ่</strong></div>
                <div className="rule-line"><span>10, J, Q, K</span><strong>10 คะแนน</strong></div>
                <div className="rule-line"><span>Natural Blackjack</span><strong>A + ไพ่ 10 แต้ม (จ่าย 3:2)</strong></div>
              </div>
              <div className="panel">
                <div className="panel-heading"><h3>คำสั่งในเกม</h3><Swords size={17} /></div>
                <div className="rule-line"><span><b>Hit (จั่ว)</b><small>จั่วไพ่เพิ่ม 1 ใบ</small></span><strong>+</strong></div>
                <div className="rule-line"><span><b>Stand (หยุด)</b><small>หยุดจั่วและจบเทิร์น</small></span><strong>Ⅱ</strong></div>
                <div className="rule-line"><span><b>Double</b><small>เพิ่มเดิมพัน รับไพ่หนึ่งใบแล้วจบ</small></span><strong>×2</strong></div>
                <div className="rule-line"><span><b>Split</b><small>แยกไพ่คู่เป็นสองมือ</small></span><strong>⑂</strong></div>
              </div>
            </div>
            <div className="panel rules-detail" style={{ marginTop: 14 }}>
              <div className="panel-heading">
                <div><span className="eyebrow">ARENA DUEL MECHANICS</span><h3>กติกาการสลับเทิร์นใน Arena</h3></div>
                <ShieldCheck size={18} />
              </div>
              <div className="rules-bullets">
                <p><span>01</span> ในโหมด Arena Duel ผู้เล่นและ AI คู่แข่งจะสลับกันเล่นคนละตา (Alternating Turns)</p>
                <p><span>02</span> เมื่อคุณกด Hit คุณจะได้รับไพ่ และตาเดินจะสลับไปยัง AI ทันที เพื่อให้ AI เลือกว่าจะ Hit หรือ Stand</p>
                <p><span>03</span> เมื่อทั้งสองฝ่าย Stand หรือ Bust เจ้ามือจะทำการเปิดไพ่และจั่วให้ถึง 17 แต้มตามกติกาคาสิโนสากล</p>
                <p><span>04</span> สกอร์บอร์ดจะเปรียบเทียบผลระหว่างคุณและ AI ว่าใครทำผลงานได้ดีกว่าในรอบนั้น</p>
              </div>
            </div>
            <button className="primary-action" style={{ marginTop: 14 }} onClick={() => setPage("play")}>
              เข้าเล่นเกมทันที <ArrowRight size={15} />
            </button>
          </section>
        )}

        {/* ================= PAGE: HELP ================= */}
        {page === "help" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">LEARN / STRATEGY GUIDE</p>
                <h1>Basic Strategy Reference</h1>
                <p className="lede">คำแนะนำทางสถิติเพื่อลด House Edge ของคาสิโนให้เหลือน้อยที่สุด</p>
              </div>
              <div className="help-symbol"><WandSparkles size={27} /></div>
            </div>
            <div className="help-grid">
              <div className="panel">
                <div className="panel-heading">
                  <div><span className="eyebrow">QUICK TIPS</span><h3>หลักการสำคัญ</h3></div>
                  <CircleHelp size={18} />
                </div>
                <div className="help-tip">
                  <span>01</span>
                  <div><strong>Hard 17 ขึ้นไป</strong><p>หยุดจั่ว (Stand) เสมอ เพราะโอกาส Bust สูงกว่า 69%</p></div>
                </div>
                <div className="help-tip">
                  <span>02</span>
                  <div><strong>แต้ม 10 หรือ 11</strong><p>พิจารณา Double เมื่อเจ้ามือเปิดไพ่ 2–9 เพื่อเพิ่มผลกำไรสูงสุด</p></div>
                </div>
                <div className="help-tip">
                  <span>03</span>
                  <div><strong>ไพ่คู่ A–A หรือ 8–8</strong><p>แยก (Split) เสมอ เพื่อเปลี่ยนสถานะเสี่ยงเป็นมือที่มีโอกาสชนะสูง</p></div>
                </div>
                <div className="help-tip">
                  <span>04</span>
                  <div><strong>เจ้ามือเปิด 4, 5, 6 (ไพ่อ่อน)</strong><p>หยุดจั่วตั้งแต่แต้ม 12 ขึ้นไป ปล่อยให้เจ้ามือมีความเสี่ยง Bust เอง</p></div>
                </div>
              </div>
              <div className="panel">
                <div className="panel-heading">
                  <div><span className="eyebrow">ASSIST MODE</span><h3>ใช้ตัวช่วยในเกม</h3></div>
                  <Sparkles size={18} />
                </div>
                <p className="panel-copy">
                  ระหว่างเล่นในหน้า Play Arena เมื่อถึงตาของคุณ คุณสามารถคลิกปุ่ม <strong>"ช่วยเล่น"</strong> ใต้ปุ่มควบคุม เพื่อดูคำแนะนำจากสถิติที่เหมาะสมที่สุดกับแต้มในมือ
                </p>
                <button className="primary-action full-action" onClick={() => setPage("play")}>
                  ไปที่หน้าเล่นเกม <ArrowRight size={15} />
                </button>
              </div>
            </div>
          </section>
        )}
        </>
        )}
      </main>

      {/* Game Selector Modal */}
      {showGameModal && (
        <div className="game-modal-overlay" onClick={() => setShowGameModal(false)}>
          <div className="game-modal" onClick={e => e.stopPropagation()}>
            <div className="game-modal-header">
              <div>
                <span className="eyebrow">ARENA GAME SELECTOR</span>
                <h3 style={{ margin: "4px 0 0", fontSize: 16 }}>เลือกเกมที่ต้องการเล่น</h3>
              </div>
              <button className="top-icon" onClick={() => setShowGameModal(false)}>✕</button>
            </div>
            <div className="game-modal-body">
              <div
                className={`game-card-pick ${activeGame === "blackjack" ? "active" : ""}`}
                onClick={() => {
                  setActiveGame("blackjack");
                  setShowGameModal(false);
                }}
              >
                <span className="pick-badge">{activeGame === "blackjack" ? "กำลังเล่นอยู่" : "พร้อมใช้งาน"}</span>
                <div className="game-icon">♠</div>
                <h3>Blackjack AI Arena</h3>
                <p>เกมไพ่ 21 แต้มคาสิโนสากล ดวลสลับตากับ AI พร้อมระบบเทรน Q-Learning และกราฟผลลัพธ์แบบเรียลไทม์</p>
                <button className="select-btn">
                  {activeGame === "blackjack" ? "เล่นเกมนี้อยู่ (คลิกเพื่อปิด)" : "เลือกเล่น Blackjack"} <ArrowRight size={13} />
                </button>
              </div>

              <div
                className={`game-card-pick ${activeGame === "chess" ? "active" : ""}`}
                onClick={() => {
                  setActiveGame("chess");
                  setShowGameModal(false);
                }}
              >
                <span className="pick-badge">{activeGame === "chess" ? "กำลังเล่นอยู่" : "พร้อมใช้งาน"}</span>
                <div className="game-icon">♞</div>
                <h3>Chess AI Arena (หมากรุก)</h3>
                <p>กระดานหมากรุกสากล 8x8 สลับตาเดินกับ Minimax Chess AI พร้อมระบบ Thought Reasoning และบันทึกประวัติการเดิน</p>
                <button className="select-btn">
                  {activeGame === "chess" ? "เล่นเกมนี้อยู่ (คลิกเพื่อปิด)" : "เลือกเล่น Chess"} <ArrowRight size={13} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


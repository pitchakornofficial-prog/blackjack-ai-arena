import { useEffect, useRef, useState } from "react";
import {
  BrainCircuit, Cpu, FolderKanban, Pause, Play, Plus, RotateCcw, ShieldCheck, Sparkles, Swords, Trash2,
  BookOpen, Maximize2, Minimize2
} from "lucide-react";
import {
  applyAiMove, applyPlayerMove, newTTTGame, TTTOpponent, TTTState
} from "./tttEngine";
import {
  ALGORITHM_INFO, TTTAlgorithm, TTTMetricPoint, TTTTrainingProject, TTTQTable,
  applyProjectAiMove, initialTTTProjects, knownStatesCount, trainTTTBatch
} from "./tttTraining";
import { STORAGE_KEYS, isProjectRecordArray, loadStoredData, saveStoredData } from "./persistence";

type TTTViewProps = {
  page: string;
  setPage: (p: any) => void;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
};

const OPPONENTS: { key: TTTOpponent; name: string; desc: string; tag: string }[] = [
  { key: "easy", name: "Beginner AI", desc: "สุ่มเดิน อ่อนแรง เหมาะกับผู้เริ่มต้น", tag: "RANDOM · EASY" },
  { key: "medium", name: "Thinker AI", desc: "ผสมไหวพริบ บางตาใช้ minimax", tag: "65% MINIMAX · MEDIUM" },
  { key: "unbeatable", name: "Perfect AI", desc: "Minimax เต็มรูปแบบ แพ้ไม่ได้เลยจริง ๆ", tag: "MINIMAX · UNBEATABLE" }
];

/* ---------- Restore / revive TTT projects from localStorage ---------- */

function reviveTTTProject(raw: TTTTrainingProject): TTTTrainingProject {
  const qTable: TTTQTable = (typeof raw.qTable === "object" && raw.qTable !== null && !Array.isArray(raw.qTable))
    ? (raw.qTable as TTTQTable)
    : {};
  const epsilon = typeof raw.epsilon === "number" && raw.epsilon > 0 ? raw.epsilon : 0.3;
  return {
    ...initialTTTProjects[0],
    ...raw,
    // โปรเจกต์ที่กำลังเทรนตอนปิดหน้า → กลับมาเป็น Ready (คุมเหมือนเกมอื่น ๆ)
    status: raw.status === "Training" ? ("Ready" as const) : raw.status,
    qTable,
    epsilon,
    winRate: typeof raw.winRate === "number" ? raw.winRate : 33.3,
    lossRate: typeof raw.lossRate === "number" ? raw.lossRate : 33.3,
    drawRate: typeof raw.drawRate === "number" ? raw.drawRate : 33.3
  };
}

function restoreTTTProjects(): TTTTrainingProject[] {
  const saved = loadStoredData<TTTTrainingProject[]>(STORAGE_KEYS.tttProjects, isProjectRecordArray);
  if (!saved) return initialTTTProjects;
  return saved.map(reviveTTTProject);
}

export default function TTTView({ page, setPage, isFullscreen, onToggleFullscreen }: TTTViewProps) {
  // Board / play state
  const [ttt, setTtt] = useState<TTTState>(() => newTTTGame());
  const [tttOpponent, setTttOpponent] = useState<TTTOpponent>("medium");
  const [tttStats, setTttStats] = useState({ playerWins: 0, aiWins: 0, draws: 0 });

  // Training state (projects restored from localStorage across reloads)
  const [tttProjects, setTttProjects] = useState<TTTTrainingProject[]>(restoreTTTProjects);
  const [activeTrainingId, setActiveTrainingId] = useState<number>(1);
  const [selectedPlayProjectId, setSelectedPlayProjectId] = useState<number>(1);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [trainingBaseSeconds, setTrainingBaseSeconds] = useState(0);
  const [trainingSeconds, setTrainingSeconds] = useState(0);
  const [trainMinutes, setTrainMinutes] = useState(1);
  const [batchSize, setBatchSize] = useState(100);
  const [learningRate, setLearningRate] = useState(0.12);
  const [discountFactor, setDiscountFactor] = useState(0.85);
  const [epsilon, setEpsilon] = useState(0.3);
  const [metricHistory, setMetricHistory] = useState<TTTMetricPoint[]>([]);
  const [chartView, setChartView] = useState<"winrate" | "reward" | "epsilon">("winrate");
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("Nova O-Model");
  const [newProjectAlgo, setNewProjectAlgo] = useState<TTTAlgorithm>("selfplay-rl");

  const totalTrainingSeconds = trainMinutes * 60;
  const activeProject = tttProjects.find(p => p.id === activeTrainingId) ?? null;
  const latestMetric = metricHistory.length > 0 ? metricHistory[metricHistory.length - 1] : null;
  const selectedPlayProject = tttProjects.find(p => p.id === selectedPlayProjectId) ?? tttProjects[0] ?? null;

  // Persist projects whenever they change
  useEffect(() => {
    saveStoredData(STORAGE_KEYS.tttProjects, tttProjects);
  }, [tttProjects]);

  // บันทึกผลเมื่อรอบจบ (ทริกเกอร์ครั้งเดียวต่อรอบผ่าน round)
  useEffect(() => {
    if (ttt.phase === "over" && ttt.round > 1 && ttt.winner) {
      if (ttt.winner === "player") setTttStats(s => ({ ...s, playerWins: s.playerWins + 1 }));
      else if (ttt.winner === "ai") setTttStats(s => ({ ...s, aiWins: s.aiWins + 1 }));
      else if (ttt.winner === "draw") setTttStats(s => ({ ...s, draws: s.draws + 1 }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ttt.round]);

  // Refs ล่าสุดสำหรับ AI move ให้ timer ไม่ถูกรีเซ็ตเมื่อเทรนทำงานอยู่เบื้องหลัง
  const opponentRef = useRef(tttOpponent);
  const playProjectRef = useRef(selectedPlayProject);
  useEffect(() => { opponentRef.current = tttOpponent; }, [tttOpponent]);
  useEffect(() => { playProjectRef.current = selectedPlayProject; }, [selectedPlayProject]);

  // AI เดินเมื่อถึงตา (หน่วงเวลาเล็กน้อยให้เห็นการคิด) — deps ยึด phase เดียว
  useEffect(() => {
    if (ttt.phase !== "ai") return;
    const t = window.setTimeout(() => {
      setTtt(g => opponentRef.current === "project"
        ? applyProjectAiMove(g, playProjectRef.current?.qTable)
        : applyAiMove(g, opponentRef.current));
    }, 600);
    return () => window.clearTimeout(t);
  }, [ttt.phase]);

  /* ---------------- Standard wall-clock training loop (หยุดได้) ---------------- */
  useEffect(() => {
    if (!trainingRunning || !activeTrainingId || trainingStartedAt === null) return;

    const timer = window.setInterval(() => {
      const elapsed = trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000);
      const limit = totalTrainingSeconds;
      const next = Math.min(elapsed, limit);
      setTrainingSeconds(next);

      setTttProjects(old => old.map(p => {
        if (p.id !== activeTrainingId) return p;
        const result = trainTTTBatch(p.qTable, batchSize, learningRate, discountFactor, p.epsilon, p.algorithm);
        const newEpsilon = result.updatedEpsilon;
        setMetricHistory(prev => {
          const point: TTTMetricPoint = {
            step: prev.length + 1,
            timeSeconds: next,
            totalGames: (prev[prev.length - 1]?.totalGames ?? 0) + batchSize,
            winRate: Number(result.winRate.toFixed(1)),
            lossRate: Number(result.lossRate.toFixed(1)),
            drawRate: Number(result.drawRate.toFixed(1)),
            avgReward: Number(result.avgReward.toFixed(3)),
            epsilon: Number(newEpsilon.toFixed(3)),
            knownStates: result.knownStates
          };
          return [...prev, point];
        });
        return {
          ...p,
          games: p.games + batchSize,
          qTable: result.updatedQTable,
          epsilon: newEpsilon,
          winRate: Number(result.winRate.toFixed(1)),
          lossRate: Number(result.lossRate.toFixed(1)),
          drawRate: Number(result.drawRate.toFixed(1)),
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    trainingRunning, activeTrainingId, trainingStartedAt, trainingBaseSeconds,
    totalTrainingSeconds, batchSize, learningRate, discountFactor
  ]);

  /* ---------------- Training control functions ---------------- */

  function beginTraining(project: TTTTrainingProject, minutes: number) {
    setTttProjects(old => old.map(p => p.id === project.id
      ? { ...p, minutes, status: "Training", progress: 0 }
      : p));
    setActiveTrainingId(project.id);
    setTrainMinutes(minutes);
    setTrainingSeconds(0);
    setTrainingBaseSeconds(0);
    setTrainingStartedAt(Date.now());
    setTrainingRunning(true);
    if (project.games === 0) setMetricHistory([]);
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
      setTttProjects(old => old.map(p => p.id === activeTrainingId
        ? { ...p, status: "Ready", progress: 0, games: 0, qTable: {}, epsilon: 0.3 }
        : p));
    }
  }

  function handleCreateProject() {
    if (!newProjectName.trim()) return;
    const created: TTTTrainingProject = {
      id: Date.now(),
      name: newProjectName.trim(),
      minutes: 1,
      status: "Ready",
      progress: 0,
      games: 0,
      created: "สร้างใหม่",
      algorithm: newProjectAlgo,
      qTable: {},
      epsilon: 0.3,
      winRate: 33.3,
      lossRate: 33.3,
      drawRate: 33.3
    };
    setTttProjects(prev => [created, ...prev]);
    setActiveTrainingId(created.id);
    setSelectedPlayProjectId(created.id);
    setShowNewProject(false);
    setNewProjectName("Nova O-Model");
  }

  function openTrainingFor(project: TTTTrainingProject) {
    setActiveTrainingId(project.id);
    setTrainMinutes(project.minutes || 1);
    setMetricHistory([]);
    setPage("training");
  }

  function handleDeleteProject(id: number) {
    if (tttProjects.length <= 1) return;
    if (activeTrainingId === id) {
      pauseTraining();
      setTrainingSeconds(0);
      setTrainingBaseSeconds(0);
      setMetricHistory([]);
    }
    setTttProjects(old => old.filter(p => p.id !== id));
  }

  function selectPlayProject(id: number) {
    setSelectedPlayProjectId(id);
    setTttOpponent("project");
    setTtt(newTTTGame());
  }

  /* ---------------- Play helpers ---------------- */

  function clickCell(idx: number) {
    setTtt(g => applyPlayerMove(g, idx));
  }

  function resetRound() {
    setTtt(newTTTGame());
  }

  function fmt(seconds: number) {
    return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  }

  /* ---------------- Sparkline chart (metric history) ---------------- */
  function renderSparkline() {
    const points = metricHistory;
    if (points.length < 2) {
      return (
        <div className="notice" style={{ marginTop: 8 }}>
          <Sparkles size={15} />
          <span>ยังไม่มีข้อมูลกราฟ — เริ่มฝึกอย่างน้อย 2 tick เพื่อพล็อตเมตริกแบบเรียลไทม์</span>
        </div>
      );
    }
    const W = 640;
    const H = 120;
    const n = points.length;
    const series = chartView === "winrate"
      ? points.map(p => p.winRate)
      : chartView === "reward"
        ? points.map(p => p.avgReward)
        : points.map(p => p.epsilon);
    const min = chartView === "reward" ? Math.min(0, ...series) : 0;
    const max = Math.max(...series, chartView === "reward" ? 0.1 : 1);
    const x = (i: number) => (i / Math.max(1, n - 1)) * W;
    const y = (v: number) => H - ((v - min) / (max - min || 1)) * (H - 12) - 6;
    const path = series.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 120, marginTop: 8 }}>
        <line x1={0} y1={H - 6} x2={W} y2={H - 6} stroke="#e1e1dc" strokeWidth={1} />
        <path d={path} fill="none" stroke="#495057" strokeWidth={2} />
        <circle cx={x(n - 1)} cy={y(series[n - 1])} r={3} fill="#495057" />
      </svg>
    );
  }

  const phase = ttt.phase;

  return (
    <>
      {page === "home" && (
        <section className="page">
          <div className="welcome-row">
            <div>
              <p className="eyebrow">STRATEGY MEETS INSTINCT</p>
              <h1>Tic. Tac. <em>Toe.</em></h1>
              <p className="lede">สนามทดลองกลยุทธ์ 3x3 ดวลกับ AI หลายระดับ ตั้งแต่โมเดลที่คุณฝึกเอง จนถึง Minimax ที่แพ้ไม่ได้</p>
            </div>
            <div className="hero-stamp">
              <span>⭘</span>
              <small>EST. 3x3</small>
            </div>
          </div>
          <div className="quick-grid">
            <button className="feature-card dark-card" onClick={() => setPage("play")}>
              <div className="feature-top"><span>01 / PLAY</span></div>
              <div className="feature-icon"><Swords size={23} /></div>
              <h2>ดวลกับ AI</h2>
              <p>ลงสนาม 9 ช่อง ปะทะ AI ตั้งแต่โมเดลที่คุณฝึกเองถึงแพ้ไม่ได้</p>
              <div className="feature-link">ENTER ARENA</div>
            </button>
            <button className="feature-card" onClick={() => setPage("training")}>
              <div className="feature-top"><span>02 / TRAIN</span></div>
              <div className="feature-icon"><BrainCircuit size={23} /></div>
              <h2>เทรน AI (หยุดได้)</h2>
              <p>ฝึกโมเดล 3 แบบ Self-Play · Minimax Teacher · Random Drill — พัก/ทำต่อได้ทุกเมื่อ</p>
              <div className="feature-link">OPEN LAB</div>
            </button>
            <button className="feature-card" onClick={() => setPage("projects")}>
              <div className="feature-top"><span>03 / MODELS</span></div>
              <div className="feature-icon"><FolderKanban size={23} /></div>
              <h2>โปรเจกต์โมเดล</h2>
              <p>จัดการโมเดลทุกตัว ล้างความรู้ หรือส่งลงสนามประลอง</p>
              <div className="feature-link">VIEW MODELS</div>
            </button>
            <button className="feature-card" onClick={() => setPage("rules")}>
              <div className="feature-top"><span>04 / GUIDE</span></div>
              <div className="feature-icon"><Sparkles size={23} /></div>
              <h2>วิธีเล่นและกติกา</h2>
              <p>ทำความเข้าใจกติกาและแนวทางการเล่นก่อนลงสนาม</p>
              <div className="feature-link">VIEW GUIDE</div>
            </button>
          </div>
        </section>
      )}

      {page === "play" && (
        <section className="page play-page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">01 / PLAY ARENA</p>
              <h1>Tic Tac Toe Arena</h1>
              <p className="lede">คุณเล่น X — กดช่องว่างเพื่อวาง แล้ว AI จะเดิน O สลับกันจนมีผู้ชนะหรือเสมอ</p>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div className="scoreboard-badge">
                <span>ARENA DUEL:</span>
                <strong>คุณ {tttStats.playerWins}</strong> · <strong>AI {tttStats.aiWins}</strong> · เสมอ {tttStats.draws}
              </div>
              {onToggleFullscreen && (
                <button
                  className={`fullscreen-btn ${isFullscreen ? "active" : ""}`}
                  onClick={onToggleFullscreen}
                  title="เปิด/ปิดโหมดเต็มจอ"
                >
                  {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  <span>{isFullscreen ? "ออกจากเต็มจอ" : "เต็มจอ"}</span>
                </button>
              )}
              <button className="ghost-button" onClick={() => setPage("rules")}>
                <BookOpen size={15} /> กติกา
              </button>
            </div>
          </div>

          <div className="opponent-grid four-cols">
            <button
              className={`opponent-card ${tttOpponent === "project" ? "chosen" : ""}`}
              onClick={() => selectPlayProject(selectedPlayProject?.id ?? tttProjects[0].id)}
              title="คู่แข่งจากโปรเจกต์ที่คุณเทรนเอง"
            >
              <span className="opponent-icon"><Cpu /></span>
              <span className="opponent-tag">FROM YOUR LAB</span>
              <h3>{selectedPlayProject?.name ?? "โมเดลของคุณ"}</h3>
              <p>เดินด้วย Q-Table ที่เทรนมา {selectedPlayProject ? selectedPlayProject.games.toLocaleString() : 0} เกม · จำ {selectedPlayProject ? knownStatesCount(selectedPlayProject.qTable) : 0} สถานะ</p>
              <small>{selectedPlayProject ? ALGORITHM_INFO[selectedPlayProject.algorithm].tag : "MODEL"}</small>
            </button>
            {OPPONENTS.map(o => (
              <button
                key={o.key}
                className={`opponent-card ${tttOpponent === o.key ? "chosen" : ""}`}
                onClick={() => { setTttOpponent(o.key); resetRound(); }}
              >
                <span className="opponent-icon"><ShieldCheck /></span>
                <span className="opponent-tag">{o.tag}</span>
                <h3>{o.name}</h3>
                <p>{o.desc}</p>
                <small>{o.tag}</small>
              </button>
            ))}
          </div>

          <div className="game-layout">
            <div className="game-board">
              {/* Turn banner */}
              <div className={`turn-banner ${phase === "player" ? "player-turn" : phase === "ai" ? "ai-turn" : "dealer-turn"}`}>
                <div>
                  {phase === "over" ? (
                    <span>
                      {ttt.winner === "player"
                        ? <strong>🎉 คุณชนะ! — เยี่ยมมาก</strong>
                        : ttt.winner === "ai"
                          ? <strong>💻 AI ชนะรอบนี้</strong>
                          : <strong>🤝 เสมอ!</strong>}
                    </span>
                  ) : phase === "player" ? (
                    <span>🟢 <strong>ตาของคุณ:</strong> กดช่องว่างเพื่อวาง X</span>
                  ) : (
                    <span>🟡 <strong>ตาของ AI:</strong> กำลังคิดวิเคราะห์...</span>
                  )}
                </div>
                <span style={{ font: "9px var(--mono)", color: "#888" }}>Round {ttt.round}</span>
              </div>

              {/* Board */}
              <div className="ttt-stage">
                <div className="ttt-board-wrap">
                  <div className="ttt-grid">
                    {ttt.board.map((c, i) => {
                      const isWin = ttt.winLine?.includes(i);
                      return (
                        <button
                          key={i}
                          className={`ttt-cell ${isWin ? "ttt-cell-win" : ""} ${c === "X" ? "mark-x" : c === "O" ? "mark-o" : ""}`}
                          disabled={phase !== "player" || c !== ""}
                          onClick={() => clickCell(i)}
                          aria-label={`ช่อง ${i + 1}`}
                        >
                          {c}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {ttt.aiThought && (
                  <div className="ai-thought-box">
                    <BrainCircuit size={15} />
                    <div>{ttt.aiThought}</div>
                  </div>
                )}
                {tttOpponent === "project" && (
                  <div className="ai-thought-box" style={{ opacity: 0.85 }}>
                    <Cpu size={15} />
                    <div><strong>โหมดโมเดลของคุณ:</strong> กำลังใช้ลาเทนต์จากไลบรารีที่คุณฝึก (สถานะ {selectedPlayProject ? knownStatesCount(selectedPlayProject.qTable) : 0})</div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="controls">
                <button className="primary-action action-button" onClick={resetRound}>
                  <RotateCcw size={15} /> เริ่มรอบใหม่
                </button>
                <button className="secondary-action action-button" onClick={() => setTtt(newTTTGame())}>
                  <Sparkles size={15} /> เริ่มเซสชันใหม่
                </button>
              </div>
            </div>

            <aside className="side-panel">
              <span className="eyebrow">MATCH INFO</span>
              <h3>สถานะการดวล</h3>
              <div className="setting-row"><span>AI คู่แข่ง</span><strong>{tttOpponent === "project" ? selectedPlayProject?.name : OPPONENTS.find(o => o.key === tttOpponent)?.name}</strong></div>
              <div className="setting-row"><span>กติกา</span><strong>3 ช่องเรียงกันชนะ</strong></div>
              <div className="setting-row"><span>ผลรวม</span><strong>{tttStats.playerWins}W - {tttStats.aiWins}L - {tttStats.draws}D</strong></div>
              <div className="setting-row"><span>AI Nodes (ล่าสุด)</span><strong>{ttt.thinkDepth.toLocaleString()}</strong></div>
              <button className="outline-full" onClick={() => setTtt(newTTTGame())}>
                <RotateCcw size={14} /> เริ่มเซสชันใหม่
              </button>
            </aside>
          </div>
        </section>
      )}

      {/* ================= TRAINING ================= */}
      {page === "training" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">02 / AI TRAINING LAB</p>
              <h1>Tic Tac Toe Training Lab</h1>
              <p className="lede">ฝึกโมเดลด้วยเกมจริงแบบ Self-Play เลือกอัลกอริทึมได้ 3 แบบ เทรน/พัก/ทำต่อ/รีเซ็ตความรู้ได้ทุกเมื่อ</p>
            </div>
            <div className="training-chip">
              <span className={trainingRunning ? "pulse-dot" : ""} />
              {trainingRunning ? "TRAINING RUNNING (REAL SELF-PLAY)" : "SIMULATION LAB"}
            </div>
          </div>

          <div className="training-grid">
            {/* Config panel */}
            <div className="panel training-config">
              <div className="panel-heading">
                <div><span className="eyebrow">TRAINING CONFIG</span><h3>ตั้งค่าการฝึก</h3></div>
                <BrainCircuit size={18} />
              </div>

              <label className="field-label">โปรเจกต์ที่ฝึก</label>
              <select value={activeTrainingId} onChange={e => setActiveTrainingId(Number(e.target.value))}>
                {tttProjects.map(p => (
                  <option key={p.id} value={p.id}>{p.name} ({ALGORITHM_INFO[p.algorithm].name})</option>
                ))}
              </select>

              <label className="field-label">รูปแบบการเทรน (อัพเดตกับโปรเจกต์ทันที)</label>
              <div className="engine-mode-tabs">
                {(Object.keys(ALGORITHM_INFO) as TTTAlgorithm[]).map(a => (
                  <button
                    key={a}
                    className={`engine-tab ${activeProject?.algorithm === a ? "selected" : ""}`}
                    onClick={() => {
                      const id = activeProject?.id;
                      if (id == null || trainingRunning) return;
                      setTttProjects(old => old.map(p => p.id === id ? { ...p, algorithm: a } : p));
                    }}
                    disabled={trainingRunning}
                  >
                    <Cpu size={14} /> {ALGORITHM_INFO[a].name}
                  </button>
                ))}
              </div>
              {activeProject && (
                <div className="field-help" style={{ display: "flex", gap: 6 }}>
                  <Sparkles size={13} /> {ALGORITHM_INFO[activeProject.algorithm].desc}
                </div>
              )}

              <label className="field-label">ระยะเวลาฝึก (1-10 นาที)</label>
              <div className="duration-options">
                {[1, 2, 5, 10].map(m => (
                  <button
                    key={m}
                    className={trainMinutes === m ? "selected" : ""}
                    onClick={() => {
                      setTrainMinutes(m);
                      if (activeTrainingId) {
                        setTttProjects(prev => prev.map(p => p.id === activeTrainingId ? { ...p, minutes: m } : p));
                      }
                    }}
                  >
                    {m}<small> min</small>
                  </button>
                ))}
              </div>

              <label className="field-label">เกมจริงต่อ Batch (SIMULATED SELF-PLAY GAMES)</label>
              <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}>
                <option value={50}>50 games / batch</option>
                <option value={100}>100 games / batch</option>
                <option value={300}>300 games / batch</option>
                <option value={500}>500 games / batch</option>
              </select>

              <label className="field-label">Learning rate: {learningRate.toFixed(2)}</label>
              <input type="range" min={0.01} max={0.5} step={0.01} value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />
              <label className="field-label">Discount factor: {discountFactor.toFixed(2)}</label>
              <input type="range" min={0.5} max={0.99} step={0.01} value={discountFactor} onChange={e => setDiscountFactor(Number(e.target.value))} />
              <label className="field-label">เป้าหมาย ε เริ่มต้น: {epsilon.toFixed(2)} (decay 0.98/batch)</label>
              <input type="range" min={0.05} max={0.8} step={0.05} value={epsilon} onChange={e => setEpsilon(Number(e.target.value))} />
              {activeProject && (
                <div className="field-help">
                  <Cpu size={14} /> โปรเจกต์ใช้ ε จริงตอนนี้ = {activeProject.epsilon.toFixed(3)} · จำได้ {knownStatesCount(activeProject.qTable)} สถานะ
                </div>
              )}

              <div className="field-help">
                Hacker? ทุกตาที่เทรนคือเกมจริงบนกระดาน 3x3 ที่โมเดลเล่นกับตัวเอง/ครู Minimax/คู่สุ่ม จากนั้นสะสมค่า Q จริง ๆ (อัพเดตทุก batch)
              </div>

              <button
                className="primary-action full-action"
                disabled={!activeTrainingId || trainingRunning}
                onClick={() => {
                  if (activeProject) beginTraining({ ...activeProject }, trainMinutes);
                }}
              >
                <Play size={15} />
                {activeProject && activeProject.games > 0
                  ? `ฝึกสะสมต่ออีก ${trainMinutes} นาที (สะสมต่อจาก ${activeProject.games.toLocaleString()} เกมเดิม)`
                  : `เริ่มฝึกตามเวลาที่กำหนด (${trainMinutes} นาที)`}
              </button>
              {activeProject && activeProject.games > 0 && (
                <button
                  className="outline-full"
                  style={{ marginTop: 6 }}
                  disabled={trainingRunning}
                  onClick={() => {
                    if (window.confirm(`ล้างความรู้ของ ${activeProject.name} (Q-Table ทั้งหมด) แล้วเริ่มฝึกใหม่จากศูนย์หรือไม่?`)) {
                      setTttProjects(old => old.map(p => p.id === activeProject.id
                        ? { ...p, games: 0, qTable: {}, epsilon, progress: 0, status: "Ready" as const }
                        : p));
                      setMetricHistory([]);
                    }
                  }}
                >
                  <RotateCcw size={13} /> ล้างความรู้เดิม (เริ่มฝึกใหม่จาก 0)
                </button>
              )}
              <button className="outline-full" style={{ marginTop: 6 }} onClick={() => { setShowNewProject(true); setPage("projects"); }}>
                <Plus size={15} /> สร้างโปรเจกต์ TTT ใหม่
              </button>
            </div>

            {/* Live training panel */}
            <div className="panel run-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">LIVE TRAINING SESSION</span>
                  <h3>{activeProject?.name ?? "ยังไม่มี Training Session"}</h3>
                </div>
                <div className="run-status">
                  {trainingRunning ? "REAL SELF-PLAY RUNNING" : activeProject?.status === "Complete" ? "COMPLETE" : "IDLE"}
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
                  <span>REAL GAMES PLAYED</span>
                  <strong>{(activeProject?.games ?? 0).toLocaleString()}</strong>
                </div>
                <div>
                  <span>ELAPSED TIME</span>
                  <strong>{fmt(trainingSeconds)}</strong>
                </div>
                <div>
                  <span>KNOWN STATES</span>
                  <strong>{activeProject ? knownStatesCount(activeProject.qTable) : 0}</strong>
                </div>
              </div>

              <div className="run-actions">
                <button className="secondary-action" disabled={!trainingRunning} onClick={pauseTraining}>
                  <Pause size={14} /> พักการฝึก
                </button>
                <button className="primary-action" disabled={!activeTrainingId || trainingRunning || trainingSeconds >= totalTrainingSeconds} onClick={resumeTraining}>
                  <Play size={14} /> ทำต่อ
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
                    selectPlayProject(activeProject.id);
                    setTtt(newTTTGame());
                    setPage("play");
                  }}
                >
                  <Swords size={15} /> นำโมเดล {activeProject.name} ไปประลองใน Arena ทันที
                </button>
              )}

              <div className="notice" style={{ marginTop: 18 }}>
                <Sparkles size={15} />
                <span>
                  ระหว่างเวลาฝึก ระบบจะเล่นเกมจริง 3x3 ระหว่างโมเดลกับคู่ตามอัลกอริทึม (ครู Minimax / ตัวเอง / คู่สุ่ม) อัพเดตค่า Q ทุกตา และพล็อตกราฟอัตราชนะแบบเรียลไทม์
                </span>
              </div>
            </div>
          </div>

          {/* Metrics chart */}
          <div className="panel chart-panel" style={{ marginTop: 16 }}>
            <div className="chart-header-row">
              <div>
                <span className="eyebrow">REAL SELF-PLAY METRICS</span>
                <h3 style={{ margin: "4px 0 0" }}>กราฟผลการฝึกจริง (Real-Time Metrics)</h3>
              </div>
              <div className="chart-type-buttons">
                <button className={`chart-type-btn ${chartView === "winrate" ? "active" : ""}`} onClick={() => setChartView("winrate")}>อัตราชนะ (Win Rate %)</button>
                <button className={`chart-type-btn ${chartView === "reward" ? "active" : ""}`} onClick={() => setChartView("reward")}>รางวัลเฉลี่ย (Avg Reward)</button>
                <button className={`chart-type-btn ${chartView === "epsilon" ? "active" : ""}`} onClick={() => setChartView("epsilon")}>Exploration (ε)</button>
              </div>
            </div>

            {renderSparkline()}

            <div className="chart-live-stats">
              <div className="stat-box">
                <span>WIN RATE (ปัจจุบัน)</span>
                <strong>{latestMetric ? `${latestMetric.winRate}%` : "—"}</strong>
              </div>
              <div className="stat-box">
                <span>AVG REWARD / GAME</span>
                <strong>{latestMetric ? latestMetric.avgReward : "—"}</strong>
              </div>
              <div className="stat-box">
                <span>KNOWN STATES</span>
                <strong>{latestMetric ? latestMetric.knownStates : "—"}</strong>
              </div>
              <div className="stat-box">
                <span>EXPLORATION (ε)</span>
                <strong>{latestMetric ? latestMetric.epsilon : "—"}</strong>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ================= PROJECTS ================= */}
      {page === "projects" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">03 / TTT PROJECTS</p>
              <h1>Your Tic Tac Toe Models</h1>
              <p className="lede">จัดการโมเดล O-X ที่คุณฝึกเอง อาจเทรน 3 แบบ และส่งไปดวลในสนามจริง</p>
            </div>
            <button className="primary-action" onClick={() => setShowNewProject(v => !v)}>
              <Plus size={15} /> {showNewProject ? "ปิดฟอร์ม" : "สร้างโปรเจกต์ TTT AI"}
            </button>
          </div>

          {showNewProject && (
            <div className="panel new-project" style={{ marginBottom: 20 }}>
              <div className="panel-heading">
                <div><span className="eyebrow">NEW MODEL</span><h3>สร้างโปรเจกต์ใหม่</h3></div>
                <FolderKanban size={18} />
              </div>
              <label className="field-label">ชื่อโมเดล</label>
              <input
                type="text"
                value={newProjectName}
                onChange={e => setNewProjectName(e.target.value)}
                placeholder="เช่น Nova-v2"
              />
              <label className="field-label">รูปแบบการเทรน</label>
              <select value={newProjectAlgo} onChange={e => setNewProjectAlgo(e.target.value as TTTAlgorithm)}>
                <option value="selfplay-rl">{ALGORITHM_INFO["selfplay-rl"].name} — {ALGORITHM_INFO["selfplay-rl"].tag}</option>
                <option value="minimax-teacher">{ALGORITHM_INFO["minimax-teacher"].name} — {ALGORITHM_INFO["minimax-teacher"].tag}</option>
                <option value="random-drill">{ALGORITHM_INFO["random-drill"].name} — {ALGORITHM_INFO["random-drill"].tag}</option>
              </select>
              <button className="primary-action full-action" onClick={handleCreateProject}>
                <Plus size={14} /> บันทึกและเริ่มโปรเจกต์
              </button>
            </div>
          )}

          <div className="project-list">
            {tttProjects.map(p => (
              <div className="project-card" key={p.id}>
                <div className="project-symbol">
                  <span style={{ fontSize: 24 }}>⭘</span>
                </div>
                <div className="project-main">
                  <div className="project-title">
                    <h3>{p.name}</h3>
                    <span className={`project-status ${p.status.toLowerCase()}`}>
                      {p.status === "Complete" ? "เทรนเสร็จสมบูรณ์" : p.status === "Training" ? "กำลังฝึก" : "พร้อมฝึก"}
                    </span>
                    <span style={{ font: "8px var(--mono)", color: "#888", border: "1px solid #e1e1dc", padding: "2px 6px" }}>
                      {ALGORITHM_INFO[p.algorithm].tag}
                    </span>
                  </div>
                  <p>
                    ผ่านการเล่นจริง <strong>{p.games.toLocaleString()} เกม</strong> · Win Rate <strong>{p.winRate}%</strong> · Loss <strong>{p.lossRate}%</strong> · Draw <strong>{p.drawRate}%</strong> · จำได้ <strong>{knownStatesCount(p.qTable)} สถานะ</strong>
                  </p>
                  <div className="mini-progress">
                    <div style={{ width: `${p.progress}%` }} />
                  </div>
                  <div className="project-meta">
                    <span>PROGRESS: {Math.round(p.progress)}%</span>
                    <span>{p.minutes} MIN TARGET</span>
                  </div>
                </div>

                <div className="project-actions">
                  <button
                    className="primary-action action-button"
                    onClick={() => selectPlayProject(p.id)}
                  >
                    <Swords size={13} /> ประลอง
                  </button>
                  <button
                    className="secondary-action action-button"
                    onClick={() => openTrainingFor(p)}
                  >
                    <BrainCircuit size={13} /> ฝึกต่อ
                  </button>
                  {tttProjects.length > 1 && (
                    <button
                      className="top-icon"
                      title="ลบโปรเจกต์"
                      onClick={() => handleDeleteProject(p.id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {page === "rules" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">RULES & GUIDE</p>
              <h1>วิธีเล่นและกติกา</h1>
              <p className="lede">กฎพื้นฐานของเกม Tic-Tac-Toe (O-X) ในโหมด Arena</p>
            </div>
          </div>
          <div className="panel">
            <div className="panel-heading">
              <div><span className="eyebrow">BASICS</span><h3>กติกาการเล่น</h3></div>
              <ShieldCheck size={18} />
            </div>
            <div className="rules-bullets">
              <p><span>1.</span> เล่นบนตาราง 3x3 คุณเป็น X, AI เป็น O สลับกันเดินจนมีผู้ชนะหรือเสมอ</p>
              <p><span>2.</span> ผู้ที่วางเครื่องหมายเรียงกัน 3 ช่อง (แนวนอน แนวตั้ง หรือเฉียง) ชนะทันที</p>
              <p><span>3.</span> เลือก AI ได้ 4 แบบ — โมเดลที่คุณเทรนเอง, Beginner (สุ่ม), Thinker (ผสม), Perfect (Minimax เต็ม)</p>
              <p><span>4.</span> Perfect AI แพ้ไม่ได้ — เป้าหมายสูงสุดคือทำให้เสมอ</p>
            </div>
          </div>
        </section>
      )}

      {page === "help" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">HELP & STRATEGY</p>
              <h1>ตัวช่วยตัดสินใจ</h1>
              <p className="lede">กลยุทธ์และการวางตำแหน่งหมากใน Tic-Tac-Toe</p>
            </div>
          </div>
          <div className="panel" style={{ marginBottom: 14 }}>
            <div className="panel-heading">
              <div><span className="eyebrow">STRATEGY TIPS</span><h3>แนวทางกลยุทธ์</h3></div>
              <Sparkles size={18} />
            </div>
            <div className="setting-row">
              <span>ยึดช่องกลาง (Center)</span>
              <strong>ช่องที่ 5 สำคัญสุด ให้เส้นทางชนะมากที่สุด</strong>
            </div>
            <div className="setting-row">
              <span>คุมมุมกระดาน (Corners)</span>
              <strong>สร้างโอกาสเข้าทำแบบ Fork (2 ทางเลือก)</strong>
            </div>
            <div className="setting-row">
              <span>ดวลกับ Perfect AI</span>
              <strong>Minimax จะคำนวณทุกความเป็นไปได้ — การเสมอคือผลลัพธ์ที่ดีที่สุด</strong>
            </div>
          </div>
        </section>
      )}
    </>
  );
}

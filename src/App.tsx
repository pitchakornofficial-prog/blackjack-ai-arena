import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft, ArrowRight, BrainCircuit, BookOpen, CircleHelp, Clock3, Code2,
  Cpu, FolderKanban, Gauge, House, Info, Play, Plus, RotateCcw, Settings2,
  ShieldCheck, Sparkles, Swords, Trophy, WandSparkles, X, CheckCircle2,
  Layers, Maximize2, Minimize2, Home, Edit2, Trash2
} from "lucide-react";
import {
  Action, Card as PlayingCard, GameState, applyAction, applyAiAction,
  basicStrategy, explainAction, legalActions, newGame, scoreHand, startRound
} from "./game/engine";
import { QTable, MetricPoint, trainBatch } from "./game/trainer";
import { getAiDecision } from "./game/aiAgent";
import { detectGpuHardware, dispatchGpuBatchEvaluation } from "./game/gpuAccelerator";
import ProjectCreateModal, { TRAINING_DURATION_OPTIONS } from "./game/ProjectCreateModal";
import ProjectEditModal from "./game/ProjectEditModal";
import ChessView from "./game/ChessView";
import TTTView from "./game/TTTView";
import { loadStoredData, saveStoredData, isProjectRecordArray, STORAGE_KEYS } from "./game/persistence";

type ActiveGame = "blackjack" | "chess" | "ttt" | "undecided";
type GameCategory = "card" | "board";

type Page = "home" | "play" | "training" | "projects" | "rules" | "help";
type Opponent = "baseline" | "thinking" | "project";
type TrainingDevice = "cpu" | "gpu";
type TrainingAlgorithm = "qlearning" | "baseline";

type TrainingProject = {
  id: number;
  name: string;
  minutes: number;
  status: "Ready" | "Training" | "Complete";
  progress: number;
  games: number;
  created: string;
  trainingDevice: TrainingDevice;
  trainingAlgorithm: TrainingAlgorithm;
  elapsedSeconds: number;
  metricHistory: MetricPoint[];
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

function GameSelection({
  category,
  onSelectCategory,
  activeGame,
  onPickGame
}: {
  category: GameCategory | null;
  onSelectCategory: (category: GameCategory | null) => void;
  activeGame: ActiveGame;
  onPickGame: (game: ActiveGame) => void;
}) {
  if (!category) {
    return (
      <div className="game-category-grid">
        <button className="game-category-card" onClick={() => onSelectCategory("card")}>
          <span className="game-category-icon">♠</span>
          <strong>เกมการ์ด</strong>
          <p>เกมไพ่และการวางกลยุทธ์</p>
          <span className="game-category-count">1 เกม</span>
        </button>
        <button className="game-category-card" onClick={() => onSelectCategory("board")}>
          <span className="game-category-icon">♟</span>
          <strong>เกมกระดาน</strong>
          <p>เกมวางแผนและดวลบนกระดาน</p>
          <span className="game-category-count">2 เกม</span>
        </button>
      </div>
    );
  }

  const games = category === "card"
    ? [{
        id: "blackjack" as const,
        icon: "♠",
        name: "Blackjack AI Arena",
        description: "สนามไพ่ 21 แต้ม ดวลสลับตากับ AI พร้อมเทรน Q-Learning",
        tag: "ไพ่ · สลับเทิร์น · Q-Learning"
      }]
    : [{
        id: "chess" as const,
        icon: "♞",
        name: "Chess AI Arena (หมากรุก)",
        description: "กระดานหมากรุกสากล 8x8 ดวลสลับตากับ Minimax Chess AI",
        tag: "หมากรุก · สลับเทิร์น · Minimax"
      }, {
        id: "ttt" as const,
        icon: "⭘",
        name: "Tic Tac Toe AI Arena",
        description: "เกม 3 ช่องชนะ ดวลกับ AI ตั้งแต่อ่อนแรงถึง Minimax ที่แพ้ไม่ได้",
        tag: "O/X · สลับเทิร์น · Minimax"
      }];

  return (
    <div className="game-selection">
      <div className="game-selection-heading">
        <button className="game-category-back" onClick={() => onSelectCategory(null)}>
          <ArrowLeft size={13} /> หมวดหมู่เกม
        </button>
        <span>{category === "card" ? "เกมการ์ด" : "เกมกระดาน"}</span>
      </div>
      <div className="game-choice-grid game-selection-games">
        {games.map(game => (
          <button
            key={game.id}
            className={`game-choice-card ${activeGame === game.id ? "dark-card" : ""}`}
            onClick={() => onPickGame(game.id)}
          >
            <div className="game-choice-icon">{game.icon}</div>
            <strong>{game.name}</strong>
            <p>{game.description}</p>
            <span className="game-choice-tag">{game.tag}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// Restore saved projects (with trained Q-Tables) from localStorage; keep the
// in-memory fallback for fresh browsers so the workspace is never empty.
const savedBlackjackProjects = loadStoredData<TrainingProject[]>(
  STORAGE_KEYS.blackjackProjects,
  isProjectRecordArray
);

const initialProjects: TrainingProject[] = [
  {
    id: 1,
    name: "Baseline Challenger",
    minutes: 10,
    status: "Ready",
    progress: 0,
    games: 0,
    created: "ตัวอย่างโปรเจกต์",
    trainingDevice: "cpu",
    trainingAlgorithm: "qlearning",
    elapsedSeconds: 0,
    metricHistory: []
  }
];

export default function App() {
  const [activeGame, setActiveGame] = useState<ActiveGame>("undecided");
  const [gamePicked, setGamePicked] = useState(false);
  const showInitialGamePicker = activeGame === "undecided" && !gamePicked;
  const [showGameModal, setShowGameModal] = useState(false);
  const [selectedGameCategory, setSelectedGameCategory] = useState<GameCategory | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [page, setPage] = useState<Page>("home");
  const [game, setGame] = useState<GameState>(newGame(Math.random, "play", "duel"));
  const [bet, setBet] = useState(100);
  const [opponent, setOpponent] = useState<Opponent>("baseline");
  const [showHelp, setShowHelp] = useState(false);

  // Projects restored from localStorage; anything that was mid-training when
  // the page was closed is reset to "Ready" (progress/games/Q-Table kept).
  const [projects, setProjects] = useState<TrainingProject[]>(() =>
    (savedBlackjackProjects ?? initialProjects).map(p => ({
      ...p,
      status: p.status === "Training" ? ("Ready" as const) : p.status,
      trainingDevice: p.trainingDevice === "gpu" ? "gpu" as const : "cpu" as const,
      trainingAlgorithm: p.trainingAlgorithm === "baseline" ? p.trainingAlgorithm : "qlearning",
      elapsedSeconds: typeof p.elapsedSeconds === "number" ? p.elapsedSeconds : 0,
      metricHistory: Array.isArray(p.metricHistory) ? p.metricHistory : [],
      qTable: p.qTable && typeof p.qTable === "object" ? p.qTable : undefined
    }))
  );
  const projectsRef = useRef(projects);
  projectsRef.current = projects;
  function updateProjects(update: (current: TrainingProject[]) => TrainingProject[]) {
    const updated = update(projectsRef.current);
    projectsRef.current = updated;
    setProjects(updated);
  }
  const [projectName, setProjectName] = useState("My Blackjack Agent");
  const [trainMinutes, setTrainMinutes] = useState(() => projects[0]?.minutes ?? 1);
  const [activeTrainingId, setActiveTrainingId] = useState<number | null>(() => projects[0]?.id ?? null);
  const [selectedPlayProjectId, setSelectedPlayProjectId] = useState<number | null>(1);
  const [trainingSeconds, setTrainingSeconds] = useState(() => projects[0]?.elapsedSeconds ?? 0);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [trainingBaseSeconds, setTrainingBaseSeconds] = useState(() => projects[0]?.elapsedSeconds ?? 0);

  // Training parameters
  const [trainingOpponent, setTrainingOpponent] = useState("baseline");
  const [trainingAlgorithm, setTrainingAlgorithm] = useState<TrainingAlgorithm>("qlearning");
  const [trainingDevice, setTrainingDevice] = useState<TrainingDevice>("cpu");
  const [gpuSupported] = useState(() => detectGpuHardware().supported);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [episodesPerBatch, setEpisodesPerBatch] = useState(5000);
  const [learningRate, setLearningRate] = useState(0.1);
  const [discountFactor, setDiscountFactor] = useState(0.95);
  const [explorationRate, setExplorationRate] = useState(0.2);
  const [showNewProject, setShowNewProject] = useState(false);
  const [editingProject, setEditingProject] = useState<TrainingProject | null>(null);
  const [helpAction, setHelpAction] = useState<Action | null>(null);

  // Real Metrics Chart State
  const [metricHistory, setMetricHistory] = useState<MetricPoint[]>(() => projects[0]?.metricHistory ?? []);
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

  // Periodically save all game state and flush the latest snapshot when the page hides.
  useEffect(() => {
    const persist = () => saveStoredData(STORAGE_KEYS.blackjackProjects, projectsRef.current);
    persist();
    const saveTimer = window.setInterval(persist, 2000);
    window.addEventListener("pagehide", persist);
    document.addEventListener("visibilitychange", persist);
    return () => {
      window.clearInterval(saveTimer);
      window.removeEventListener("pagehide", persist);
      document.removeEventListener("visibilitychange", persist);
      persist();
    };
  }, []);

  // Update the selected project's training duration without changing its method.
  function handleSelectDuration(minutes: number) {
    setTrainMinutes(minutes);
    if (activeTrainingId !== null) {
      updateProjects(prev => prev.map(p => p.id === activeTrainingId ? { ...p, minutes } : p));
    }
  }

  // Sync project dropdown
  function handlePickGame(game: ActiveGame) {
    setActiveGame(game);
    setGamePicked(true);
    setShowGameModal(false);
    setSelectedGameCategory(null);
    // Start a fresh session in the chosen game
    setGame(newGame(Math.random, "play", "duel"));
    setPage("home");
  }

  function handleSelectProject(id: number | null) {
    if (trainingRunning) pauseTraining();
    setActiveTrainingId(id);
    if (id !== null) {
      const p = projectsRef.current.find(x => x.id === id);
      if (p) {
        setTrainMinutes(p.minutes);
        setTrainingDevice(p.trainingDevice);
        setTrainingAlgorithm(p.trainingAlgorithm);
        setTrainingSeconds(p.elapsedSeconds);
        setTrainingBaseSeconds(p.elapsedSeconds);
        setMetricHistory(p.metricHistory);
      }
    } else {
      setTrainingSeconds(0);
      setTrainingBaseSeconds(0);
      setMetricHistory([]);
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!trainingRunning || !activeTrainingId || trainingStartedAt === null) return;

    const timer = window.setInterval(() => {
      const elapsed = trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000);
      const limit = totalTrainingSeconds;
      const next = Math.min(elapsed, limit);
      setTrainingSeconds(next);

      // Run real simulation batch on each tick!
      const batchSize = Math.max(50, Math.min(500, Math.round(episodesPerBatch / 20)));
      const currentQTable: QTable = activeProject?.qTable
        ? Object.fromEntries(Object.entries(activeProject.qTable).map(([key, values]) => [key, { ...values }]))
        : {};
      const previousMetrics = activeProject?.metricHistory ?? metricHistory;
      const currentEpsilon = previousMetrics.length > 0
        ? previousMetrics[previousMetrics.length - 1].epsilon
        : explorationRate;
      if (activeProject?.trainingDevice === "gpu") {
        try {
          if (dispatchGpuBatchEvaluation(batchSize, learningRate) === 0) {
            throw new Error("GPU acceleration could not initialize a WebGL compute context.");
          }
        } catch (error) {
          console.error("[blackjack-training] GPU batch failed:", error);
          setTrainingError("เริ่มคำนวณด้วย GPU ไม่สำเร็จ การฝึกหยุดไว้และบันทึกความคืบหน้าแล้ว");
          setTrainingRunning(false);
          setTrainingStartedAt(null);
          setTrainingBaseSeconds(next);
          updateProjects(old => old.map(p => p.id === activeTrainingId
            ? { ...p, status: "Ready", elapsedSeconds: next, progress: Math.min(100, next / Math.max(1, limit) * 100) }
            : p));
          return;
        }
      }
      const { result, updatedEpsilon } = trainBatch(
        currentQTable,
        batchSize,
        learningRate,
        discountFactor,
        currentEpsilon,
        activeProject?.trainingAlgorithm ?? "qlearning"
      );

      // Save real metrics point
      const point: MetricPoint = {
        step: previousMetrics.length + 1,
        timeSeconds: next,
        totalEpisodes: (previousMetrics[previousMetrics.length - 1]?.totalEpisodes ?? 0) + batchSize,
        winRate: Number(result.winRate.toFixed(1)),
        lossRate: Number(result.lossRate.toFixed(1)),
        pushRate: Number(result.pushRate.toFixed(1)),
        avgReward: Number(result.avgReward.toFixed(3)),
        epsilon: Number(updatedEpsilon.toFixed(3))
      };
      const updatedMetrics = [...previousMetrics, point];
      setMetricHistory(updatedMetrics);

      // Update project state with actual played hands and trained Q-Table
      updateProjects(old => old.map(p => {
        if (p.id !== activeTrainingId) return p;
        const newGames = p.games + batchSize;
        return {
          ...p,
          games: newGames,
          qTable: currentQTable,
          elapsedSeconds: next,
          metricHistory: updatedMetrics,
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
    explorationRate, metricHistory, activeProject?.qTable,
    activeProject?.trainingDevice, activeProject?.trainingAlgorithm,
    activeProject?.metricHistory
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
    if (trainingRunning) pauseTraining();
    const id = Math.max(0, ...projects.map(p => p.id)) + 1;
    const newProj: TrainingProject = {
      id,
      name: projectName.trim() || `Agent Project ${id}`,
      minutes: trainMinutes,
      status: "Ready",
      progress: 0,
      games: 0,
      created: new Date().toLocaleDateString("th-TH"),
      trainingDevice,
      trainingAlgorithm,
      elapsedSeconds: 0,
      metricHistory: []
    };
    updateProjects(current => [...current, newProj]);
    setActiveTrainingId(id);
    setTrainingSeconds(0);
    setTrainingBaseSeconds(0);
    setMetricHistory([]);
    setShowNewProject(false);
    setPage("projects");
  }

  function saveProjectEdits(name: string, minutes: number) {
    if (!editingProject) return;
    updateProjects(current => current.map(project => project.id === editingProject.id
      ? { ...project, name, minutes }
      : project));
    if (activeTrainingId === editingProject.id) setTrainMinutes(minutes);
    setEditingProject(null);
  }

  function deleteProject(projectId: number) {
    const project = projectsRef.current.find(item => item.id === projectId);
    if (!project || !window.confirm(`ต้องการลบโปรเจกต์ "${project.name}" และข้อมูลการฝึกทั้งหมดหรือไม่?`)) return;
    if (trainingRunning && activeTrainingId === projectId) pauseTraining();
    const remaining = projectsRef.current.filter(item => item.id !== projectId);
    updateProjects(() => remaining);
    if (activeTrainingId === projectId) {
      const next = remaining[0];
      setActiveTrainingId(next?.id ?? null);
      setTrainMinutes(next?.minutes ?? 1);
      setTrainingSeconds(next?.elapsedSeconds ?? 0);
      setTrainingBaseSeconds(next?.elapsedSeconds ?? 0);
      setMetricHistory(next?.metricHistory ?? []);
    }
    if (selectedPlayProjectId === projectId) setSelectedPlayProjectId(remaining[0]?.id ?? null);
  }

  function openNewProject() {
    setProjectName("My Blackjack Agent");
    setTrainMinutes(1);
    setTrainingDevice("cpu");
    setTrainingAlgorithm("qlearning");
    setShowNewProject(true);
  }

  function beginTraining(project: TrainingProject) {
    const limit = trainMinutes * 60;
    const elapsed = project.elapsedSeconds >= limit ? 0 : project.elapsedSeconds;
    const history = project.metricHistory;
    updateProjects(old => old.map(p => {
      if (p.id !== project.id) return p;
      return {
        ...p,
        minutes: trainMinutes,
        status: "Training",
        progress: Math.min(100, (elapsed / Math.max(1, trainMinutes * 60)) * 100),
        games: p.games,
        elapsedSeconds: elapsed,
        metricHistory: history,
        qTable: p.qTable ? { ...p.qTable } : {}
      };
    }));
    setActiveTrainingId(project.id);
    setTrainingSeconds(elapsed);
    setTrainingBaseSeconds(elapsed);
    setMetricHistory(history);
    setTrainingError(null);
    setTrainingStartedAt(Date.now());
    setTrainingRunning(true);
    setPage("training");
  }

  function pauseTraining() {
    if (trainingStartedAt !== null) {
      const elapsed = Math.min(totalTrainingSeconds, trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000));
      setTrainingSeconds(elapsed);
      setTrainingBaseSeconds(elapsed);
      if (activeTrainingId !== null) {
        updateProjects(old => old.map(p => p.id === activeTrainingId
          ? { ...p, status: "Ready", elapsedSeconds: elapsed, progress: Math.min(100, elapsed / Math.max(1, totalTrainingSeconds) * 100) }
          : p));
      }
    }
    setTrainingStartedAt(null);
    setTrainingRunning(false);
  }

  function resumeTraining() {
    if (!activeTrainingId) return;
    const project = projectsRef.current.find(item => item.id === activeTrainingId);
    if (project && project.elapsedSeconds >= totalTrainingSeconds) {
      beginTraining(project);
      return;
    }
    setTrainingStartedAt(Date.now());
    setTrainingRunning(true);
  }

  function resetTraining() {
    pauseTraining();
    setTrainingSeconds(0);
    setTrainingBaseSeconds(0);
    setTrainingStartedAt(null);
    setMetricHistory([]);
    setTrainingError(null);
    if (activeTrainingId) {
      updateProjects(old => old.map(p => p.id === activeTrainingId
        ? { ...p, status: "Ready", progress: 0, games: 0, elapsedSeconds: 0, metricHistory: [], qTable: {} }
        : p));
    }
  }

  function fmt(seconds: number) {
    return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  }

  function handleShowGameModal() {
    setSelectedGameCategory(null);
    setShowGameModal(true);
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
    <div className={`app-layout ${isFullscreen ? "fullscreen-mode" : ""}${showInitialGamePicker ? " show-initial-game-picker" : ""}`}>
      <aside className="sidebar">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 4px 22px" }}>
          <button className="brand" style={{ padding: 0 }} onClick={() => setPage("home")}>
            <span className="brand-icon">{activeGame === "blackjack" ? "♠" : activeGame === "ttt" ? "⭘" : "♞"}</span>
            <span>
              {activeGame === "blackjack" ? "BLACKJACK" : activeGame === "ttt" ? "TIC TAC TOE" : "CHESS"}
              <small>AI ARENA HUB</small>
            </span>
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
        ) : activeGame === "ttt" ? (
          <>
            {navItem("home", "ภาพรวม", House)}
            {navItem("play", "โต๊ะดวล O-X", Swords)}
            {navItem("training", "TTT AI Training", BrainCircuit)}
            {navItem("projects", "โปรเจกต์ TTT AI", FolderKanban)}
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
          <div className="avatar">{activeGame === "blackjack" ? "BJ" : activeGame === "ttt" ? "TT" : "CH"}</div>
          <div><strong>Arena Hub</strong><small>{activeGame === "blackjack" ? "Blackjack Mode" : activeGame === "ttt" ? "Tic Tac Toe Mode" : "Chess Mode"}</small></div>
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
              className="game-switch-btn"
              onClick={handleShowGameModal}
              title="สลับเกม (Game Hub)"
            >
              <Layers size={13} />
              <span>สลับเกม</span>
            </button>
            <button
              className={`fullscreen-btn ${isFullscreen ? "active" : ""}`}
              onClick={toggleFullscreen}
              title={isFullscreen ? "ออกจากโหมดเต็มจอ (ESC)" : "เข้าสู่โหมดเต็มจอ"}
            >
              {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              <span>{isFullscreen ? "ย่อหน้าจอ" : "เต็มจอ"}</span>
            </button>
            <span className="status-dot" /> LOCAL SESSION
          </div>
        </header>

        {/* If no game chosen yet, require selection before entering either mode */}
        {showInitialGamePicker ? (
          <div className="game-picker-overlay">
            <div className="game-picker-card">
              <div className="modal-title">เลือกหมวดหมู่เกมที่อยากเล่น</div>
              <p className="modal-lede">เลือกหมวดหมู่ แล้วเลือกเกมที่ต้องการเล่น</p>
              <GameSelection
                category={selectedGameCategory}
                onSelectCategory={setSelectedGameCategory}
                activeGame={activeGame}
                onPickGame={handlePickGame}
              />
            </div>
          </div>
        ) : activeGame === "chess" ? (
          <ChessView
            page={page}
            setPage={setPage}
            isFullscreen={isFullscreen}
            onToggleFullscreen={toggleFullscreen}
          />
        ) : activeGame === "ttt" ? (
          <TTTView
            page={page}
            setPage={setPage}
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
                <p>ฝึก Blackjack AI ด้วย CPU หรือ GPU ในเครื่อง พร้อมบันทึกความคืบหน้าและกราฟอัตโนมัติ</p>
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
                  {projects.length === 0
                    ? <option value="">ยังไม่มีโปรเจกต์ — สร้างในหน้า AI Training</option>
                    : projects.map(p => (
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
                <p className="lede">ฝึกด้วย CPU หรือ GPU ในเครื่อง วิธีที่เลือกจะผูกกับโปรเจกต์และเปลี่ยนไม่ได้หลังสร้าง</p>
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

                <label className="field-label">วิธีเทรนที่ผูกกับโปรเจกต์ (เปลี่ยนไม่ได้)</label>
                <div className="inline-note">
                  {activeProject?.trainingDevice === "gpu" ? <Gauge size={14} /> : <Cpu size={14} />}
                  {activeProject?.trainingDevice === "gpu" ? "GPU batch + CPU Q-table updates" : "CPU Q-Learning"}
                  {activeProject?.trainingDevice === "gpu" ? " · การอัปเดต Q-table ยังคำนวณบน CPU" : ""}
                  {" · "}
                  {activeProject?.trainingAlgorithm === "baseline" ? "Baseline" : "Q-Learning"}
                </div>

                <label className="field-label">เวลาฝึก (ปรับได้)</label>
                <div className="duration-options">
                  {TRAINING_DURATION_OPTIONS.map(minutes => (
                    <button
                      key={minutes}
                      className={trainMinutes === minutes ? "selected" : ""}
                      onClick={() => handleSelectDuration(minutes)}
                    >
                      {minutes < 60 ? `${minutes} น.` : "1 ชม."}
                    </button>
                  ))}
                </div>
                {trainingError && <div className="inline-note" role="alert">{trainingError}</div>}

                <div className="advanced-settings">
                  <div className="range-setting">
                    <label><span>LOADED SAMPLES / BATCH</span><strong style={{ fontFamily: "var(--mono)", color: "#222" }}>{episodesPerBatch.toLocaleString()}</strong></label>
                    <input type="range" min={1000} max={20000} step={1000} value={episodesPerBatch} onChange={e => setEpisodesPerBatch(Number(e.target.value))} />
                  </div>
                  <div className="range-setting">
                    <label><span>LEARNING RATE</span><strong style={{ fontFamily: "var(--mono)", color: "#222" }}>{learningRate.toFixed(2)}</strong></label>
                    <input type="range" min={0.01} max={1} step={0.01} value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />
                  </div>
                  <div className="range-setting">
                    <label><span>DISCOUNT FACTOR (γ)</span><strong style={{ fontFamily: "var(--mono)", color: "#222" }}>{discountFactor.toFixed(2)}</strong></label>
                    <input type="range" min={0.5} max={1} step={0.01} value={discountFactor} onChange={e => setDiscountFactor(Number(e.target.value))} />
                  </div>
                  <div className="range-setting">
                    <label><span>EXPLORATION RATE (ε)</span><strong style={{ fontFamily: "var(--mono)", color: "#222" }}>{explorationRate.toFixed(2)}</strong></label>
                    <input type="range" min={0} max={1} step={0.01} value={explorationRate} onChange={e => setExplorationRate(Number(e.target.value))} />
                  </div>
                </div>
                <div className="field-help" style={{ marginTop: 14 }}>
                  <Info size={14} />
                  เมื่อเริ่มฝึก ระบบจะจำลองมือ Blackjack ด้วย Q-Learning แบบ Real-time พร้อมอัปเดตกราฟสด
                </div>

                <button className="outline-full full-action project-create-trigger" onClick={openNewProject}>
                  <Plus size={14} /> สร้างโปรเจกต์ใหม่
                </button>

              </div>

              {/* Run / Visual Panel */}
              <div className="panel run-panel">
                <div className="panel-heading">
                  <div><span className="eyebrow">STATUS</span><h3>สถานะการฝึก</h3></div>
                  <Gauge size={18} />
                </div>

                {!activeTrainingId ? (
                  <div className="notice">
                    <BrainCircuit size={18} />
                    <div><strong>ยังไม่เลือกโปรเจกต์</strong><small>สร้างโปรเจกต์หรือเลือกจากรายการด้านบนเพื่อเริ่มฝึก</small></div>
                  </div>
                ) : !projects.find(p => p.id === activeTrainingId) ? (
                  <div className="notice">
                    <BrainCircuit size={18} />
                    <div><strong>โปรเจกต์ไม่พบ</strong><small>โปรเจกต์ที่เลือกไม่มีอยู่ในระบบ (อาจถูกลบไปแล้ว)</small></div>
                  </div>
                ) : activeProject?.status === "Complete" ? (
                  <div>
                    <div className="timer" style={{ textAlign: "center", margin: "35px 0 18px" }}>
                      {fmt(trainingSeconds)}
                      <small>สรุปการฝึกเสร็จสิ้น</small>
                    </div>
                    <div className="metric-grid" style={{ marginTop: 10 }}>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>WIN RATE</span><strong>{(latestMetric?.winRate ?? 50).toFixed(1)}%</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>TOTAL HANDS</span><strong>{(activeProject?.games ?? 0).toLocaleString()}</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>GAMES COMPLETED</span><strong>{(activeProject?.games ?? 0).toLocaleString()}</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>MODELS</span><strong>1</strong></div>
                    </div>
                    <button className="primary-action full-action" onClick={() => beginTraining(activeProject!)}>
                      <Play size={14} /> ฝึกต่อ
                    </button>
                    <p style={{ font: "9px var(--mono)", color: "#777", marginTop: 14, textAlign: "center" }}>โมเดลและข้อมูลเดิมจะถูกเก็บไว้เมื่อฝึกต่อ</p>
                  </div>
                ) : trainingRunning ? (
                  <div>
                    <div className="timer" style={{ textAlign: "center" }}>
                      {fmt(trainingSeconds)}
                      <small>{totalTrainingSeconds}s</small>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${Math.min(100, (trainingSeconds / totalTrainingSeconds) * 100)}%` }} />
                    </div>
                    <div className="progress-meta">
                      <span>ระยะเวลาฝึก</span>
                      <span>{trainingSeconds}s / {totalTrainingSeconds}s</span>
                      <span>ความคืบหน้า</span>
                      <span>{Math.round((trainingSeconds / totalTrainingSeconds) * 100)}%</span>
                    </div>

                    <div className="metric-grid" style={{ marginTop: 31 }}>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>WIN RATE</span><strong>{(latestMetric?.winRate ?? 50).toFixed(1)}%</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>LOSS RATE</span><strong>{(latestMetric?.lossRate ?? 50).toFixed(1)}%</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>PUSH RATE</span><strong>{(latestMetric?.pushRate ?? 0).toFixed(1)}%</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>AVG REWARD</span><strong>{(latestMetric?.avgReward ?? 0).toFixed(3)}</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>STATE COUNT</span><strong>{(activeProject?.qTable ? Object.keys(activeProject.qTable).length : 0).toLocaleString()}</strong></div>
                      <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>EPISODES</span><strong>{metricHistory.reduce((sum, metric) => sum + metric.totalEpisodes, 0).toLocaleString()}</strong></div>
                    </div>

                    <div style={{ marginTop: 20, display: "flex", gap: 12, flexWrap: "wrap" }}>
                      <button className="outline-full" style={{ flex: 1, minWidth: 140 }} onClick={pauseTraining}>
                        <Clock3 size={14} /> หยุดชั่วคราว
                      </button>
                      <button className="ghost-button" style={{ flex: 1, minWidth: 140 }} onClick={resetTraining}>
                        <RotateCcw size={14} /> รีเซ็ต
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div style={{ textAlign: "center", padding: "20px 0" }}>
                      <div className="timer" style={{ margin: "25px 0 15px" }}>
                        {fmt(trainingSeconds)}
                        <small>{totalTrainingSeconds}s</small>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${(trainingSeconds / totalTrainingSeconds) * 100}%` }} />
                      </div>
                      <div className="progress-meta">
                        <span>ระยะเวลาฝึก</span>
                        <span>{trainingSeconds}s / {totalTrainingSeconds}s</span>
                        <span>ความคืบหน้า</span>
                        <span>{Math.round((trainingSeconds / totalTrainingSeconds) * 100)}%</span>
                      </div>

                      <div className="metric-grid" style={{ marginTop: 31 }}>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>WIN RATE</span><strong>{(latestMetric?.winRate ?? 50).toFixed(1)}%</strong></div>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>LOSS RATE</span><strong>{(latestMetric?.lossRate ?? 50).toFixed(1)}%</strong></div>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>PUSH RATE</span><strong>{(latestMetric?.pushRate ?? 0).toFixed(1)}%</strong></div>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>AVG REWARD</span><strong>{(latestMetric?.avgReward ?? 0).toFixed(3)}</strong></div>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>STATE COUNT</span><strong>{(activeProject?.qTable ? Object.keys(activeProject.qTable).length : 0).toLocaleString()}</strong></div>
                        <div><span style={{ font: "8px var(--mono)", color: "#999", letterSpacing: ".05em" }}>EPISODES</span><strong>{(activeProject?.games ?? 0).toLocaleString()}</strong></div>
                      </div>

                      <div style={{ marginTop: 24, display: "flex", gap: 10, flexWrap: "wrap" }}>
                        <button className="primary-action" style={{ flex: 2, minWidth: 160 }} onClick={() => beginTraining(activeProject!)}>
                          <Play size={14} /> เริ่มฝึก
                        </button>
                        <button className="ghost-button" style={{ flex: 1 }} onClick={() => setActiveTrainingId(null)}>
                          <X size={14} /> ยกเลิก
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Real-time chart area */}
                {metricHistory.length > 0 && (
                  <div style={{ marginTop: 28 }}>
                    <div className="chart-header-row">
                      <span style={{ font: "9px var(--mono)", color: "#888", letterSpacing: ".1em" }}>LOOKBACK CHART</span>
                      <div className="chart-type-buttons">
                        <button className={`chart-type-btn ${chartView === "winrate" ? "active" : ""}`} onClick={() => setChartView("winrate")}>Win Rate</button>
                        <button className={`chart-type-btn ${chartView === "reward" ? "active" : ""}`} onClick={() => setChartView("reward")}>Reward</button>
                        <button className={`chart-type-btn ${chartView === "epsilon" ? "active" : ""}`} onClick={() => setChartView("epsilon")}>ε-Greedy</button>
                      </div>
                    </div>
                    {renderRealChart()}

                    <div className="chart-live-stats" style={{ marginTop: 16 }}>
                      <div className="stat-box">
                        <span>WIN RATE LAST</span>
                        <strong>{(latestMetric?.winRate ?? 0).toFixed(1)}%</strong>
                      </div>
                      <div className="stat-box">
                        <span>LOSS RATE LAST</span>
                        <strong>{(latestMetric?.lossRate ?? 0).toFixed(1)}%</strong>
                      </div>
                      <div className="stat-box">
                        <span>PUSH RATE LAST</span>
                        <strong>{(latestMetric?.pushRate ?? 0).toFixed(1)}%</strong>
                      </div>
                      <div className="stat-box">
                        <span>AVG REWARD LAST</span>
                        <strong>{(latestMetric?.avgReward ?? 0).toFixed(3)}</strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* ================= PAGE: PROJECTS ================= */}
        {page === "projects" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">03 / BLACKJACK PROJECTS</p>
                <h1>โปรเจกต์ AI ของฉัน</h1>
                <p className="lede">ข้อมูลโปรเจกต์และผลฝึกถูกจัดเก็บแยกในหมวด Blackjack บนเครื่องโดยอัตโนมัติ</p>
              </div>
              <button className="primary-action" onClick={openNewProject}>
                <Plus size={15} /> สร้างโปรเจกต์ใหม่
              </button>
            </div>

            {projects.length === 0 ? (
              <div className="notice">
                <FolderKanban size={18} />
                <div><strong>ยังไม่มีโปรเจกต์</strong><small>สร้างโปรเจกต์ใหม่เพื่อเริ่มเทรน Agent</small></div>
              </div>
            ) : (
              <div className="project-list">
                {projects.map(p => (
                <div className="project-card" key={p.id}>
                    <div className="project-symbol">
                      {p.trainingDevice === "gpu" ? <Gauge size={22} /> : <Cpu size={22} />}
                    </div>
                    <div className="project-main">
                      <div className="project-title">
                        <h3>{p.name}</h3>
                        <span className={`project-status ${p.status.toLowerCase()}`}>{p.status}</span>
                      </div>
                      <p>{p.created}</p>
                      <div className="mini-progress">
                        <div style={{ width: `${p.progress}%` }} />
                      </div>
                      <div className="project-meta">
                        <span>{p.minutes} นาที</span>
                        <span>{p.trainingDevice.toUpperCase()} · {p.trainingAlgorithm.toUpperCase()}</span>
                        <span>{p.games.toLocaleString()} มือ</span>
                        <span>{Math.floor(p.elapsedSeconds / 60)}:{String(p.elapsedSeconds % 60).padStart(2, "0")} · {p.progress.toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="project-actions">
                      <button
                        className="outline-full"
                        onClick={() => {
                          handleSelectProject(p.id);
                          setPage("training");
                        }}
                      >
                        <Play size={13} /> ฝึก
                      </button>
                      <button className="outline-full" onClick={() => setEditingProject(p)}>
                        <Edit2 size={13} /> แก้ไข
                      </button>
                      <button className="outline-full" onClick={() => deleteProject(p.id)}>
                        <Trash2 size={13} /> ลบ
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {page === "rules" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">RULES & GUIDE</p>
                <h1>วิธีเล่นและกติกา</h1>
                <p className="lede">กฎพื้นฐานของ Blackjack สำหรับโหมด Arena Duel</p>
              </div>
            </div>
            <div className="panel" style={{ marginBottom: 18 }}>
              <div className="panel-heading">
                <div><span className="eyebrow">BASICS</span><h3>กติกาพื้นฐาน</h3></div>
                <BookOpen size={18} />
              </div>
              <div className="rules-bullets">
                <p><span>1.</span> เป้าหมายคือให้ผลรวมไพ่เข้าใกล้ 21 มากที่สุด โดยไม่เกิน 21 (Bust)</p>
                <p><span>2.</span> รับไพ่ 2 ใบแรก ทั้งหงาย (เจ้ามือมีหนึ่งใบคว่ำ)</p>
                <p><span>3.</span> ผู้เล่นเลือก Hit (จั่ว), Stand (หยุด), Double (เพิ่มเดิมพัน 2 เท่า + จั่ว 1 ใบ), Split (แยกไพ่เมื่อไพ่ 2 ใบเท่ากัน)</p>
                <p><span>4.</span> เจ้ามือจะจั่วจนถึง 17 ขึ้นไป (S17)</p>
              </div>
            </div>
            <div className="panel">
              <div className="panel-heading">
                <div><span className="eyebrow">PAYOUTS</span><h3>อัตราจ่าย</h3></div>
                <Trophy size={18} />
              </div>
              <div className="setting-row"><span>Blackjack (A + 10)</span><strong>3:2</strong></div>
              <div className="setting-row"><span>ชนะปกติ</span><strong>1:1</strong></div>
              <div className="setting-row"><span>เจ้ามือ Bust</span><strong>1:1</strong></div>
              <div className="setting-row"><span>เสมอ (Push)</span><strong>คืนเดิมพัน</strong></div>
            </div>
          </section>
        )}

        {page === "help" && (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">HELP & STRATEGY</p>
                <h1>ตัวช่วยตัดสินใจ</h1>
                <p className="lede">คำแนะนำจาก Basic Strategy เพื่อใช้เป็นแนวทางในการเล่น Blackjack</p>
              </div>
            </div>
            <div className="panel" style={{ marginBottom: 14 }}>
              <div className="panel-heading">
                <div><span className="eyebrow">BASIC STRATEGY</span><h3>ตารางพื้นฐาน</h3></div>
                <Sparkles size={18} />
              </div>
              <p className="panel-copy">ช่วยแนะนำการเล่นตามความน่าจะเป็น (Basic Strategy) ซึ่งสามารถลดโอกาสเสียเปรียบให้เจ้ามือได้ ใช้เป็นแนวทางในการตัดสินใจทุกตาเดิน</p>
              <div className="setting-row"><span>ไพ่ผู้เล่น Hard 12–16 vs เจ้ามือ 2–6</span><strong>Stand (หยุด)</strong></div>
              <div className="setting-row"><span>ไพ่ผู้เล่น Hard 12–16 vs เจ้ามือ 7–A</span><strong>Hit (จั่ว)</strong></div>
              <div className="setting-row"><span>ไพ่ผู้เล่น 11</span><strong>Double (เดิมพันเพิ่ม)</strong></div>
              <div className="setting-row"><span>ไพ่ผู้เล่น A-A หรือ 8-8</span><strong>Split (แยก)</strong></div>
            </div>
            <div className="panel">
              <div className="panel-heading">
                <div><span className="eyebrow">QUICK START</span><h3>เริ่มต้นใช้งาน</h3></div>
                <BookOpen size={18} />
              </div>
              <button className="guide-row" onClick={() => setPage("play")}>
                <span className="guide-icon"><Swords size={17} /></span>
                <span><strong>ลงโต๊ะดวล AI</strong><small>ไปหน้าเล่นเกม Arena Duel</small></span>
                <ArrowRight size={16} />
              </button>
              <button className="guide-row" onClick={() => setPage("rules")}>
                <span className="guide-icon"><BookOpen size={17} /></span>
                <span><strong>กติกาพื้นฐาน</strong><small>ดูกฎกติกา Blackjack ทั้งหมด</small></span>
                <ArrowRight size={16} />
              </button>
            </div>
          </section>
        )}
        {showNewProject && (page === "training" || page === "projects") && (
          <ProjectCreateModal
            gameName="Blackjack"
            onClose={() => setShowNewProject(false)}
            onCreate={createProject}
          >
            <label className="field-label" htmlFor="blackjack-project-name">ชื่อโปรเจกต์</label>
            <input
              id="blackjack-project-name"
              type="text"
              value={projectName}
              onChange={event => setProjectName(event.target.value)}
              placeholder="ตัวอย่าง: Deep Q-Agent"
              required
              autoFocus
            />

            <label className="field-label">ระยะเวลาฝึก (นาที)</label>
            <div className="duration-options">
              {TRAINING_DURATION_OPTIONS.map(minutes => (
                <button
                  key={minutes}
                  type="button"
                  className={trainMinutes === minutes ? "selected" : ""}
                  onClick={() => setTrainMinutes(minutes)}
                >
                  {minutes < 60 ? `${minutes} น.` : "1 ชม."}
                </button>
              ))}
            </div>

            <label className="field-label">อุปกรณ์ฝึก (ล็อกไว้กับโปรเจกต์)</label>
            <div className="compute-options">
              <button
                className={trainingDevice === "cpu" ? "selected" : ""}
                onClick={() => setTrainingDevice("cpu")}
                type="button"
              >
                <Cpu size={16} /><strong>CPU</strong><small>ฝึก Q-Learning ในเครื่อง</small>
              </button>
              <button
                className={trainingDevice === "gpu" ? "selected" : ""}
                onClick={() => setTrainingDevice("gpu")}
                disabled={!gpuSupported}
                type="button"
              >
                <Gauge size={16} /><strong>GPU</strong><small>{gpuSupported ? "เร่งงานแบบ batch ผ่าน WebGL" : "เครื่อง/เบราว์เซอร์นี้ไม่รองรับ WebGL"}</small>
              </button>
            </div>

            <label className="field-label" htmlFor="blackjack-project-algorithm">อัลกอริทึมเริ่มต้น (ล็อกไว้กับโปรเจกต์)</label>
            <select
              id="blackjack-project-algorithm"
              value={trainingAlgorithm}
              onChange={event => setTrainingAlgorithm(event.target.value as TrainingAlgorithm)}
            >
              <option value="qlearning">Q-Learning (tabular)</option>
              <option value="baseline">Baseline (no training)</option>
            </select>
          </ProjectCreateModal>
        )}
        {editingProject && (
          <ProjectEditModal
            gameName="Blackjack"
            projectName={editingProject.name}
            minutes={editingProject.minutes}
            onClose={() => setEditingProject(null)}
            onSave={saveProjectEdits}
          />
        )}
        </>
        )}
          {showGameModal && (
            <div className="game-modal-overlay" onClick={() => setShowGameModal(false)}>
              <section
                className="game-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="game-modal-title"
                onClick={event => event.stopPropagation()}
              >
                <div className="game-modal-header">
                  <div>
                    <span className="eyebrow">ARENA GAME HUB</span>
                    <h2 id="game-modal-title" className="game-modal-title">เลือกเกมที่ต้องการเล่น</h2>
                  </div>
                  <button
                    className="top-icon"
                    onClick={() => setShowGameModal(false)}
                    aria-label="ปิดหน้าต่างเลือกเกม"
                  >
                    <X size={15} />
                  </button>
                </div>
                <p className="modal-lede">เลือกหมวดหมู่ แล้วเลือกเกมเพื่อสลับไปยังสนามที่ต้องการ</p>
                <div className="game-modal-body game-selection-container">
                  <GameSelection
                    category={selectedGameCategory}
                    onSelectCategory={setSelectedGameCategory}
                    activeGame={activeGame}
                    onPickGame={handlePickGame}
                  />
                </div>
              </section>
            </div>
          )}
      </main>
    </div>
  );
}

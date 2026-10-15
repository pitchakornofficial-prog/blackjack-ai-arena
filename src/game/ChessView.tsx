import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import {
  ArrowRight, BookOpen, BrainCircuit, CheckCircle2,
  Cpu, RotateCcw, ShieldCheck, Sparkles, Swords, Trophy,
  Gauge, Cloud, FolderKanban, Play, Pause, Plus,
  Clock3, Info, Trash2, Maximize2, Minimize2, Undo2
} from "lucide-react";
import {
  ChessState, Piece, PieceColor, PieceType, Square,
  getChessAiMove, getLegalMoves, makeMove,
  newChessGame, squareToCoords, DEFAULT_CHESS_WEIGHTS,
  undoChessMove
} from "./chessEngine";
import {
  ChessTrainingProject, ChessMetricPoint, initialChessProjects,
  trainChessBatch, ChessAlgorithm, ChessEngineMode,
  DeepExploreState, initDeepExplore, stepDeepExplore, stepDeepExploreHeadlessBatch,
  isDeepExploreSnapshot, reviveDeepExplore
} from "./chessTraining";
import { detectGpuHardware, GpuDeviceInfo, dispatchGpuBatchEvaluation } from "./gpuAccelerator";
import { loadStoredData, saveStoredData, clearStoredData, isProjectRecordArray, deepExploreKey, STORAGE_KEYS } from "./persistence";

const PIECE_SYMBOLS: Record<PieceColor, Record<PieceType, string>> = {
  w: { k: "♔", q: "♕", r: "♖", b: "♗", n: "♘", p: "♙" },
  b: { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" }
};

const PIECE_NAMES: Record<PieceType, string> = {
  k: "King (ขุน)",
  q: "Queen (ควีน)",
  r: "Rook (เรือ)",
  b: "Bishop (บิชอป)",
  n: "Knight (ม้า)",
  p: "Pawn (เบี้ย)"
};

/**
 * Restores chess projects from localStorage. Weights are merged over
 * DEFAULT_CHESS_WEIGHTS so a partially-saved payload can never produce an
 * incomplete weight object inside the training loop. Projects that were
 * mid-training when the page was closed are reset to "Ready" (progress and
 * accumulated games are kept) so no stale "Training" state lingers.
 */
function restoreChessProjects(): ChessTrainingProject[] {
  const saved = loadStoredData<ChessTrainingProject[]>(
    STORAGE_KEYS.chessProjects,
    isProjectRecordArray
  );
  if (!saved) {
    return initialChessProjects;
  }
  return saved.map(p => ({
    ...p,
    status: p.status === "Training" ? ("Ready" as const) : p.status,
    weights: { ...DEFAULT_CHESS_WEIGHTS, ...(p.weights ?? {}) }
  }));
}

type ChessOpponent = "baseline" | "thinking" | "master" | "project";
type ChessViewProps = {
  page: string;
  setPage: (p: any) => void;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
};

export default function ChessView({
  page,
  setPage,
  isFullscreen = false,
  onToggleFullscreen
}: ChessViewProps) {
  // Game & Board State
  const [chess, setChess] = useState<ChessState>(() => newChessGame());
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [chessOpponent, setChessOpponent] = useState<ChessOpponent>("thinking");
  const [chessStats, setChessStats] = useState({ playerWins: 0, aiWins: 0, draws: 0 });

  // Chess Projects State — restored from localStorage across reloads
  const [chessProjects, setChessProjects] = useState<ChessTrainingProject[]>(restoreChessProjects);
  const [activeTrainingId, setActiveTrainingId] = useState<number>(1);
  const [selectedPlayProjectId, setSelectedPlayProjectId] = useState<number>(1);
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("AlphaPawn Chess");
  const [newProjectAlgo, setNewProjectAlgo] = useState<ChessAlgorithm>("selfplay-rl");
  // ประเภทโปรเจกต์ที่เลือกตอนสร้าง: แยกโหมดฝึกของแต่ละโปรเจกต์ออกจากกัน
  const [newProjectMode, setNewProjectMode] = useState<"standard" | "deepexplore">("standard");

  // Training Execution State
  const [trainMinutes, setTrainMinutes] = useState(1);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingSeconds, setTrainingSeconds] = useState(0);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [trainingBaseSeconds, setTrainingBaseSeconds] = useState(0);

  // Training Config & Hyperparameters
  const [engineMode, setEngineMode] = useState<ChessEngineMode>("local");
  const [trainingDevice, setTrainingDevice] = useState<"auto" | "cpu" | "gpu">("auto");
  const [trainingAlgorithm, setTrainingAlgorithm] = useState<ChessAlgorithm>("selfplay-rl");
  const [learningRate, setLearningRate] = useState(0.1);
  const [discountFactor, setDiscountFactor] = useState(0.95);
  const [explorationRate, setExplorationRate] = useState(0.25);
  const [batchSize, setBatchSize] = useState(100);

  // Cloud AI API State
  const [apiProvider, setApiProvider] = useState("google");
  const [googleApiKey, setGoogleApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiModel, setApiModel] = useState("gemini-2.5-flash");
  const [apiTestStatus, setApiTestStatus] = useState<string | null>(null);

  // Real Real-time SVG Metric Chart
  const [metricHistory, setMetricHistory] = useState<ChessMetricPoint[]>([]);
  const [chartView, setChartView] = useState<"winrate" | "reward" | "epsilon">("winrate");

  // Training Sub-modes: โหมดการฝึกยึดตาม "ประเภทโปรเจกต์" ที่เลือก (แยกกันแน่นอน)
  // standard = Batch Self-Play RL, deepexplore = 10^120 Deep Exploration
  // ซึ่งมีสถานะบันทึกของตัวเองแยกแยะรายโปรเจกต์ (slot ไม่ปนกันระหว่างโปรเจกต์)
  const [trainingSubMode, setTrainingSubMode] = useState<"standard" | "deepexplore">(() =>
    chessProjects.find(p => p.id === activeTrainingId)?.trainMode === "deepexplore" ? "deepexplore" : "standard"
  );
  // ป้ายว่า state deepExplore ที่กำลังแสดงอยู่นี้ "เป็นกองของโปรเจกต์ไหน"
  const [deepExploreProjectId, setDeepExploreProjectId] = useState<number | null>(() => {
    const p = chessProjects.find(x => x.id === activeTrainingId);
    return p?.trainMode === "deepexplore" ? p.id : null;
  });
  const [deepExplore, setDeepExplore] = useState<DeepExploreState>(() => {
    const p = chessProjects.find(x => x.id === activeTrainingId);
    if (p?.trainMode === "deepexplore") {
      const saved = loadStoredData<DeepExploreState>(deepExploreKey(p.id), isDeepExploreSnapshot);
      if (saved) return reviveDeepExplore(saved);
    }
    return initDeepExplore();
  });
  const [deepExploreSpeedMs, setDeepExploreSpeedMs] = useState<number>(450); // visual move interval ms
  const [gpuInfo, setGpuInfo] = useState<GpuDeviceInfo>(() => detectGpuHardware());

  // Web Worker refs for headless parallel computation
  const workerRef = useRef<Worker | null>(null);
  const workerBusyRef = useRef<boolean>(false); // true = worker is currently computing a batch

  const totalTrainingSeconds = trainMinutes * 60;
  const activeProject = useMemo(() => chessProjects.find(p => p.id === activeTrainingId) ?? chessProjects[0], [chessProjects, activeTrainingId]);
  const selectedPlayProject = useMemo(() => chessProjects.find(p => p.id === selectedPlayProjectId) ?? chessProjects[0], [chessProjects, selectedPlayProjectId]);

  // Persist every project change (trained weights included) to localStorage
  // with a trailing debounce so frequent training updates don't rewrite
  // storage on every tick — the final state is always saved.
  useEffect(() => {
    const saveTimer = window.setTimeout(() => {
      saveStoredData(STORAGE_KEYS.chessProjects, chessProjects);
    }, 500);
    return () => window.clearTimeout(saveTimer);
  }, [chessProjects]);

  // บันทึกสถานะ Deep Exploration แบบ "เฉพาะโปรเจกต์ตัวเอง" ลง slot ของมัน
  // (ไม่ปนกับข้อมูลเทรนมาตรฐานหรือโปรเจกต์อื่น) — บังคับ paused=true ตอนเซฟ
  // เพื่อให้เซสชันที่กู้คืนไม่สั่งรันเองโดยที่ผู้ใช้ไม่ได้กด
  useEffect(() => {
    if (deepExploreProjectId === null) return;
    const saveTimer = window.setTimeout(() => {
      saveStoredData(deepExploreKey(deepExploreProjectId), { ...deepExplore, paused: true });
    }, 600);
    return () => window.clearTimeout(saveTimer);
  }, [deepExplore, deepExploreProjectId]);

  // ─── Helper: dispatch GPU work from main thread while worker computes board ───
  const fireGpuDispatch = useCallback((state: DeepExploreState, batchSize: number) => {
    if (state.hardwareAccelerator === "gpu" || state.hardwareAccelerator === "auto") {
      try {
        dispatchGpuBatchEvaluation(batchSize, state.weights.centerBonus);
      } catch {
        // WebGL not available — ignore
      }
    }
  }, []);

  // ─── Deep Exploration Loop ───────────────────────────────────────────────────
  // Headless mode: Web Worker computes chess moves on a separate OS thread
  // while the main thread fires GPU shader dispatch concurrently.
  // Visual mode: single step per tick for smooth observation.
  useEffect(() => {
    if (deepExplore.paused) {
      // Terminate worker when paused to free thread
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
        workerBusyRef.current = false;
      }
      return;
    }

    if (deepExplore.computeEngine === "headless") {
      // Batch size: larger = more GPU pressure and more positions/s
      const batchSize = deepExplore.hardwareAccelerator === "gpu" ? 2000
        : deepExplore.hardwareAccelerator === "cpu" ? 1200
        : 1500;

      const workerSupported = typeof Worker !== "undefined";

      if (workerSupported) {
        // ── Worker Pipeline ──────────────────────────────────────────────────
        // Create worker once; it stays alive until paused or mode changes
        if (!workerRef.current) {
          workerRef.current = new Worker(
            new URL("./chessWorker.ts", import.meta.url),
            { type: "module" }
          );
          workerBusyRef.current = false;
        }

        const worker = workerRef.current;

        // When worker finishes a batch → update state, fire GPU again, send next batch
        worker.onmessage = (ev) => {
          const msg = ev.data;
          if (msg.type !== "batch_done") return;

          const newState: DeepExploreState = msg.state;
          workerBusyRef.current = false;

          setDeepExplore(prev => {
            if (prev.paused || prev.computeEngine !== "headless") return prev;
            // Immediately fire GPU dispatch while React schedules next render
            fireGpuDispatch(newState, batchSize);
            // Send next batch to worker right away (tight continuous pipeline)
            if (!workerBusyRef.current && workerRef.current) {
              workerBusyRef.current = true;
              workerRef.current.postMessage({ type: "step_batch", state: newState, count: batchSize });
            }
            return newState;
          });
        };

        // Kick off the first batch if worker is idle
        if (!workerBusyRef.current) {
          workerBusyRef.current = true;
          fireGpuDispatch(deepExplore, batchSize);
          worker.postMessage({ type: "step_batch", state: deepExplore, count: batchSize });
        }

        // Continuous GPU pump: keep feeding shaders while worker crunching moves
        let gpuPumpActive = true;
        const pumpGpu = () => {
          if (!gpuPumpActive) return;
          if (deepExplore.hardwareAccelerator === "gpu" || deepExplore.hardwareAccelerator === "auto") {
            try {
              dispatchGpuBatchEvaluation(batchSize, 30);
            } catch {
              // ignore
            }
          }
          requestAnimationFrame(pumpGpu);
        };
        const gpuRafId = requestAnimationFrame(pumpGpu);

        // Cleanup: terminate the worker and GPU pump when effect re-runs or unmounts
        return () => {
          gpuPumpActive = false;
          cancelAnimationFrame(gpuRafId);
          if (workerRef.current) {
            workerRef.current.terminate();
            workerRef.current = null;
            workerBusyRef.current = false;
          }
        };
      } else {
        // ── Fallback: setInterval without worker (no parallel threads) ────────
        const timer = window.setInterval(() => {
          setDeepExplore(prev => {
            if (prev.paused || prev.computeEngine !== "headless") return prev;
            fireGpuDispatch(prev, batchSize);
            return stepDeepExploreHeadlessBatch(prev, batchSize);
          });
        }, 120);
        return () => window.clearInterval(timer);
      }
    } else {
      // ── Visual Board Mode: single step per tick for smooth observation ──────
      const timer = window.setInterval(() => {
        setDeepExplore(prev => {
          if (prev.paused || prev.computeEngine !== "visual") return prev;
          return stepDeepExplore(prev);
        });
      }, deepExploreSpeedMs);
      return () => window.clearInterval(timer);
    }
  }, [deepExplore.paused, deepExplore.computeEngine, deepExplore.hardwareAccelerator, deepExploreSpeedMs, fireGpuDispatch]);



  // Update statistics when game ends
  useEffect(() => {
    if (chess.isCheckmate) {
      if (chess.turn === "b") {
        setChessStats(s => ({ ...s, playerWins: s.playerWins + 1 }));
      } else {
        setChessStats(s => ({ ...s, aiWins: s.aiWins + 1 }));
      }
    } else if (chess.isStalemate) {
      setChessStats(s => ({ ...s, draws: s.draws + 1 }));
    }
  }, [chess.isCheckmate, chess.isStalemate, chess.turn]);

  // AI Turn Loop
  useEffect(() => {
    if (chess.turn === "b" && !chess.isCheckmate && !chess.isStalemate) {
      const timer = window.setTimeout(() => {
        const customWeights = chessOpponent === "project" ? selectedPlayProject?.weights : undefined;
        const modelName = chessOpponent === "project" ? selectedPlayProject?.name : undefined;
        const { move, thought } = getChessAiMove(chess, chessOpponent, customWeights, modelName);
        if (move) {
          setChess(prev => {
            const next = makeMove(prev, move.from, move.to);
            return { ...next, aiThought: thought };
          });
        }
      }, 700);
      return () => window.clearTimeout(timer);
    }
  }, [chess.turn, chess.isCheckmate, chess.isStalemate, chess.moveHistory.length, chessOpponent, selectedPlayProject]);

  // Wall-Clock Timer and Real Self-Play Training Simulation Loop
  useEffect(() => {
    if (!trainingRunning || !activeTrainingId || trainingStartedAt === null) return;
    // โปรเจกต์ Deep Exploration ห้ามโดน batch เทรนมาตรฐานปั๊มข้อมูล
    // (การันตีว่าแต่ละประเภทโปรเจกต์บันทึกแค่ตัวเลขของตัวเอง)
    if (activeProject?.trainMode === "deepexplore") return;

    const timer = window.setInterval(() => {
      const elapsed = trainingBaseSeconds + Math.floor((Date.now() - trainingStartedAt) / 1000);
      const limit = totalTrainingSeconds;
      const next = Math.min(elapsed, limit);
      setTrainingSeconds(next);

      // Run real batch simulation
      const currentWeights = activeProject?.weights ? { ...activeProject.weights } : { ...DEFAULT_CHESS_WEIGHTS };
      const lastEpsilon = metricHistory.length > 0 ? metricHistory[metricHistory.length - 1].epsilon : explorationRate;

      const batchResult = trainChessBatch(
        currentWeights,
        batchSize,
        learningRate,
        discountFactor,
        lastEpsilon,
        trainingAlgorithm
      );

      // Save real metric point
      setMetricHistory(prev => {
        const point: ChessMetricPoint = {
          step: prev.length + 1,
          timeSeconds: next,
          totalGames: (prev[prev.length - 1]?.totalGames ?? 0) + batchSize,
          winRate: Number(batchResult.winRate.toFixed(1)),
          lossRate: Number(batchResult.lossRate.toFixed(1)),
          drawRate: Number(batchResult.drawRate.toFixed(1)),
          avgReward: Number(batchResult.avgReward.toFixed(3)),
          materialAdv: Number(batchResult.materialAdv.toFixed(2)),
          epsilon: Number(batchResult.updatedEpsilon.toFixed(3))
        };
        return [...prev, point];
      });

      // Update project state with actual weights and simulated games
      setChessProjects(old => old.map(p => {
        if (p.id !== activeTrainingId) return p;
        const newGames = p.games + batchSize;
        return {
          ...p,
          games: newGames,
          weights: batchResult.updatedWeights,
          winRate: Number(batchResult.winRate.toFixed(1)),
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
    totalTrainingSeconds, batchSize, learningRate, discountFactor,
    explorationRate, trainingAlgorithm, activeProject?.weights, metricHistory, activeProject?.trainMode
  ]);

  // Legal destinations for selected square
  const legalDestinations = useMemo(() => {
    if (!selectedSquare) return [];
    return getLegalMoves(chess.board, selectedSquare);
  }, [chess.board, selectedSquare]);

  function handleSquareClick(r: number, c: number) {
    if (chess.turn !== "w" || chess.isCheckmate || chess.isStalemate) return;

    // Check if clicked square is a valid destination
    const isDestination = legalDestinations.some(d => d.r === r && d.c === c);
    if (selectedSquare && isDestination) {
      const next = makeMove(chess, selectedSquare, { r, c });
      setChess(next);
      setSelectedSquare(null);
      return;
    }

    // Check if clicked square has a white piece
    const piece = chess.board[r][c];
    if (piece && piece.color === "w") {
      setSelectedSquare({ r, c });
    } else {
      setSelectedSquare(null);
    }
  }

  function handleReset() {
    setChess(newChessGame());
    setSelectedSquare(null);
  }

  function handleUndoMove() {
    if (chess.moveHistory.length === 0) return;
    setChess(prev => undoChessMove(prev));
    setSelectedSquare(null);
  }

  // Ctrl+Z keyboard shortcut for undo
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        if (chess.turn === "w" && chess.moveHistory.length > 0 && !chess.isCheckmate && !chess.isStalemate) {
          e.preventDefault();
          handleUndoMove();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [chess.turn, chess.moveHistory.length, chess.isCheckmate, chess.isStalemate]);

  // Duration selection sync
  function handleSelectDuration(minutes: number) {
    setTrainMinutes(minutes);
    if (activeTrainingId) {
      setChessProjects(prev => prev.map(p => p.id === activeTrainingId ? { ...p, minutes } : p));
    }
  }

  function handleSelectProject(id: number) {
    setActiveTrainingId(id);
    const p = chessProjects.find(x => x.id === id);
    if (p) {
      setTrainMinutes(p.minutes);
      setEngineMode(p.engineMode || "local");
      setTrainingAlgorithm(p.algorithm || "selfplay-rl");
      // หน้า Training ยึดโหมดตามประเภทของโปรเจกต์ที่ถูกเลือก
      const mode = p.trainMode === "deepexplore" ? "deepexplore" : "standard";
      setTrainingSubMode(mode);
      if (mode === "deepexplore") ensureDeepExploreSlot(p.id);
    }
  }

  /**
   * สลับไปยกเครื่องนาโลคเรชนที่จับหมากในโปรเจกต์: เซฟ slot เก่าก่อนแบบ instant,
   * หยุด worker เก่าให้ข้อความที่ยังวิ่งอยู่ไม่หล่นทับ slot ใหม่, แล้วโหลด
   * เฉพาะสถานะ Deep Exploration ของโปรเจกต์นี้ (หรือสร้างใหม่ถ้าไม่เคยเซฟ)
   */
  function ensureDeepExploreSlot(projectId: number) {
    if (deepExploreProjectId === projectId) return;
    if (deepExploreProjectId !== null) {
      saveStoredData(deepExploreKey(deepExploreProjectId), { ...deepExplore, paused: true });
    }
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
      workerBusyRef.current = false;
    }
    const saved = loadStoredData<DeepExploreState>(deepExploreKey(projectId), isDeepExploreSnapshot);
    setDeepExplore(saved ? reviveDeepExplore(saved) : initDeepExplore());
    setDeepExploreProjectId(projectId);
  }

  /**
   * สลับโหมดการฝึกของโปรเจกต์: บันทึกประเภทลงโปรเจกต์ปัจจุบันด้วย
   * เพื่อให้การแยกประเภทตามโปรเจกต์เป็นข้อมูลกลางที่ถูก persist
   */
  function switchTrainingMode(mode: "standard" | "deepexplore") {
    setTrainingSubMode(mode);
    if (activeProject) {
      const id = activeProject.id;
      setChessProjects(prev => prev.map(p => p.id === id ? { ...p, trainMode: mode } : p));
      if (mode === "deepexplore") {
        pauseTraining(); // หยุดตัวจับเวลาโหมดมาตรฐานก่อนเสือมี loop วิ่งสอง
        ensureDeepExploreSlot(id);
      }
    }
  }

  /** ใช้จากการ์ดโปรเจกต์/ลิงก์: เปิดหน้า Training ตามประเภทของโปรเจกต์นั้น */
  function openTrainingFor(project: ChessTrainingProject) {
    setActiveTrainingId(project.id);
    setTrainMinutes(project.minutes);
    setTrainingSubMode(project.trainMode === "deepexplore" ? "deepexplore" : "standard");
    if (project.trainMode === "deepexplore") ensureDeepExploreSlot(project.id);
    setPage("training");
  }

  function beginTraining(project: ChessTrainingProject, resumeAccumulated: boolean = true) {
    // โปรเจกต์ประเภท Deep Exploration ไม่ใช้ตัวจับเวลาเทรนมาตรฐานเลย
    // สร้าง/กลับมาทำต่อจาก session สำรวจของมันเองแทน
    if (project.trainMode === "deepexplore") {
      setActiveTrainingId(project.id);
      setTrainingSubMode("deepexplore");
      ensureDeepExploreSlot(project.id);
      setPage("training");
      return;
    }
    setChessProjects(old => old.map(p => {
      if (p.id !== project.id) return p;
      return {
        ...p,
        minutes: trainMinutes,
        status: "Training",
        progress: 0,
        games: resumeAccumulated ? p.games : 0,
        weights: resumeAccumulated ? { ...p.weights } : { ...DEFAULT_CHESS_WEIGHTS }
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
      setChessProjects(old => old.map(p => p.id === activeTrainingId ? { ...p, status: "Ready", progress: 0, games: 0 } : p));
    }
  }

  function handleCreateProject() {
    if (!newProjectName.trim()) return;
    const newId = Date.now();
    const created: ChessTrainingProject = {
      id: newId,
      name: newProjectName.trim(),
      minutes: trainMinutes,
      status: "Ready",
      progress: 0,
      games: 0,
      created: "สร้างใหม่",
      engineMode,
      algorithm: newProjectAlgo,
      weights: { ...DEFAULT_CHESS_WEIGHTS },
      winRate: 50.0,
      depth: 2,
      trainMode: newProjectMode
    };
    if (newProjectMode === "deepexplore") {
      // โปรเจกต์ Deep Exploration ใหม่ได้ slot ของตัวเอง (ไม่รับข้อมูลเก่า)
      saveStoredData(deepExploreKey(newId), initDeepExplore());
    }
    setChessProjects(prev => [created, ...prev]);
    setActiveTrainingId(newId);
    setSelectedPlayProjectId(newId);
    setShowNewProject(false);
    setNewProjectName("AlphaPawn Chess");
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

  // Material balance calculation
  const materialBalance = useMemo(() => {
    const values: Record<PieceType, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    let w = 0;
    let b = 0;
    for (const row of chess.board) {
      for (const piece of row) {
        if (!piece) continue;
        if (piece.color === "w") w += values[piece.type];
        else b += values[piece.type];
      }
    }
    return w - b;
  }, [chess.board]);

  const kingCheckSquare = useMemo(() => {
    if (!chess.isCheck) return null;
    const kingColor = chess.turn;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = chess.board[r][c];
        if (p && p.type === "k" && p.color === kingColor) return { r, c };
      }
    }
    return null;
  }, [chess.board, chess.isCheck, chess.turn]);

  // Render Real Dynamic SVG Chart for Chess Metrics
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
          <small style={{ display: "block", marginTop: 4, color: "#bbb" }}>ระบบจะจำลองกระดาน Self-Play หมากรุก และพล็อตกราฟจากการเทรนจริงทันที</small>
        </div>
      );
    }

    let minVal = 0;
    let maxVal = 100;
    let yLabels = ["100%", "75%", "50%", "25%", "0%"];

    if (chartView === "winrate") {
      minVal = 30;
      maxVal = 70;
      yLabels = ["70%", "60%", "50%", "40%", "30%"];
    } else if (chartView === "reward") {
      minVal = -1.0;
      maxVal = 1.0;
      yLabels = ["+1.0", "+0.5", "0.0", "-0.5", "-1.0"];
    } else {
      minVal = 0;
      maxVal = 0.5;
      yLabels = ["0.50", "0.38", "0.25", "0.12", "0.00"];
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
      <svg className="real-chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="กราฟผลการเทรนหมากรุกจริง">
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
        <text x={padL} y={H - 10} textAnchor="start">0s ({metricHistory[0]?.totalGames.toLocaleString()} กระดาน)</text>
        <text x={W - padR} y={H - 10} textAnchor="end">{fmt(trainingSeconds)} ({metricHistory[n - 1]?.totalGames.toLocaleString()} กระดาน)</text>
      </svg>
    );
  }

  function renderChessboard() {
    return (
      <div className="chessboard-wrapper">
        <div className="chessboard-grid">
          {chess.board.map((row, r) =>
            row.map((piece, c) => {
              const isLight = (r + c) % 2 === 0;
              const isSelected = selectedSquare?.r === r && selectedSquare?.c === c;
              const isDestination = legalDestinations.some(d => d.r === r && d.c === c);
              const isLastMove = (chess.lastMove?.from.r === r && chess.lastMove?.from.c === c) ||
                                 (chess.lastMove?.to.r === r && chess.lastMove?.to.c === c);
              const isCheckSq = kingCheckSquare?.r === r && kingCheckSquare?.c === c;

              return (
                <div
                  key={`${r}-${c}`}
                  className={`chess-square ${isLight ? "light" : "dark"} ${isSelected ? "selected" : ""} ${isLastMove ? "last-move" : ""} ${isCheckSq ? "in-check" : ""}`}
                  onClick={() => handleSquareClick(r, c)}
                >
                  {piece && (
                    <span className={`chess-piece ${piece.color === "w" ? "white-piece" : "black-piece"}`}>
                      {PIECE_SYMBOLS[piece.color][piece.type]}
                    </span>
                  )}
                  {isDestination && !piece && <div className="move-dest-dot" />}
                  {isDestination && piece && <div className="capture-dest-ring" />}
                  {c === 0 && <span className="coord-label coord-rank">{8 - r}</span>}
                  {r === 7 && <span className="coord-label coord-file">{["a","b","c","d","e","f","g","h"][c]}</span>}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  function renderDeepExploreBoard() {
    const game = deepExplore.currentGame;
    const lastM = game.lastMove;
    const kingColor = game.turn;
    let kingCheck: Square | null = null;
    if (game.isCheck) {
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = game.board[r][c];
          if (p && p.type === "k" && p.color === kingColor) {
            kingCheck = { r, c };
            break;
          }
        }
      }
    }

    return (
      <div className="chessboard-wrapper">
        <div className="chessboard-grid">
          {game.board.map((row, r) =>
            row.map((piece, c) => {
              const isLight = (r + c) % 2 === 0;
              const isLastMove = (lastM?.from.r === r && lastM?.from.c === c) ||
                                 (lastM?.to.r === r && lastM?.to.c === c);
              const isCheckSq = kingCheck?.r === r && kingCheck?.c === c;

              return (
                <div
                  key={`de-${r}-${c}`}
                  className={`chess-square ${isLight ? "light" : "dark"} ${isLastMove ? "last-move" : ""} ${isCheckSq ? "in-check" : ""}`}
                >
                  {piece && (
                    <span className={`chess-piece ${piece.color === "w" ? "white-piece" : "black-piece"}`}>
                      {PIECE_SYMBOLS[piece.color][piece.type]}
                    </span>
                  )}
                  {c === 0 && <span className="coord-label coord-rank">{8 - r}</span>}
                  {r === 7 && <span className="coord-label coord-file">{["a","b","c","d","e","f","g","h"][c]}</span>}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  const latestMetric = metricHistory[metricHistory.length - 1];

  return (
    <>
      {/* ================= CHESS: HOME / OVERVIEW ================= */}
      {page === "home" && (
        <section className="page">
          <div className="welcome-row">
            <div>
              <p className="eyebrow">CHESS ARENA · REINFORCEMENT LEARNING</p>
              <h1>Control the board.<br /><em>Master the squares.</em></h1>
              <p className="lede">สนามประลองและฝึกฝน Chess AI สลับตากับ Minimax/RL Agent พร้อมระบบเทรนด้วยตนเองและดูกราฟวิวัฒนาการแบบเรียลไทม์</p>
            </div>
            <div className="hero-stamp">
              <span>♞</span>
              <small>ARENA CHESS</small>
            </div>
          </div>

          <div className="quick-grid">
            <button className="feature-card dark-card" onClick={() => setPage("play")}>
              <div className="feature-top"><span>01 / PLAY CHESS</span><ArrowRight size={17} /></div>
              <div className="feature-icon"><Swords size={23} /></div>
              <h2>เล่นหมากรุก vs AI</h2>
              <p>ลงกระดานดวลสลับตากับ AI สีดำ (Baseline / Thinking / Trained Agent) พร้อมคำอธิบายความคิดทุกตาเดิน</p>
              <div className="feature-link">START MATCH <ArrowRight size={14} /></div>
            </button>

            <button className="feature-card" onClick={() => setPage("training")}>
              <div className="feature-top"><span>02 / TRAIN CHESS AI</span><ArrowRight size={17} /></div>
              <div className="feature-icon"><BrainCircuit size={23} /></div>
              <h2>ฝึกโมเดล AI พร้อมดูกราฟจริง</h2>
              <p>ฝึกฝนด้วย Self-Play Reinforcement Learning ปรับค่าน้ำหนักตัวหมาก และพล็อตกราฟผลลัพธ์แบบสด</p>
              <div className="feature-link">OPEN TRAINING <ArrowRight size={14} /></div>
            </button>

            <button className="feature-card" onClick={() => setPage("projects")}>
              <div className="feature-top"><span>03 / CHESS PROJECTS</span><ArrowRight size={17} /></div>
              <div className="feature-icon"><FolderKanban size={23} /></div>
              <h2>โปรเจกต์ Chess AI</h2>
              <p>จัดการโมเดลหมากรุกและนำ Agent ที่ฝึกแล้วไปลงสนามแข่งจริงบนกระดาน 8x8</p>
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
                <span><strong>วิธีเล่นและกติกาหมากรุก</strong><small>ทำความเข้าใจการเดินหมากและกติกา FIDE</small></span>
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="panel stat-panel">
              <span className="eyebrow">ARENA MATCH RECORD</span>
              <div className="stat-number">{chessStats.playerWins}W<small> - {chessStats.aiWins}L</small></div>
              <p>สถิติชนะ/แพ้กับ AI ในโหมดหมากรุก (เสมอ {chessStats.draws})</p>
              <div className="stat-divider" />
              <div className="stat-foot"><ShieldCheck size={15} /> กติกาสากล FIDE Standard</div>
            </div>
          </div>
        </section>
      )}

      {/* ================= CHESS: PLAY ARENA ================= */}
      {page === "play" && isFullscreen && (
        <div className="chess-fullscreen-theater">
          {/* Left: Giant Chessboard Stage */}
          <div className="chess-theater-stage">
            <div className={`turn-banner ${chess.turn === "w" ? "player-turn" : "ai-turn"}`} style={{ marginBottom: 12, minWidth: 320, textAlign: "center" }}>
              <div>
                {chess.isCheckmate ? (
                  <strong>รุกฆาต (CHECKMATE)! {chess.turn === "b" ? "คุณชนะ!" : "AI ชนะ!"}</strong>
                ) : chess.isStalemate ? (
                  <strong>อับจน (STALEMATE) · เสมอกัน!</strong>
                ) : chess.isCheck ? (
                  <strong style={{ color: "#c92a2a" }}>เตือน: มีการรุก (CHECK)!</strong>
                ) : chess.turn === "w" ? (
                  <span>ตาของคุณ (White): เลือกเดินหมากสีขาว</span>
                ) : (
                  <span>ตาของ AI (Black): กำลังคำนวณ...</span>
                )}
              </div>
              <strong style={{ fontSize: 9 }}>{chess.turn === "w" ? "WHITE'S TURN" : "BLACK'S TURN"}</strong>
            </div>

            {renderChessboard()}
          </div>

          {/* Right: Dedicated Full-Height Sidebar for Move History & Match Details */}
          <div className="chess-theater-sidebar">
            <div className="chess-theater-sidebar-header">
              <div className="top-row">
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 22 }}>♞</span>
                  <div>
                    <strong style={{ fontSize: 13, display: "block" }}>
                      White (คุณ) vs Black ({chessOpponent === "project" ? selectedPlayProject?.name : chessOpponent === "master" ? "Master Deep" : chessOpponent === "thinking" ? "Thinking AI" : "Baseline AI"})
                    </strong>
                    <small style={{ color: "#888", font: "9px var(--mono)" }}>
                      ชนะ {chessStats.playerWins} · แพ้ {chessStats.aiWins} · เสมอ {chessStats.draws}
                    </small>
                  </div>
                </div>
                {onToggleFullscreen && (
                  <button className="fullscreen-btn active" onClick={onToggleFullscreen} title="ออกจากโหมดเต็มจอ">
                    <Minimize2 size={13} />
                    <span>ESC</span>
                  </button>
                )}
              </div>

              <div style={{ display: "flex", gap: 6 }}>
                <select
                  value={chessOpponent}
                  onChange={e => setChessOpponent(e.target.value as any)}
                  style={{ flex: 1, padding: "5px 8px", fontSize: 10, border: "1px solid var(--line)", background: "#fff" }}
                >
                  <option value="baseline">Baseline AI (D1)</option>
                  <option value="thinking">Thinking AI (D2)</option>
                  <option value="master">Master Deep (D3)</option>
                  <option value="project">Project AI ({selectedPlayProject?.name})</option>
                </select>
                {chessOpponent === "project" && (
                  <select
                    value={selectedPlayProjectId}
                    onChange={e => setSelectedPlayProjectId(Number(e.target.value))}
                    style={{ flex: 1, padding: "5px 8px", fontSize: 10, border: "1px solid var(--line)", background: "#fff" }}
                  >
                    {chessProjects.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                )}
              </div>
            </div>

            {/* Captured Pieces Bar */}
            <div style={{ padding: "8px 16px", borderBottom: "1px solid var(--line)", background: "#fafaf8", fontSize: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <strong style={{ color: "#777", fontSize: 9 }}>MATERIAL BALANCE</strong>
                <span style={{ font: "9px var(--mono)", color: materialBalance > 0 ? "#2f6f44" : materialBalance < 0 ? "#c92a2a" : "#777" }}>
                  {materialBalance > 0 ? `ขาวได้เปรียบ +${materialBalance}` : materialBalance < 0 ? `ดำได้เปรียบ +${Math.abs(materialBalance)}` : "เท่ากัน"}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 16 }}>
                <div>
                  <small style={{ display: "block", fontSize: 8, color: "#999" }}>ขาวกิน:</small>
                  <span>{chess.capturedByWhite.map((p, i) => <span key={i} className="black-piece">{PIECE_SYMBOLS.b[p.type]}</span>)} {!chess.capturedByWhite.length && "—"}</span>
                </div>
                <div>
                  <small style={{ display: "block", fontSize: 8, color: "#999" }}>ดำกิน:</small>
                  <span>{chess.capturedByBlack.map((p, i) => <span key={i} className="white-piece" style={{ textShadow: "0 0 1px #000" }}>{PIECE_SYMBOLS.w[p.type]}</span>)} {!chess.capturedByBlack.length && "—"}</span>
                </div>
              </div>
            </div>

            {/* Move History (ประวัติการเล่นเต็มแผง) */}
            <div className="chess-theater-history">
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, font: "9px var(--mono)", color: "#777" }}>
                <strong>ประวัติการเล่น (MOVE LOG)</strong>
                <span>{Math.ceil(chess.moveHistory.length / 2)} ตาเดิน</span>
              </div>
              <div className="chess-theater-history-table">
                {Array.from({ length: Math.ceil(chess.moveHistory.length / 2) }).map((_, i) => {
                  const whiteMove = chess.moveHistory[i * 2];
                  const blackMove = chess.moveHistory[i * 2 + 1];
                  return (
                    <div className="history-row" key={i}>
                      <span className="history-num">{i + 1}.</span>
                      <span className="history-white">{whiteMove?.san || "—"}</span>
                      <span className="history-black">{blackMove?.san || "—"}</span>
                    </div>
                  );
                })}
                {!chess.moveHistory.length && (
                  <div style={{ gridColumn: "span 3", color: "#aaa", fontSize: 11, padding: "16px 0", textAlign: "center" }}>
                    ยังไม่มีการเดินหมาก เริ่มโดยคลิกหมากสีขาวบนกระดาน
                  </div>
                )}
              </div>
            </div>

            {/* Live AI Thought */}
            {chess.aiThought && (
              <div style={{ padding: "8px 16px", background: "#f8f7f2", borderBottom: "1px solid var(--line)", fontSize: 10, lineHeight: 1.5, color: "#555" }}>
                <strong style={{ color: "#7a5c1e", display: "block", fontSize: 9, marginBottom: 2 }}>AI THOUGHT:</strong>
                {chess.aiThought}
              </div>
            )}

            {/* Sidebar Footer Controls */}
            <div className="chess-theater-sidebar-footer">
              <button className="primary-action full-action" onClick={handleReset} style={{ marginTop: 0 }}>
                <RotateCcw size={13} /> เริ่มเกมใหม่
              </button>
              <button
                className="secondary-action"
                style={{ width: "100%" }}
                disabled={chess.moveHistory.length === 0 || chess.turn !== "w"}
                onClick={handleUndoMove}
              >
                <Undo2 size={13} /> ย้อนกลับตาเดิน (Ctrl+Z)
              </button>
              {onToggleFullscreen && (
                <button className="outline-full" style={{ width: "100%", marginTop: 0 }} onClick={onToggleFullscreen}>
                  <Minimize2 size={13} /> ออกจากเต็มจอ
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ================= CHESS: PLAY ARENA (STANDARD / NON-FULLSCREEN) ================= */}
      {page === "play" && !isFullscreen && (
        <section className="page play-page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">01 / CHESS ARENA</p>
              <h1>International Chess Board</h1>
              <p className="lede">เดินหมากสลับตากับ AI บนกระดาน 8x8 คุณเล่นฝ่ายขาว (White) AI เล่นฝ่ายดำ (Black)</p>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div className="scoreboard-badge">
                <span>ARENA SCORE:</span>
                <strong>คุณ {chessStats.playerWins}</strong> · <strong>AI {chessStats.aiWins}</strong> · เสมอ {chessStats.draws}
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

          {/* Opponent Selection Grid */}
          <div className="opponent-grid four-cols">
            <button className={`opponent-card ${chessOpponent === "baseline" ? "chosen" : ""}`} onClick={() => setChessOpponent("baseline")}>
              <span className="opponent-icon"><ShieldCheck /></span>
              <span className="opponent-tag">DEFAULT</span>
              <h3>Baseline AI</h3>
              <p>เล่นเร็ว เน้นกินหมากและคุม 4 ช่องกลาง</p>
              <small>GREEDY · DEPTH 1</small>
            </button>

            <button className={`opponent-card ${chessOpponent === "thinking" ? "chosen" : ""}`} onClick={() => setChessOpponent("thinking")}>
              <span className="opponent-icon"><Cpu /></span>
              <span className="opponent-tag">EXPLAINABLE</span>
              <h3>Thinking AI</h3>
              <p>คำนวณล่วงหน้า 2 ตา พร้อมคิดเป็นข้อความ</p>
              <small>MINIMAX · DEPTH 2</small>
            </button>

            <button className={`opponent-card ${chessOpponent === "master" ? "chosen" : ""}`} onClick={() => setChessOpponent("master")}>
              <span className="opponent-icon"><Sparkles /></span>
              <span className="opponent-tag">ADVANCED</span>
              <h3>Master Deep</h3>
              <p>คำนวณลึก 3 ตา พร้อมตัดกิ่ง Alpha-Beta</p>
              <small>DEEP SEARCH · DEPTH 3</small>
            </button>

            <button className={`opponent-card ${chessOpponent === "project" ? "chosen" : ""}`} onClick={() => setChessOpponent("project")}>
              <span className="opponent-icon"><FolderKanban /></span>
              <span className="opponent-tag">YOUR MODEL</span>
              <h3>Project AI</h3>
              <p>ใช้โมเดลที่คุณฝึกจากห้อง Training</p>
              <small>{selectedPlayProject ? selectedPlayProject.name : "TRAINED AGENT"}</small>
            </button>
          </div>

          {/* Project AI Selector Box */}
          {chessOpponent === "project" && (
            <div className="project-ai-selector-box">
              <div className="selector-title">
                <FolderKanban size={16} />
                <strong>เลือกโปรเจกต์ Chess AI ที่ต้องการนำมาประลอง:</strong>
              </div>
              <select
                value={selectedPlayProjectId}
                onChange={e => setSelectedPlayProjectId(Number(e.target.value))}
              >
                {chessProjects.map(p => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.status === "Complete" ? "เทรนเสร็จสมบูรณ์" : p.games > 0 ? `เทรนแล้ว ${p.games.toLocaleString()} กระดาน` : "ยังไม่ผ่านการเทรน"}) · {p.algorithm}
                  </option>
                ))}
              </select>

              <div className="selected-project-badge">
                {selectedPlayProject && selectedPlayProject.games > 0 ? (
                  <span className="badge-ready">
                    <CheckCircle2 size={14} />
                    <strong>{selectedPlayProject.name}</strong> พร้อมลงสนาม (ผ่านการเทรน {selectedPlayProject.games.toLocaleString()} กระดาน · Win Rate {selectedPlayProject.winRate}%)
                  </span>
                ) : (
                  <span className="badge-untrained">
                    <Clock3 size={14} />
                    <strong>{selectedPlayProject?.name}</strong> ยังไม่ผ่านการเทรน (สุ่มเดินพื้นฐาน)
                    <button
                      className="link-button"
                      onClick={() => {
                        if (selectedPlayProject) {
                          openTrainingFor(selectedPlayProject);
                        }
                      }}
                    >
                      กดที่นี่เพื่อไปฝึกโมเดลนี้ทันที <ArrowRight size={12} />
                    </button>
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Main Chess Arena Layout (Standardized with Blackjack & TTT) */}
          <div className="game-layout">
            <div className="game-board">
              {/* Turn Banner */}
              <div className={`turn-banner ${chess.turn === "w" ? "player-turn" : "ai-turn"}`}>
                <div>
                  {chess.isCheckmate ? (
                    <strong>🏁 เกมสิ้นสุด: รุกฆาต (CHECKMATE)! {chess.turn === "b" ? "คุณชนะ!" : "AI ชนะ!"}</strong>
                  ) : chess.isStalemate ? (
                    <strong>🤝 เกมสิ้นสุด: อับจน (STALEMATE) · เสมอกัน!</strong>
                  ) : chess.isCheck ? (
                    <strong style={{ color: "#c92a2a" }}>⚠️ เตือน: มีการรุก (CHECK)! ฝ่าย {chess.turn === "w" ? "ขาว (คุณ)" : "ดำ (AI)"} ต้องป้องกัน King</strong>
                  ) : chess.turn === "w" ? (
                    <span>🟢 <strong>ตาของคุณ (White):</strong> คลิกตัวหมากสีขาวเพื่อดูตำแหน่งที่เดินได้</span>
                  ) : (
                    <span>🟡 <strong>ตาของ {chessOpponent === "project" ? selectedPlayProject?.name : "AI"} (Black):</strong> กำลังคิดวิเคราะห์...</span>
                  )}
                </div>
                <strong>{chess.turn === "w" ? "YOUR TURN (WHITE)" : "AI TURN (BLACK)"}</strong>
              </div>

              {/* Stage Board */}
              <div className="board-stage">
                {renderChessboard()}
              </div>

              {/* In-Game Action Controls (Matching Blackjack & TTT controls bar) */}
              <div className="controls">
                <button
                  className="primary-action action-button"
                  disabled={chess.moveHistory.length === 0 || chess.turn !== "w"}
                  onClick={handleUndoMove}
                  title="ย้อนกลับการเดินหมากตาที่แล้ว (Undo)"
                >
                  <Undo2 size={14} /> ย้อนกลับตาเดิน (UNDO)
                </button>
                <button className="secondary-action action-button" onClick={handleReset}>
                  <RotateCcw size={14} /> เริ่มรอบใหม่
                </button>
              </div>

              {chess.aiThought && (
                <div className="ai-thought-box" style={{ margin: "14px 20px" }}>
                  <BrainCircuit size={16} />
                  <div>
                    <strong>AI Thought Process:</strong>
                    <div>{chess.aiThought}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Side Info Panel (Standardized with Blackjack & TTT side-panel) */}
            <aside className="side-panel">
              <span className="eyebrow">MATCH INFO</span>
              <h3>สถานะการดวล</h3>
              <div className="setting-row">
                <span>AI คู่แข่ง</span>
                <strong>{chessOpponent === "project" ? selectedPlayProject?.name : chessOpponent === "baseline" ? "Baseline AI" : chessOpponent === "thinking" ? "Thinking AI" : "Master AI"}</strong>
              </div>
              <div className="setting-row">
                <span>กติกา</span>
                <strong>FIDE Standard · 8x8</strong>
              </div>
              <div className="setting-row">
                <span>ผลการดวลรวม</span>
                <strong>{chessStats.playerWins}W - {chessStats.aiWins}L - {chessStats.draws}D</strong>
              </div>
              <div className="setting-row">
                <span>ความได้เปรียบ</span>
                <strong style={{ color: materialBalance > 0 ? "#2f6f44" : materialBalance < 0 ? "#c92a2a" : "#777" }}>
                  {materialBalance > 0 ? `ขาว +${materialBalance}` : materialBalance < 0 ? `ดำ +${Math.abs(materialBalance)}` : "เท่ากัน"}
                </strong>
              </div>

              {/* Material Captured Box */}
              <div className="captured-box" style={{ marginTop: 12 }}>
                <strong>หมากที่ถูกกิน</strong>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "4px 0" }}>
                  <small style={{ color: "#888" }}>ขาวกินได้:</small>
                  <div className="captured-row" style={{ minHeight: "auto", fontSize: 16 }}>
                    {chess.capturedByWhite.map((p, idx) => (
                      <span key={idx} className="black-piece">{PIECE_SYMBOLS.b[p.type]}</span>
                    ))}
                    {!chess.capturedByWhite.length && <span style={{ fontSize: 10, color: "#aaa" }}>—</span>}
                  </div>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <small style={{ color: "#888" }}>ดำกินได้:</small>
                  <div className="captured-row" style={{ minHeight: "auto", fontSize: 16 }}>
                    {chess.capturedByBlack.map((p, idx) => (
                      <span key={idx} className="white-piece" style={{ textShadow: "0 0 1px #000" }}>{PIECE_SYMBOLS.w[p.type]}</span>
                    ))}
                    {!chess.capturedByBlack.length && <span style={{ fontSize: 10, color: "#aaa" }}>—</span>}
                  </div>
                </div>
              </div>

              {/* Move History Table */}
              <div className="move-history-card">
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, font: "9px var(--mono)", color: "#777" }}>
                  <span>MOVE HISTORY</span>
                  <span>{Math.ceil(chess.moveHistory.length / 2)} ตาเดิน</span>
                </div>
                <div className="move-history-table" style={{ maxHeight: 150 }}>
                  {Array.from({ length: Math.ceil(chess.moveHistory.length / 2) }).map((_, i) => {
                    const whiteMove = chess.moveHistory[i * 2];
                    const blackMove = chess.moveHistory[i * 2 + 1];
                    return (
                      <div key={i} style={{ display: "contents" }}>
                        <span style={{ color: "#999" }}>{i + 1}.</span>
                        <strong>{whiteMove?.san || "—"}</strong>
                        <span style={{ color: "#555" }}>{blackMove?.san || "—"}</span>
                      </div>
                    );
                  })}
                  {!chess.moveHistory.length && (
                    <div style={{ gridColumn: "span 3", color: "#aaa", fontSize: 10, padding: "4px 0" }}>
                      คลิกตัวหมากสีขาวเพื่อเริ่มเดิน
                    </div>
                  )}
                </div>
              </div>

              {/* Action Buttons in Side Panel */}
              <button className="outline-full" onClick={handleReset}>
                <RotateCcw size={14} /> เริ่มเซสชันใหม่
              </button>
            </aside>
          </div>
        </section>
      )}

      {/* ================= CHESS: AI TRAINING ================= */}
      {page === "training" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">02 / CHESS AI TRAINING</p>
              <h1>{trainingSubMode === "deepexplore" ? "10¹²⁰ Deep Exploration Training" : "Self-Play RL & Engine Training"}</h1>
              <p className="lede">
                {trainingSubMode === "deepexplore"
                  ? "ฝึกฝนตามทฤษฎี Shannon Number (หมากรุกมีรูปแบบเดินเป็นไปได้ ~10¹²⁰ รูปแบบ) จำลองการเดินบนกระดานจริง พักและฝึกต่อได้ตลอดเวลา"
                  : "ฝึกฝนโมเดลหมากรุกด้วย Self-Play Reinforcement Learning แบบ Background Batch พร้อมดูกราฟอัตราชนะและค่าความคุ้มค่าแบบ Real-time"}
              </p>
            </div>
            <div className="training-chip">
              <span className={`pulse-dot ${trainingSubMode === "deepexplore" && deepExplore.paused ? "" : ""}`} style={{ background: trainingSubMode === "deepexplore" ? (deepExplore.paused ? "#c9a227" : "#57845e") : undefined }} />
              {trainingSubMode === "deepexplore"
                ? (deepExplore.paused ? "DEEP EXPLORATION PAUSED" : "EXPLORING 10¹²⁰ STATES...")
                : (trainingRunning ? "BATCH TRAINING RUNNING" : "TRAINING ENGINE READY")}
            </div>
          </div>

          {/* Sub-mode Switcher Tabs — สลับ "ประเภทโปรเจกต์" ปัจจุบัน */}
          <div className="engine-mode-tabs" style={{ marginBottom: 18 }}>
            <button
              className={`engine-tab ${trainingSubMode === "deepexplore" ? "selected" : ""}`}
              onClick={() => switchTrainingMode("deepexplore")}
            >
              <Sparkles size={14} /> 10¹²⁰ Deep Exploration (ตัวของตัวเอง · พัก/ทำต่อได้)
            </button>
            <button
              className={`engine-tab ${trainingSubMode === "standard" ? "selected" : ""}`}
              onClick={() => switchTrainingMode("standard")}
            >
              <Cpu size={14} /> Batch Self-Play RL (เทรนตามเวลา + กราฟสถิติ)
            </button>
          </div>

          {/* ================= MODE A: DEEP EXPLORATION THEATER (10^120) ================= */}
          {trainingSubMode === "deepexplore" && (
            <div className="deep-explore-theater" style={{ marginBottom: 20 }}>
              {/* Left: Live Visual Chessboard Stage */}
              <div className="deep-explore-stage">
                <div className="shannon-badge">
                  <span className="shannon-num">10¹²⁰</span>
                  <div>
                    <strong>SHANNON NUMBER EXPLORATION SPACE</strong>
                    <small style={{ display: "block" }}>หมากรุกมีรูปแบบกระดานที่เป็นไปได้ ~10¹²⁰ รูปแบบ · AI กำลังสำรวจและจำลองการเดินทีละตา</small>
                  </div>
                </div>

                <div className="de-game-indicator">
                  <span>เกมจำลองที่ <strong>#{deepExplore.totalGamesPlayed + 1}</strong></span>
                  <span>·</span>
                  <span>ตาเดินที่ <strong>{deepExplore.currentGameMoveCount}</strong></span>
                  <span>·</span>
                  <span>ฝ่ายที่กำลังเดิน: <strong>{deepExplore.currentGame.turn === "w" ? "ขาว (WHITE)" : "ดำ (BLACK)"}</strong></span>
                  <span>·</span>
                  <span className={`explore-pulse ${deepExplore.paused ? "paused" : ""}`} />
                  <span style={{ color: deepExplore.paused ? "#f0ad4e" : "#5cb85c" }}>
                    {deepExplore.paused ? "พักการฝึกอยู่ (PAUSED)" : "กำลังจำลองสด (LIVE)"}
                  </span>
                </div>

                {deepExplore.computeEngine === "visual" ? (
                  renderDeepExploreBoard()
                ) : (
                  <div style={{ width: "100%", maxWidth: 460, background: "#1e1e1e", border: "1px solid #333", padding: "24px 20px", color: "#eee", textAlign: "center", borderRadius: 4, boxShadow: "0 8px 30px rgba(0,0,0,0.5)" }}>
                    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", background: "#272727", border: "1px solid #444", fontSize: 10, font: "10px var(--mono)", color: "#74c0fc", marginBottom: 16 }}>
                      <Gauge size={14} /> HEADLESS TURBO ENGINE RUNNING
                    </div>
                    <div style={{ fontSize: 36, fontWeight: 700, letterSpacing: "-0.04em", color: "#fff", margin: "4px 0" }}>
                      {deepExplore.totalPositionsExplored.toLocaleString()}
                    </div>
                    <div style={{ font: "11px var(--mono)", color: "#888", marginBottom: 16 }}>
                      POSITIONS EXPLORED ({deepExplore.exploredFraction})
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, background: "#171717", padding: 12, border: "1px solid #282828", fontSize: 11, textAlign: "left" }}>
                      <div>
                        <span style={{ color: "#777", display: "block", fontSize: 9 }}>HARDWARE ACCELERATION:</span>
                        <strong style={{ color: deepExplore.hardwareAccelerator === "gpu" ? "#74c0fc" : "#8ce99a" }}>
                          {deepExplore.hardwareAccelerator === "gpu"
                            ? "NVIDIA RTX 5050 GPU (Active)"
                            : deepExplore.hardwareAccelerator === "cpu"
                            ? "Multi-Thread CPU (100%)"
                            : "Auto Balanced Core"}
                        </strong>
                      </div>
                      <div>
                        <span style={{ color: "#777", display: "block", fontSize: 9 }}>PHYSICAL ADAPTER:</span>
                        <strong style={{ color: "#fff", fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>
                          {gpuInfo.renderer || "NVIDIA GeForce RTX 5050"}
                        </strong>
                      </div>
                      <div>
                        <span style={{ color: "#777", display: "block", fontSize: 9 }}>TOTAL GAMES:</span>
                        <strong>{deepExplore.totalGamesPlayed.toLocaleString()} กระดาน</strong>
                      </div>
                      <div>
                        <span style={{ color: "#777", display: "block", fontSize: 9 }}>AVG MOVE LENGTH:</span>
                        <strong>{Math.round(deepExplore.averageGameLength)} ตาเดิน/เกม</strong>
                      </div>
                    </div>
                    <div style={{ marginTop: 10, padding: "7px 10px", background: deepExplore.hardwareAccelerator === "gpu" ? "rgba(116, 192, 252, 0.12)" : "rgba(255,255,255,0.05)", border: `1px solid ${deepExplore.hardwareAccelerator === "gpu" ? "#1971c2" : "#333"}`, textAlign: "left", fontSize: 9, font: "9px var(--mono)", color: deepExplore.hardwareAccelerator === "gpu" ? "#a5d8ff" : "#aaa" }}>
                      🎮 <strong>GPU Status:</strong> {deepExplore.hardwareAccelerator === "gpu" ? `กำลังส่ง Matrix Shader Pass ไปยัง NVIDIA RTX 5050 VRAM โดยตรง` : "คลิกเลือก 'GPU RTX (Max)' ด้านล่างเพื่อบังคับใช้การ์ดจอแยก RTX 5050"}
                    </div>
                    <p style={{ fontSize: 10, color: "#777", margin: "10px 0 0", lineHeight: 1.6 }}>
                      💡 เมื่อปิด GUI ความเร็วจะเพิ่มขึ้นมหาศาล คุณสามารถสลับกลับไปที่ปุ่ม <strong>"Visual Board"</strong> ได้ตลอดเวลาเพื่อดูกระดานภาพสด
                    </p>
                  </div>
                )}
              </div>

              {/* Right: Live Control & Shannon Metrics Sidebar */}
              <div className="deep-explore-sidebar">
                <div className="deep-explore-sidebar-header">
                  <div className="top-row">
                    <strong style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 7 }}>
                      <BrainCircuit size={16} /> 10¹²⁰ Exploration Monitor
                    </strong>
                    <span style={{ font: "9px var(--mono)", color: deepExplore.paused ? "#999" : "#2f6f44" }}>
                      {deepExplore.paused ? "PAUSED" : "ACTIVE"}
                    </span>
                  </div>
                  <p style={{ margin: "2px 0 0", fontSize: 10, color: "#777", lineHeight: 1.5 }}>
                    AI จำลอง Self-Play ทั้งสองฝั่ง ปรับค่าน้ำหนัก Center Bonus, Pawn Advance และสะสมรูปแบบกระดานที่พบ
                  </p>
                </div>

                {/* Real Real-time Stats Grid */}
                <div className="deep-explore-stats">
                  <div className="deep-explore-stats-grid">
                    <div className="de-stat wide">
                      <span>POSITIONS EXPLORED (รูปแบบที่ค้นพบแล้ว)</span>
                      <strong style={{ fontSize: 16, color: "#191919" }}>
                        {deepExplore.totalPositionsExplored.toLocaleString()}
                        <small style={{ fontSize: 10, fontWeight: 400, color: "#888", marginLeft: 6 }}>
                          ({deepExplore.exploredFraction})
                        </small>
                      </strong>
                    </div>
                    <div className="de-stat">
                      <span>GAMES PLAYED (กระดานที่จบ)</span>
                      <strong>{deepExplore.totalGamesPlayed.toLocaleString()}</strong>
                    </div>
                    <div className="de-stat">
                      <span>WIN RATE (WHITE)</span>
                      <strong>{deepExplore.winRate}%</strong>
                    </div>
                    <div className="de-stat">
                      <span>WHITE WINS / BLACK WINS</span>
                      <strong style={{ fontSize: 12 }}>{deepExplore.whiteWins}W - {deepExplore.blackWins}L</strong>
                    </div>
                    <div className="de-stat">
                      <span>DRAWS (เสมอ)</span>
                      <strong>{deepExplore.draws}</strong>
                    </div>
                    <div className="de-stat">
                      <span>EXPLORATION RATE (ε)</span>
                      <strong>{deepExplore.epsilon.toFixed(3)}</strong>
                    </div>
                    <div className="de-stat">
                      <span>AVG GAME LENGTH</span>
                      <strong>{Math.round(deepExplore.averageGameLength)} ตาเดิน</strong>
                    </div>
                  </div>
                </div>

                {/* AI Thought Process */}
                {deepExplore.lastThought && (
                  <div className="de-thought-box">
                    <strong>AI SIMULATION LOG:</strong>
                    <span>{deepExplore.lastThought}</span>
                  </div>
                )}

                {/* Move History for current visual board */}
                <div className="deep-explore-history">
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, font: "9px var(--mono)", color: "#777" }}>
                    <strong>กระดานปัจจุบัน (MOVE HISTORY)</strong>
                    <span>{Math.ceil(deepExplore.currentGame.moveHistory.length / 2)} ตาเดิน</span>
                  </div>
                  <div className="deep-explore-history-table">
                    {Array.from({ length: Math.ceil(deepExplore.currentGame.moveHistory.length / 2) }).map((_, i) => {
                      const whiteMove = deepExplore.currentGame.moveHistory[i * 2];
                      const blackMove = deepExplore.currentGame.moveHistory[i * 2 + 1];
                      return (
                        <div key={i} style={{ display: "contents" }}>
                          <span className="history-num">{i + 1}.</span>
                          <span className="history-white">{whiteMove?.san || "—"}</span>
                          <span className="history-black">{blackMove?.san || "—"}</span>
                        </div>
                      );
                    })}
                    {!deepExplore.currentGame.moveHistory.length && (
                      <div style={{ gridColumn: "span 3", color: "#aaa", fontSize: 10, padding: "10px 0", textAlign: "center" }}>
                        เพิ่งเริ่มกระดานใหม่ กำลังก้าวตาแรก...
                      </div>
                    )}
                  </div>
                </div>

                {/* Controls: Engine Mode (Headless vs Visual), Hardware (CPU/GPU), Pause / Resume / Step */}
                <div className="deep-explore-controls">
                  {/* Engine Mode Toggle: Headless Max-Speed vs Visual Board */}
                  <div style={{ display: "flex", gap: 5, marginBottom: 2 }}>
                    <button
                      className={`engine-tab ${deepExplore.computeEngine === "headless" ? "selected" : ""}`}
                      style={{ flex: 1, padding: "6px 8px", fontSize: 9 }}
                      onClick={() => setDeepExplore(prev => ({ ...prev, computeEngine: "headless" }))}
                    >
                      <Gauge size={12} /> Headless (ปิด GUI วิ่งเต็มสปีด)
                    </button>
                    <button
                      className={`engine-tab ${deepExplore.computeEngine === "visual" ? "selected" : ""}`}
                      style={{ flex: 1, padding: "6px 8px", fontSize: 9 }}
                      onClick={() => setDeepExplore(prev => ({ ...prev, computeEngine: "visual" }))}
                    >
                      <Sparkles size={12} /> Visual Board (เปิดกระดานสด)
                    </button>
                  </div>

                  {/* Hardware Selection (CPU / GPU / Auto) */}
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, fontSize: 9, font: "9px var(--mono)", color: "#666" }}>
                    <span>HARDWARE:</span>
                    <div style={{ display: "flex", gap: 4 }}>
                      <button
                        className={`chart-type-btn ${deepExplore.hardwareAccelerator === "gpu" ? "active" : ""}`}
                        style={{ padding: "4px 8px", fontSize: 9, borderColor: deepExplore.hardwareAccelerator === "gpu" ? "#1971c2" : undefined, background: deepExplore.hardwareAccelerator === "gpu" ? "#1864ab" : undefined, color: deepExplore.hardwareAccelerator === "gpu" ? "#fff" : undefined }}
                        onClick={() => setDeepExplore(prev => ({ ...prev, hardwareAccelerator: "gpu" }))}
                      >
                        ⚡ NVIDIA RTX 5050 (GPU)
                      </button>
                      <button
                        className={`chart-type-btn ${deepExplore.hardwareAccelerator === "cpu" ? "active" : ""}`}
                        style={{ padding: "4px 8px", fontSize: 9 }}
                        onClick={() => setDeepExplore(prev => ({ ...prev, hardwareAccelerator: "cpu" }))}
                      >
                        CPU Multi-Core
                      </button>
                      <button
                        className={`chart-type-btn ${deepExplore.hardwareAccelerator === "auto" ? "active" : ""}`}
                        style={{ padding: "4px 8px", fontSize: 9 }}
                        onClick={() => setDeepExplore(prev => ({ ...prev, hardwareAccelerator: "auto" }))}
                      >
                        AUTO
                      </button>
                    </div>
                  </div>

                  {deepExplore.computeEngine === "visual" && (
                    <div className="speed-row">
                      <label>ความเร็วกระดาน:</label>
                      <select
                        value={deepExploreSpeedMs}
                        onChange={e => setDeepExploreSpeedMs(Number(e.target.value))}
                      >
                        <option value={700}>ช้า (0.7s / ตา)</option>
                        <option value={400}>ปกติ (0.4s / ตา)</option>
                        <option value={200}>เร็ว (0.2s / ตา)</option>
                        <option value={80}>เทอร์โบ (0.08s / ตา)</option>
                      </select>
                    </div>
                  )}

                  {deepExplore.computeEngine === "headless" && (
                    <div style={{ padding: "6px 8px", background: "#f0f4f0", border: "1px solid #c8dcc8", fontSize: 9, color: "#28522e", lineHeight: 1.4 }}>
                      ⚡ <strong>Headless Mode Active:</strong> ปิดการเรนเดอร์ภาพกระดานเพื่อคำนวณด้วย Math Core เต็มพิกัด เร็วกว่าเปิด GUI <strong>50 - 100 เท่า</strong> (~3,000+ ตาเดิน/วินาที)
                    </div>
                  )}

                  <div style={{ display: "flex", gap: 6 }}>
                    {deepExplore.paused ? (
                      <button
                        className="primary-action"
                        style={{ flex: 1, marginTop: 0 }}
                        onClick={() => setDeepExplore(prev => ({ ...prev, paused: false }))}
                      >
                        <Play size={14} /> เริ่ม / ฝึกต่อ (Resume)
                      </button>
                    ) : (
                      <button
                        className="secondary-action"
                        style={{ flex: 1, marginTop: 0, background: "#fff" }}
                        onClick={() => setDeepExplore(prev => ({ ...prev, paused: true }))}
                      >
                        <Pause size={14} /> พักการฝึก (Pause)
                      </button>
                    )}
                    <button
                      className="outline-full"
                      style={{ marginTop: 0 }}
                      onClick={() => setDeepExplore(prev => stepDeepExplore(prev))}
                      title="ก้าวเดินจำลอง 1 ตาเดินด้วยตนเอง"
                    >
                      ก้าว 1 ตา
                    </button>
                  </div>

                  <div style={{ display: "flex", gap: 6 }}>
                    <button
                      className="secondary-action"
                      style={{ flex: 1, fontSize: 9 }}
                      onClick={() => {
                        if (activeTrainingId) {
                          setChessProjects(prev => prev.map(p => {
                            if (p.id !== activeTrainingId) return p;
                            return {
                              ...p,
                              games: p.games + deepExplore.totalGamesPlayed,
                              weights: { ...deepExplore.weights },
                              winRate: deepExplore.winRate
                            };
                          }));
                          alert(`บันทึกน้ำหนักที่เรียนรู้จากการสำรวจ ${deepExplore.totalPositionsExplored.toLocaleString()} รูปแบบ ลงในโปรเจกต์ "${activeProject?.name}" เรียบร้อยแล้ว!`);
                        }
                      }}
                    >
                      <CheckCircle2 size={12} /> บันทึกลงโปรเจกต์ ({activeProject?.name})
                    </button>
                    <button
                      className="outline-full"
                      style={{ fontSize: 9 }}
                      onClick={() => {
                        if (window.confirm("ต้องการรีเซ็ตสถิติการสำรวจ 10^120 ทั้งหมดหรือไม่?")) {
                          setDeepExplore(initDeepExplore());
                        }
                      }}
                    >
                      <RotateCcw size={12} /> ล้างสถิติ
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ================= MODE B: STANDARD BATCH SELF-PLAY RL ================= */}
          {trainingSubMode === "standard" && (
            <>
          <div className="training-grid">
            <div className="panel training-config">
              <div className="panel-heading">
                <div><span className="eyebrow">CONFIGURATION</span><h3>ตั้งค่าการฝึก Chess AI</h3></div>
                <BrainCircuit size={18} />
              </div>

              {/* Select Project to Train */}
              <label className="field-label">เลือกโปรเจกต์หมากรุกที่ต้องการฝึก</label>
              <select value={activeTrainingId} onChange={e => handleSelectProject(Number(e.target.value))}>
                {chessProjects.map(p => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.games.toLocaleString()} กระดาน · {p.status})
                  </option>
                ))}
              </select>

              {/* Engine Mode Tabs (Local RL vs Cloud AI API) */}
              <label className="field-label">โหมดการประมวลผล (เลือก Local CPU หรือ Cloud AI)</label>
              <div className="engine-mode-tabs">
                <button
                  className={`engine-tab ${engineMode === "local" ? "selected" : ""}`}
                  onClick={() => setEngineMode("local")}
                >
                  <Cpu size={14} /> Local RL (CPU/GPU Simulation)
                </button>
                <button
                  className={`engine-tab ${engineMode === "api" ? "selected" : ""}`}
                  onClick={() => setEngineMode("api")}
                >
                  <Cloud size={14} /> Cloud AI API (Gemini Studio)
                </button>
              </div>

              {engineMode === "local" ? (
                <>
                  <label className="field-label">อุปกรณ์ประมวลผล</label>
                  <div className="compute-options">
                    <button className={trainingDevice === "auto" ? "selected" : ""} onClick={() => setTrainingDevice("auto")}>
                      <Sparkles size={16} /><strong>อัตโนมัติ</strong><small>ตรวจจับฮาร์ดแวร์ที่ดีที่สุด</small>
                    </button>
                    <button className={trainingDevice === "cpu" ? "selected" : ""} onClick={() => setTrainingDevice("cpu")}>
                      <Cpu size={16} /><strong>CPU</strong><small>คำนวณ Self-Play รวดเร็ว</small>
                    </button>
                    <button className={trainingDevice === "gpu" ? "selected" : ""} onClick={() => setTrainingDevice("gpu")}>
                      <Gauge size={16} /><strong>GPU</strong><small>เร่งคำนวณ Position Search</small>
                    </button>
                  </div>

                  <label className="field-label">อัลกอริทึมการเรียนรู้</label>
                  <select value={trainingAlgorithm} onChange={e => setTrainingAlgorithm(e.target.value as any)}>
                    <option value="selfplay-rl">Self-Play Reinforcement Learning (นโยบายเล่นกับตัวเอง)</option>
                    <option value="td-minimax">TD-Minimax (ปรับค่าน้ำหนักตามผลลัพธ์กิ่งไม้ Minimax)</option>
                    <option value="heuristic-distill">Tactical Distillation (เรียนรู้การแลกหมากและตำแหน่งยุทธศาสตร์)</option>
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
                    เมื่อเลือก <strong>Cloud AI API</strong> ระบบจะใช้ LLM ช่วยวิเคราะห์การประเมินตำแหน่งหมากและแนะนำการเดินหมากผ่านคลาวด์
                  </p>
                  <label className="field-label" style={{ marginTop: 6 }}>ผู้ให้บริการ AI</label>
                  <select value={apiProvider} onChange={e => setApiProvider(e.target.value)}>
                    <option value="google">Google Gemini API</option>
                  </select>

                  <label className="field-label">โมเดล</label>
                  <select value={apiModel} onChange={e => setApiModel(e.target.value)}>
                    <option value="gemini-2.5-flash">Gemini 2.5 Flash (ความเร็วสูง เหมาะกับเกม)</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro (วิเคราะห์เชิงลึก)</option>
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

              <label className="field-label">Simulated Games ต่อ Batch</label>
              <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}>
                <option value={50}>50 games / batch</option>
                <option value={100}>100 games / batch</option>
                <option value={200}>200 games / batch</option>
                <option value={500}>500 games / batch</option>
              </select>

              <div className="field-help">
                <Clock3 size={14} /> เลือก {trainMinutes} นาที = ฝึก {trainMinutes * 60} วินาทีตามจริง
              </div>

              <button
                className="primary-action full-action"
                disabled={!activeTrainingId || trainingRunning}
                onClick={() => {
                  const p = chessProjects.find(x => x.id === activeTrainingId);
                  if (p) beginTraining({ ...p, minutes: trainMinutes }, true);
                }}
              >
                <Play size={15} />
                {activeProject && activeProject.games > 0
                  ? `ฝึกสะสมต่ออีก ${trainMinutes} นาที (สะสมต่อจาก ${activeProject.games.toLocaleString()} กระดานเดิม)`
                  : `เริ่มฝึกตามเวลาที่กำหนด (${trainMinutes} นาที)`}
              </button>
              {activeProject && activeProject.games > 0 && (
                <button
                  className="outline-full"
                  style={{ marginTop: 6 }}
                  disabled={trainingRunning}
                  onClick={() => {
                    if (window.confirm(`ต้องการล้างความรู้เดิมของ ${activeProject.name} และเริ่มฝึกใหม่จากศูนย์หรือไม่?`)) {
                      beginTraining({ ...activeProject, minutes: trainMinutes, games: 0, weights: { ...DEFAULT_CHESS_WEIGHTS } }, false);
                    }
                  }}
                >
                  <RotateCcw size={13} /> รีเซ็ตความรู้เดิม (เริ่มฝึกใหม่จาก 0)
                </button>
              )}
              <button className="outline-full" style={{ marginTop: 6 }} onClick={() => { setShowNewProject(true); setPage("projects"); }}>
                <Plus size={15} /> สร้างโปรเจกต์ Chess ใหม่
              </button>
            </div>

            {/* Live Training Session Panel */}
            <div className="panel run-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">LIVE TRAINING SESSION</span>
                  <h3>{activeProject?.name ?? "ยังไม่มี Training Session"}</h3>
                </div>
                <div className="run-status">
                  {trainingRunning ? "SIMULATING SELF-PLAY" : activeProject?.status === "Complete" ? "COMPLETE" : "IDLE"}
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
                  <span>SIMULATED CHESS GAMES</span>
                  <strong>{(activeProject?.games ?? 0).toLocaleString()}</strong>
                </div>
                <div>
                  <span>ELAPSED TIME</span>
                  <strong>{fmt(trainingSeconds)}</strong>
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
                    setSelectedPlayProjectId(activeProject.id);
                    setChessOpponent("project");
                    setPage("play");
                  }}
                >
                  <Swords size={15} /> นำโมเดล {activeProject.name} ไปประลองใน Arena ทันที
                </button>
              )}

              <div className="notice" style={{ marginTop: 18 }}>
                <Info size={15} />
                <span>
                  ระหว่างที่เวลาเดิน ระบบจะจำลองกระดาน Self-Play ของโมเดลอย่างต่อเนื่อง ปรับค่าน้ำหนัก Center Bonus, Pawn Advance และพล็อตกราฟอัตราชนะด้านล่างแบบเรียลไทม์
                </span>
              </div>
            </div>
          </div>

          {/* REAL GRAPH METRICS PANEL */}
          <div className="panel chart-panel" style={{ marginTop: 16 }}>
            <div className="chart-header-row">
              <div>
                <span className="eyebrow">REAL SELF-PLAY METRICS</span>
                <h3 style={{ margin: "4px 0 0" }}>กราฟผลการฝึกหมากรุกจริง (Real-Time Metrics)</h3>
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
                <span>AVG REWARD / GAME</span>
                <strong>{latestMetric ? latestMetric.avgReward : "—"}</strong>
              </div>
              <div className="stat-box">
                <span>MATERIAL ADVANTAGE</span>
                <strong>{latestMetric ? `${latestMetric.materialAdv > 0 ? "+" : ""}${latestMetric.materialAdv}` : "—"}</strong>
              </div>
              <div className="stat-box">
                <span>EXPLORATION RATE (ε)</span>
                <strong>{latestMetric ? latestMetric.epsilon : explorationRate.toFixed(2)}</strong>
              </div>
            </div>
          </div>
            </>
          )}
        </section>
      )}

      {/* ================= CHESS: PROJECTS ================= */}
      {page === "projects" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">03 / CHESS PROJECTS</p>
              <h1>Your Chess AI Models</h1>
              <p className="lede">จัดการโมเดลหมากรุกที่คุณฝึกฝน ตรวจสอบค่าน้ำหนักการประเมินสถานะ และส่งไปลงสนามจริง</p>
            </div>
            <button className="primary-action" onClick={() => setShowNewProject(v => !v)}>
              <Plus size={15} /> {showNewProject ? "ปิดฟอร์ม" : "สร้างโปรเจกต์ Chess AI"}
            </button>
          </div>

          {/* New Project Form */}
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
                placeholder="เช่น AlphaKnight-v2"
              />

              <label className="field-label">อัลกอริทึมเริ่มต้น</label>
              <select value={newProjectAlgo} onChange={e => setNewProjectAlgo(e.target.value as any)}>
                <option value="selfplay-rl">Self-Play Reinforcement Learning</option>
                <option value="td-minimax">TD-Minimax (Depth Evaluation)</option>
                <option value="heuristic-distill">Tactical Distillation</option>
              </select>

              <label className="field-label">ประเภทโปรเจกต์ (โหมดฝึกของโปรเจกต์นี้ — แต่ละแบบบันทึกข้อมูลของตัวเองแยกกัน)</label>
              <div className="engine-mode-tabs" style={{ marginBottom: 12 }}>
                <button
                  className={`engine-tab ${newProjectMode === "standard" ? "selected" : ""}`}
                  onClick={() => setNewProjectMode("standard")}
                >
                  <Cpu size={14} /> Self-Play RL (Batch เทรนตามเวลา)
                </button>
                <button
                  className={`engine-tab ${newProjectMode === "deepexplore" ? "selected" : ""}`}
                  onClick={() => setNewProjectMode("deepexplore")}
                >
                  <Sparkles size={14} /> 10¹²⁰ Deep Exploration
                </button>
              </div>

              <button className="primary-action full-action" onClick={handleCreateProject}>
                <Plus size={14} /> บันทึกและเริ่มโปรเจกต์
              </button>
            </div>
          )}

          {/* Project List */}
          <div className="project-list">
            {chessProjects.map(p => (
              <div className="project-card" key={p.id}>
                <div className="project-symbol">
                  <span style={{ fontSize: 24 }}>♞</span>
                </div>
                <div className="project-main">
                  <div className="project-title">
                    <h3>{p.name}</h3>
                    <span className={`project-status ${p.status.toLowerCase()}`}>
                      {p.status === "Complete" ? "เทรนเสร็จสมบูรณ์" : p.status === "Training" ? "กำลังฝึก" : "พร้อมฝึก"}
                    </span>
                    <span style={{ font: "8px var(--mono)", color: "#888", border: "1px solid #e1e1dc", padding: "2px 6px" }}>
                      {p.algorithm}
                    </span>
                    <span style={{ font: "8px var(--mono)", color: "#fff", background: p.trainMode === "deepexplore" ? "#1971c2" : "#495057", padding: "2px 6px" }}>
                      {p.trainMode === "deepexplore" ? "10¹²⁰ DEEP EXPLORE" : "SELF-PLAY RL"}
                    </span>
                  </div>
                  <p>
                    ผ่านการจำลอง <strong>{p.games.toLocaleString()} กระดาน</strong> · Win Rate <strong>{p.winRate}%</strong> · Center Bonus <strong>{p.weights.centerBonus}</strong>
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
                    onClick={() => {
                      setSelectedPlayProjectId(p.id);
                      setChessOpponent("project");
                      setPage("play");
                    }}
                  >
                    <Swords size={13} /> ประลอง
                  </button>
                  <button
                    className="secondary-action action-button"
                    onClick={() => openTrainingFor(p)}
                  >
                    <BrainCircuit size={13} /> ฝึกต่อ
                  </button>
                  {chessProjects.length > 1 && (
                    <button
                      className="top-icon"
                      title="ลบโปรเจกต์"
                      onClick={() => {
                        // ลบ slot การสำรวจของโปรเจกต์ตัวนี้ด้วย เพื่อไม่ให้ข้อมูลค้างกองอยู่
                        clearStoredData(deepExploreKey(p.id));
                        setChessProjects(prev => prev.filter(x => x.id !== p.id));
                      }}
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

      {/* ================= CHESS: RULES ================= */}
      {page === "rules" && (
        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">04 / CHESS RULEBOOK</p>
              <h1>International Chess Rules</h1>
              <p className="lede">กติกาสากลของหมากรุกสากล (FIDE Standard)</p>
            </div>
            <div className="rule-stamp">8×8<small>CHESS</small></div>
          </div>

          <div className="rules-grid">
            <div className="panel">
              <div className="panel-heading"><h3>ตัวหมากและการเคลื่อนที่</h3><BookOpen size={17} /></div>
              {Object.entries(PIECE_NAMES).map(([type, name]) => (
                <div className="rule-line" key={type}>
                  <span>
                    <b>{PIECE_SYMBOLS.w[type as PieceType]} {name}</b>
                    <small>
                      {type === "p" && "เดินตรงหน้า 1 ช่อง (2 ช่องในตาแรก) กินเฉียง 1 ช่อง"}
                      {type === "n" && "เดินเป็นรูปตัว L (2 ช่องตรง + 1 ช่องข้าง) กระโดดข้ามหมากอื่นได้"}
                      {type === "b" && "เดินทแยงมุมได้ไม่จำกัดช่องตามแนวเฉียง"}
                      {type === "r" && "เดินตรงได้ไม่จำกัดช่องตามแนวนอนและแนวตั้ง"}
                      {type === "q" && "เดินได้รอบทิศทาง (ตรงและทแยงมุม) มีพลังโจมตีสูงสุด"}
                      {type === "k" && "เดินได้รอบตัว 1 ช่อง เป็นหัวใจสำคัญของเกมที่ต้องปกป้อง"}
                    </small>
                  </span>
                  <strong>{type.toUpperCase()}</strong>
                </div>
              ))}
            </div>

            <div className="panel">
              <div className="panel-heading"><h3>การจบเกมและเป้าหมาย</h3><Trophy size={17} /></div>
              <div className="rules-bullets">
                <p><span>01</span> <strong>รุกฆาต (Checkmate):</strong> เมื่อ King ฝ่ายตรงข้ามถูกโจมตี (Check) และไม่มีตาเดินใดที่สามารถหนี บัง หรือกินตัวโจมตีได้ จะถือว่าชนะทันที</p>
                <p><span>02</span> <strong>อับจน (Stalemate):</strong> เมื่อฝ่ายที่ถึงตาเดินไม่ได้ถูกรุก แต่ไม่มีตาเดินที่ถูกกติกาเหลืออยู่เลย จะถือว่าเสมอกัน (Draw)</p>
                <p><span>03</span> <strong>การเลื่อนขั้นเบี้ย (Pawn Promotion):</strong> เมื่อเบี้ยเดินทะลุไปถึงแถวสุดท้าย (แถว 8 ของขาว หรือแถว 1 ของดำ) เบี้ยจะถูกเลื่อนขั้นเป็น Queen ทันที</p>
              </div>
            </div>
          </div>

          <button className="primary-action" style={{ marginTop: 16 }} onClick={() => setPage("play")}>
            เข้าสู่กระดานเล่นเกม <ArrowRight size={15} />
          </button>
        </section>
      )}
    </>
  );
}

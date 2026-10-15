/**
 * Persistence helper for saving AI training projects (with trained Q-Tables /
 * Chess model weights) in game-specific localStorage buckets so they survive
 * page reloads and dev-server restarts. Each game's key is its logical
 * training-data folder.
 *
 * Data is wrapped in a versioned envelope so future schema migrations can
 * detect and convert (or discard) old payloads safely.
 */

const ENVELOPE_VERSION = 1;
const PREFIX = "arena-hub";

export const STORAGE_KEYS = {
  // Keep each game's training projects in an isolated logical folder.
  blackjackProjects: "projects:blackjack",
  chessProjects: "projects:chess",
  tttProjects: "projects:ttt"
} as const;

/** Per-project Deep Exploration session slot (each project owns its own save). */
export function deepExploreKey(projectId: number): string {
  return `deepexplore:${projectId}`;
}

type Envelope = {
  schema: number;
  savedAt: string;
  data: unknown;
};

function storageKey(key: string): string {
  return `${PREFIX}:${key}`;
}

/** Removes a stored value (ignores missing entries and storage errors). */
export function clearStoredData(key: string): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    window.localStorage.removeItem(storageKey(key));
  } catch (error) {
    console.warn(`[persistence] failed to clear "${key}":`, error);
  }
}

/**
 * Loads a previously stored value. Returns null when nothing is stored,
 * the payload is from an older schema, is corrupted JSON, or fails
 * `validate` — in every such case the caller should fall back to in-app
 * defaults instead of crashing.
 */
export function loadStoredData<T>(
  key: string,
  validate: (value: unknown) => boolean
): T | null {
  if (typeof window === "undefined" || !window.localStorage) return null;
  try {
    const raw = window.localStorage.getItem(storageKey(key));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Envelope;
    if (parsed.schema !== ENVELOPE_VERSION) return null;
    if (typeof parsed.savedAt !== "string") return null;
    if (!validate(parsed.data)) return null;
    return parsed.data as T;
  } catch (error) {
    console.warn(`[persistence] failed to load "${key}":`, error);
    return null;
  }
}

/** Saves a value in the versioned envelope. No-ops on failure (e.g. quota). */
export function saveStoredData<T>(key: string, data: T): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    const envelope: Envelope = {
      schema: ENVELOPE_VERSION,
      savedAt: new Date().toISOString(),
      data
    };
    window.localStorage.setItem(storageKey(key), JSON.stringify(envelope));
  } catch (error) {
    console.warn(`[persistence] failed to save "${key}":`, error);
  }
}

/**
 * Shared shape check for training projects: every entry must at least be an
 * object with a numeric `id` and string `name`. Specific fields (qTable,
 * chess weights, etc.) are validated by each caller for its own type.
 */
export function isProjectRecordArray(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      entry =>
        typeof entry === "object" &&
        entry !== null &&
        typeof (entry as { id?: unknown }).id === "number" &&
        typeof (entry as { name?: unknown }).name === "string"
    )
  );
}

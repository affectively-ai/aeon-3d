/**
 * voidmap-live.ts
 *
 * Bridges live inference rejection events from Aether (distributed inference)
 * and Glossolalia (semiotic ensemble) into VoidMap tombstones. This replaces
 * the Netflix demo data with real rejection history from production inference.
 *
 * Coordinate mapping for inference topology:
 *   X = layerIndex / totalLayers      (network depth)
 *   Y = hiddenStateMagnitude          (activation scale at failure)
 *   Z = tokenPosition / sequenceLen   (temporal position in generation)
 *
 * The resulting point cloud is a 3D map of WHERE inference fails:
 *   - Wall at high-X: deep layers are unreliable
 *   - Floor at low-Y: hidden states collapsing to zero
 *   - Cluster at high-Z: long sequences degrade
 */

import type {
  Vec3,
  TopologyBranch,
  TopologyFrameInput,
  TopologyFrameResult,
} from './index.js';
import type { VoidMap, VoidMapOptions, Tombstone } from './voidmap.js';
import { createVoidMap, recordFrame } from './voidmap.js';

// ---------------------------------------------------------------------------
// Inference event types
// ---------------------------------------------------------------------------

export type InferenceFailureReason =
  | 'layer-timeout'
  | 'hidden-state-explosion'
  | 'stability-clamp'
  | 'depth-diverse-exit'
  | 'oom'
  | 'rate-limited'
  | 'node-unreachable';

export interface InferenceEvent {
  /** Unique ID for this inference request */
  requestId: string;
  /** Cloud Run service name (e.g., 'inference-llama70b-layer-03') */
  serviceName: string;
  /** Layer index in the distributed stack (0-indexed) */
  layerIndex: number;
  /** Total layers in the model */
  totalLayers: number;
  /** Hidden state L2 norm at point of failure */
  hiddenStateMagnitude: number;
  /** Token position being generated when failure occurred */
  tokenPosition: number;
  /** Total sequence length so far */
  sequenceLength: number;
  /** Why this layer failed */
  reason: InferenceFailureReason;
  /** Processing time before failure (ms) */
  elapsedMs: number;
  /** Budget per layer (ms) */
  budgetMs: number;
  /** Model identifier */
  modelName: string;
  /** ISO timestamp */
  timestamp: string;
}

export type GlossolaliaStrategy =
  | 'temperature'
  | 'nucleus'
  | 'top-k'
  | 'vickrey-auction'
  | 'greedy'
  | 'contrastive';

export interface GlossolaliaEvent {
  /** Unique ID for this generation step */
  stepId: string;
  /** Which strategy was used */
  strategy: GlossolaliaStrategy;
  /** Token position in the sequence */
  tokenPosition: number;
  /** Sequence length */
  sequenceLength: number;
  /** Log probability of the chosen token under this strategy */
  logProbability: number;
  /** Whether this strategy won or lost the race */
  won: boolean;
  /** The winning strategy (if this one lost) */
  winnerId?: string;
  /** Temperature parameter (for temperature/nucleus) */
  temperature?: number;
  /** Top-k value (for top-k) */
  topK?: number;
  /** Nucleus probability mass (for nucleus) */
  nucleusP?: number;
  /** Vickrey auction bid (for vickrey) */
  auctionBid?: number;
  /** ISO timestamp */
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Coordinate mappers
// ---------------------------------------------------------------------------

/**
 * Map an inference event to a 3D position.
 * X = network depth, Y = activation scale, Z = sequence position.
 */
export function inferenceEventToPosition(event: InferenceEvent): Vec3 {
  return {
    x: event.totalLayers > 0 ? event.layerIndex / event.totalLayers : 0,
    y: Math.min(2, Math.log2(Math.max(1, event.hiddenStateMagnitude)) / 10),
    z: event.sequenceLength > 0 ? event.tokenPosition / event.sequenceLength : 0,
  };
}

/**
 * Map a Glossolalia event to a 3D position.
 * X = strategy index (deterministic ordering), Y = log probability, Z = sequence position.
 */
export function glossolaliaEventToPosition(event: GlossolaliaEvent): Vec3 {
  const strategyOrder: Record<GlossolaliaStrategy, number> = {
    'greedy': 0,
    'temperature': 0.2,
    'nucleus': 0.4,
    'top-k': 0.6,
    'contrastive': 0.8,
    'vickrey-auction': 1.0,
  };

  return {
    x: strategyOrder[event.strategy] ?? 0.5,
    y: Math.max(0, Math.min(1, (event.logProbability + 10) / 10)),
    z: event.sequenceLength > 0 ? event.tokenPosition / event.sequenceLength : 0,
  };
}

// ---------------------------------------------------------------------------
// Event to TopologyFrame converters
// ---------------------------------------------------------------------------

function inferenceReasonToTombstoneReason(
  reason: InferenceFailureReason
): Tombstone['reason'] {
  switch (reason) {
    case 'layer-timeout':
    case 'oom':
    case 'rate-limited':
    case 'node-unreachable':
      return 'over-budget';
    case 'stability-clamp':
    case 'hidden-state-explosion':
      return 'explicit-vent';
    case 'depth-diverse-exit':
      return 'race-loser';
    default:
      return 'explicit-vent';
  }
}

/**
 * Convert an inference event to a TopologyFrameInput/Result pair
 * suitable for recordFrame.
 */
export function inferenceEventToFrame(event: InferenceEvent): {
  input: TopologyFrameInput;
  result: TopologyFrameResult;
} {
  const branchId = `${event.modelName}:layer-${event.layerIndex}`;

  const branch: TopologyBranch = {
    id: branchId,
    vertices: Math.round(event.hiddenStateMagnitude * 10),
    drawCalls: 1,
    estimatedCostMs: event.elapsedMs,
    quality: event.budgetMs > 0
      ? Math.max(0, 1 - event.elapsedMs / event.budgetMs)
      : 0,
    vent: true,
  };

  const input: TopologyFrameInput = {
    branches: [branch],
    strategy: 'race',
    budgetMs: event.budgetMs,
  };

  const result: TopologyFrameResult = {
    collapsedBy: 'race',
    winnerId: null,
    survivingBranches: [],
    ventedBranchIds: [branchId],
    totalVertices: 0,
    totalDrawCalls: 0,
    totalCostMs: event.elapsedMs,
    budgetMs: event.budgetMs,
    overBudget: event.elapsedMs > event.budgetMs,
  };

  return { input, result };
}

/**
 * Convert a batch of Glossolalia events from a single generation step
 * into a TopologyFrameInput/Result pair. The winner survives, losers are vented.
 */
export function glossolaliaEventsToFrame(events: readonly GlossolaliaEvent[]): {
  input: TopologyFrameInput;
  result: TopologyFrameResult;
} | null {
  if (events.length === 0) return null;

  const branches: TopologyBranch[] = [];
  const ventedIds: string[] = [];
  let winnerId: string | null = null;
  let winnerBranch: TopologyBranch | null = null;

  for (const event of events) {
    const branchId = `glossolalia:${event.strategy}`;
    const branch: TopologyBranch = {
      id: branchId,
      vertices: Math.round(Math.abs(event.logProbability) * 100),
      drawCalls: 1,
      estimatedCostMs: Math.max(0.1, -event.logProbability),
      quality: Math.max(0, Math.min(1, (event.logProbability + 5) / 5)),
      vent: !event.won,
    };
    branches.push(branch);

    if (event.won) {
      winnerId = branchId;
      winnerBranch = branch;
    } else {
      ventedIds.push(branchId);
    }
  }

  const input: TopologyFrameInput = {
    branches,
    strategy: 'race',
    budgetMs: 10,
  };

  const result: TopologyFrameResult = {
    collapsedBy: 'race',
    winnerId,
    survivingBranches: winnerBranch ? [winnerBranch] : [],
    ventedBranchIds: ventedIds,
    totalVertices: winnerBranch?.vertices ?? 0,
    totalDrawCalls: winnerBranch?.drawCalls ?? 0,
    totalCostMs: winnerBranch?.estimatedCostMs ?? 0,
    budgetMs: 10,
    overBudget: false,
  };

  return { input, result };
}

// ---------------------------------------------------------------------------
// Live VoidMap accumulator
// ---------------------------------------------------------------------------

export interface LiveVoidMapOptions extends VoidMapOptions {
  /** Custom inference event position mapper */
  inferenceMapper?: (event: InferenceEvent) => Vec3;
  /** Custom Glossolalia event position mapper */
  glossolaliaMapper?: (event: GlossolaliaEvent) => Vec3;
}

export interface LiveVoidMap {
  map: VoidMap;
  /** Total inference events processed */
  inferenceEventCount: number;
  /** Total Glossolalia events processed */
  glossolaliaEventCount: number;
  /** Per-model tombstone counts */
  perModelCounts: ReadonlyMap<string, number>;
  /** Per-strategy win/loss counts */
  strategyStats: ReadonlyMap<string, { wins: number; losses: number }>;
}

/**
 * Create a fresh live VoidMap accumulator.
 */
export function createLiveVoidMap(options?: LiveVoidMapOptions): LiveVoidMap {
  return {
    map: createVoidMap(options),
    inferenceEventCount: 0,
    glossolaliaEventCount: 0,
    perModelCounts: new Map(),
    strategyStats: new Map(),
  };
}

/**
 * Ingest an inference failure event into the live VoidMap.
 */
export function ingestInferenceEvent(
  live: LiveVoidMap,
  event: InferenceEvent,
  options?: LiveVoidMapOptions
): LiveVoidMap {
  const mapper = options?.inferenceMapper ?? inferenceEventToPosition;
  const position = mapper(event);

  const frame = inferenceEventToFrame(event);
  const customMapper = () => position;

  const newMap = recordFrame(live.map, frame.input, frame.result, {
    ...options,
    customMapper,
  });

  const modelCounts = new Map(live.perModelCounts);
  modelCounts.set(event.modelName, (modelCounts.get(event.modelName) ?? 0) + 1);

  return {
    map: newMap,
    inferenceEventCount: live.inferenceEventCount + 1,
    glossolaliaEventCount: live.glossolaliaEventCount,
    perModelCounts: modelCounts,
    strategyStats: live.strategyStats,
  };
}

/**
 * Ingest a batch of Glossolalia events (one generation step) into the live VoidMap.
 */
export function ingestGlossolaliaStep(
  live: LiveVoidMap,
  events: readonly GlossolaliaEvent[],
  options?: LiveVoidMapOptions
): LiveVoidMap {
  const frame = glossolaliaEventsToFrame(events);
  if (!frame) return live;

  const mapper = options?.glossolaliaMapper ?? glossolaliaEventToPosition;

  // Create a position lookup for the custom mapper
  const positionMap = new Map<string, Vec3>();
  for (const event of events) {
    positionMap.set(`glossolalia:${event.strategy}`, mapper(event));
  }

  const customMapper = (branch: TopologyBranch) =>
    positionMap.get(branch.id) ?? { x: 0.5, y: 0.5, z: 0.5 };

  const newMap = recordFrame(live.map, frame.input, frame.result, {
    ...options,
    customMapper,
  });

  const strategyStats = new Map(live.strategyStats);
  for (const event of events) {
    const existing = strategyStats.get(event.strategy) ?? { wins: 0, losses: 0 };
    if (event.won) {
      strategyStats.set(event.strategy, { wins: existing.wins + 1, losses: existing.losses });
    } else {
      strategyStats.set(event.strategy, { wins: existing.wins, losses: existing.losses + 1 });
    }
  }

  return {
    map: newMap,
    inferenceEventCount: live.inferenceEventCount,
    glossolaliaEventCount: live.glossolaliaEventCount + events.length,
    perModelCounts: live.perModelCounts,
    strategyStats,
  };
}

// ---------------------------------------------------------------------------
// DashRelay subscription adapter
// ---------------------------------------------------------------------------

export interface DashRelayMessage {
  room: string;
  type: string;
  payload: Record<string, unknown>;
  timestamp: string;
}

/**
 * Parse a DashRelay message from the aether-logs room into an InferenceEvent.
 * Returns null if the message is not a recognized inference failure event.
 */
export function parseDashRelayInferenceEvent(
  message: DashRelayMessage
): InferenceEvent | null {
  if (message.room !== 'aether-logs') return null;

  const p = message.payload;
  if (
    typeof p['layerIndex'] !== 'number' ||
    typeof p['totalLayers'] !== 'number' ||
    typeof p['reason'] !== 'string'
  ) {
    return null;
  }

  const validReasons: Set<string> = new Set([
    'layer-timeout', 'hidden-state-explosion', 'stability-clamp',
    'depth-diverse-exit', 'oom', 'rate-limited', 'node-unreachable',
  ]);

  if (!validReasons.has(p['reason'] as string)) return null;

  return {
    requestId: typeof p['requestId'] === 'string' ? p['requestId'] : `req-${Date.now()}`,
    serviceName: typeof p['serviceName'] === 'string' ? p['serviceName'] : 'unknown',
    layerIndex: p['layerIndex'] as number,
    totalLayers: p['totalLayers'] as number,
    hiddenStateMagnitude: typeof p['hiddenStateMagnitude'] === 'number' ? p['hiddenStateMagnitude'] : 1,
    tokenPosition: typeof p['tokenPosition'] === 'number' ? p['tokenPosition'] : 0,
    sequenceLength: typeof p['sequenceLength'] === 'number' ? p['sequenceLength'] : 1,
    reason: p['reason'] as InferenceFailureReason,
    elapsedMs: typeof p['elapsedMs'] === 'number' ? p['elapsedMs'] : 0,
    budgetMs: typeof p['budgetMs'] === 'number' ? p['budgetMs'] : 13000,
    modelName: typeof p['modelName'] === 'string' ? p['modelName'] : 'unknown',
    timestamp: message.timestamp,
  };
}

/**
 * Parse a DashRelay message into Glossolalia events.
 * Returns null if the message is not a recognized Glossolalia event.
 */
export function parseDashRelayGlossolaliaEvents(
  message: DashRelayMessage
): readonly GlossolaliaEvent[] | null {
  if (message.room !== 'aether-logs') return null;
  if (message.type !== 'glossolalia-race') return null;

  const p = message.payload;
  if (!Array.isArray(p['candidates'])) return null;

  const events: GlossolaliaEvent[] = [];
  const candidates = p['candidates'] as Array<Record<string, unknown>>;

  for (const candidate of candidates) {
    if (typeof candidate['strategy'] !== 'string') continue;

    const validStrategies: Set<string> = new Set([
      'temperature', 'nucleus', 'top-k', 'vickrey-auction', 'greedy', 'contrastive',
    ]);
    if (!validStrategies.has(candidate['strategy'])) continue;

    events.push({
      stepId: typeof candidate['stepId'] === 'string' ? candidate['stepId'] : `step-${Date.now()}`,
      strategy: candidate['strategy'] as GlossolaliaStrategy,
      tokenPosition: typeof candidate['tokenPosition'] === 'number' ? candidate['tokenPosition'] : 0,
      sequenceLength: typeof candidate['sequenceLength'] === 'number' ? candidate['sequenceLength'] : 1,
      logProbability: typeof candidate['logProbability'] === 'number' ? candidate['logProbability'] : -5,
      won: candidate['won'] === true,
      winnerId: typeof candidate['winnerId'] === 'string' ? candidate['winnerId'] : undefined,
      temperature: typeof candidate['temperature'] === 'number' ? candidate['temperature'] : undefined,
      topK: typeof candidate['topK'] === 'number' ? candidate['topK'] : undefined,
      nucleusP: typeof candidate['nucleusP'] === 'number' ? candidate['nucleusP'] : undefined,
      auctionBid: typeof candidate['auctionBid'] === 'number' ? candidate['auctionBid'] : undefined,
      timestamp: message.timestamp,
    });
  }

  return events.length > 0 ? events : null;
}

// ---------------------------------------------------------------------------
// Heartbeat detector: 12-minute oscillation from Cloud Scheduler warmup
// ---------------------------------------------------------------------------

export interface HeartbeatState {
  /** Timestamps of detected heartbeats */
  beats: readonly number[];
  /** Expected interval (ms), default 720000 (12 minutes) */
  expectedIntervalMs: number;
  /** Whether the heartbeat is currently healthy */
  healthy: boolean;
  /** Consecutive missed beats */
  missedBeats: number;
  /** Average actual interval (ms) */
  averageIntervalMs: number;
}

/**
 * Create a heartbeat detector for monitoring Cloud Scheduler warmup pings.
 */
export function createHeartbeatDetector(
  expectedIntervalMs: number = 720_000
): HeartbeatState {
  return {
    beats: [],
    expectedIntervalMs,
    healthy: true,
    missedBeats: 0,
    averageIntervalMs: expectedIntervalMs,
  };
}

/**
 * Record a heartbeat event and update health status.
 */
export function recordHeartbeat(
  state: HeartbeatState,
  timestampMs: number
): HeartbeatState {
  const beats = [...state.beats, timestampMs];
  // Keep last 100 beats
  const trimmed = beats.length > 100 ? beats.slice(-100) : beats;

  if (trimmed.length < 2) {
    return { ...state, beats: trimmed };
  }

  // Compute intervals
  const intervals: number[] = [];
  for (let i = 1; i < trimmed.length; i++) {
    intervals.push(trimmed[i] - trimmed[i - 1]);
  }

  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const lastInterval = intervals[intervals.length - 1];

  // Missed beat if last interval > 1.5x expected
  const threshold = state.expectedIntervalMs * 1.5;
  const missedBeats = lastInterval > threshold
    ? Math.floor(lastInterval / state.expectedIntervalMs) - 1
    : 0;

  return {
    beats: trimmed,
    expectedIntervalMs: state.expectedIntervalMs,
    healthy: missedBeats === 0 && lastInterval < threshold,
    missedBeats,
    averageIntervalMs: avgInterval,
  };
}

/**
 * Check if a heartbeat is overdue given the current time.
 */
export function isHeartbeatOverdue(
  state: HeartbeatState,
  nowMs: number
): boolean {
  if (state.beats.length === 0) return false;
  const lastBeat = state.beats[state.beats.length - 1];
  return (nowMs - lastBeat) > state.expectedIntervalMs * 1.5;
}

/**
 * Extract heartbeat pattern from a VoidMap.
 * Looks for periodic tombstone-free intervals (healthy heartbeats produce
 * no tombstones; missed heartbeats produce tombstones from the warmup probe).
 */
export function detectHeartbeatPattern(
  map: VoidMap,
  options?: { windowSize?: number }
): {
  detected: boolean;
  periodRounds: number;
  confidence: number;
  missedBeatRounds: readonly number[];
} {
  const windowSize = options?.windowSize ?? 50;

  if (map.round < windowSize * 2) {
    return { detected: false, periodRounds: 0, confidence: 0, missedBeatRounds: [] };
  }

  // Count tombstones per round
  const countsPerRound = new Map<number, number>();
  for (const t of map.tombstones) {
    countsPerRound.set(t.round, (countsPerRound.get(t.round) ?? 0) + 1);
  }

  // Find rounds with zero tombstones (healthy heartbeats)
  const healthyRounds: number[] = [];
  for (let r = 1; r <= map.round; r++) {
    if ((countsPerRound.get(r) ?? 0) === 0) {
      healthyRounds.push(r);
    }
  }

  if (healthyRounds.length < 3) {
    return { detected: false, periodRounds: 0, confidence: 0, missedBeatRounds: [] };
  }

  // Compute intervals between healthy rounds
  const intervals: number[] = [];
  for (let i = 1; i < healthyRounds.length; i++) {
    intervals.push(healthyRounds[i] - healthyRounds[i - 1]);
  }

  // Find the most common interval (mode)
  const freq = new Map<number, number>();
  for (const interval of intervals) {
    freq.set(interval, (freq.get(interval) ?? 0) + 1);
  }

  let modeInterval = 1;
  let modeCount = 0;
  for (const [interval, count] of freq) {
    if (count > modeCount) {
      modeInterval = interval;
      modeCount = count;
    }
  }

  const confidence = intervals.length > 0 ? modeCount / intervals.length : 0;

  // Find missed beats: rounds where we expected a healthy heartbeat but got tombstones
  const missedBeatRounds: number[] = [];
  if (confidence > 0.3) {
    for (let r = healthyRounds[0] + modeInterval; r <= map.round; r += modeInterval) {
      if ((countsPerRound.get(r) ?? 0) > 0) {
        missedBeatRounds.push(r);
      }
    }
  }

  return {
    detected: confidence > 0.3,
    periodRounds: modeInterval,
    confidence,
    missedBeatRounds,
  };
}

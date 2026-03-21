import type {
  Vec3,
  TopologyBranch,
  TopologyFrameInput,
  TopologyFrameResult,
} from './index.js';
import { addVec3, scaleVec3, lengthVec3, subtractVec3 } from './index.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Tombstone {
  branchId: string;
  round: number;
  cost: number;
  quality: number;
  vertices: number;
  drawCalls: number;
  reason: 'over-budget' | 'explicit-vent' | 'race-loser';
  position: Vec3;
  timestamp: number;
}

export interface VoidMap {
  tombstones: readonly Tombstone[];
  round: number;
  rejectionCounts: ReadonlyMap<string, number>;
}

export interface VoidBoundaryStats {
  centroid: Vec3;
  spread: number;
  homologyRank: number;
  complementDistribution: number[];
  inverseBule: number;
  entropy: number;
  totalTombstones: number;
}

export interface VoidMapOptions {
  eta?: number;
  maxTombstones?: number;
  customMapper?: (branch: TopologyBranch, round: number, budgetMs: number) => Vec3;
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

const DEFAULT_ETA = 3.0;
const DEFAULT_MAX_TOMBSTONES = 10_000;

export function createVoidMap(options?: VoidMapOptions): VoidMap {
  return {
    tombstones: [],
    round: 0,
    rejectionCounts: new Map(),
  };
}

/**
 * Default coordinate mapping for a vented branch.
 * X = cost/budget, Y = quality, Z = log2(vertices)/20
 */
export function branchToPosition(
  branch: TopologyBranch,
  round: number,
  budgetMs: number
): Vec3 {
  return {
    x: budgetMs > 0 ? branch.estimatedCostMs / budgetMs : 0,
    y: branch.quality ?? 1,
    z: Math.log2(Math.max(1, branch.vertices)) / 20,
  };
}

function classifyReason(
  branch: TopologyBranch,
  result: TopologyFrameResult
): Tombstone['reason'] {
  if (branch.vent) return 'explicit-vent';
  if (branch.estimatedCostMs > result.budgetMs) return 'over-budget';
  return 'race-loser';
}

/**
 * Record one frame of topology execution into the void map.
 * Appends a tombstone for every vented branch.
 */
export function recordFrame(
  map: VoidMap,
  input: TopologyFrameInput,
  result: TopologyFrameResult,
  options?: VoidMapOptions
): VoidMap {
  const maxTombstones = options?.maxTombstones ?? DEFAULT_MAX_TOMBSTONES;
  const mapper = options?.customMapper ?? branchToPosition;
  const budgetMs = result.budgetMs;
  const ventedSet = new Set(result.ventedBranchIds);
  const round = map.round + 1;

  const branchLookup = new Map(input.branches.map((b) => [b.id, b]));

  const newTombstones: Tombstone[] = [];
  const newCounts = new Map(map.rejectionCounts);

  for (const ventedId of ventedSet) {
    const branch = branchLookup.get(ventedId);
    if (!branch) continue;

    const tombstone: Tombstone = {
      branchId: branch.id,
      round,
      cost: branch.estimatedCostMs,
      quality: branch.quality ?? 1,
      vertices: branch.vertices,
      drawCalls: branch.drawCalls,
      reason: classifyReason(branch, result),
      position: mapper(branch, round, budgetMs),
      timestamp: Date.now(),
    };

    newTombstones.push(tombstone);
    newCounts.set(branch.id, (newCounts.get(branch.id) ?? 0) + 1);
  }

  let allTombstones = [...map.tombstones, ...newTombstones];
  if (allTombstones.length > maxTombstones) {
    allTombstones = allTombstones.slice(allTombstones.length - maxTombstones);
  }

  return {
    tombstones: allTombstones,
    round,
    rejectionCounts: newCounts,
  };
}

// ---------------------------------------------------------------------------
// Void boundary statistics
// ---------------------------------------------------------------------------

/**
 * Complement distribution: softmax(-eta * rejectionCount).
 * Matches the canonical ch17 reference (ch17-netflix-void-walker.ts:321-327).
 */
export function complementDistribution(
  map: VoidMap,
  eta: number = DEFAULT_ETA
): number[] {
  const counts = Array.from(map.rejectionCounts.values());
  if (counts.length === 0) return [];

  const logits = counts.map((v) => -eta * v);
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Shannon entropy of the complement distribution.
 * Matches the canonical ch17 reference (ch17-netflix-void-walker.ts:334-343).
 */
export function voidEntropy(map: VoidMap, eta: number = DEFAULT_ETA): number {
  const dist = complementDistribution(map, eta);
  let h = 0;
  for (const p of dist) {
    if (p > 1e-10) {
      h -= p * Math.log2(p);
    }
  }
  return h;
}

/**
 * Compute full void boundary statistics.
 */
export function computeVoidBoundary(
  map: VoidMap,
  options?: VoidMapOptions
): VoidBoundaryStats {
  const eta = options?.eta ?? DEFAULT_ETA;
  const tombstones = map.tombstones;
  const total = tombstones.length;

  if (total === 0) {
    return {
      centroid: { x: 0, y: 0, z: 0 },
      spread: 0,
      homologyRank: 0,
      complementDistribution: [],
      inverseBule: 0,
      entropy: 0,
      totalTombstones: 0,
    };
  }

  // Centroid
  let sum: Vec3 = { x: 0, y: 0, z: 0 };
  for (const t of tombstones) {
    sum = addVec3(sum, t.position);
  }
  const centroid = scaleVec3(sum, 1 / total);

  // Spread (RMS distance from centroid)
  let spreadSum = 0;
  for (const t of tombstones) {
    const d = lengthVec3(subtractVec3(t.position, centroid));
    spreadSum += d * d;
  }
  const spread = Math.sqrt(spreadSum / total);

  const compDist = complementDistribution(map, eta);
  const entropy = voidEntropy(map, eta);

  // Inverse Bule: 1 - entropy / maxEntropy
  // When entropy is maximal (uniform), inverseBule = 0 (no structure).
  // When entropy is minimal (peaked), inverseBule -> 1 (strong rejection signal).
  const maxEntropy = compDist.length > 1 ? Math.log2(compDist.length) : 1;
  const inverseBule = maxEntropy > 0 ? 1 - entropy / maxEntropy : 0;

  // Homology rank estimate: count of distinct branches that appear in tombstones
  const distinctBranches = new Set(tombstones.map((t) => t.branchId));
  const homologyRank = distinctBranches.size;

  return {
    centroid,
    spread,
    homologyRank,
    complementDistribution: compDist,
    inverseBule,
    entropy,
    totalTombstones: total,
  };
}

import type { Vec3 } from './index.js';
import { addVec3, scaleVec3, subtractVec3, lengthVec3 } from './index.js';
import type {
  VoidMap,
  VoidMapOptions,
  VoidBoundaryStats,
  Tombstone,
} from './voidmap.js';
import {
  createVoidMap,
  complementDistribution,
  voidEntropy,
  computeVoidBoundary,
} from './voidmap.js';

// ---------------------------------------------------------------------------
// Temporal slice: git checkout <round>
// ---------------------------------------------------------------------------

/**
 * Slice a VoidMap to only include tombstones up to a given round.
 * This is `git checkout <round>` -- view the void as it existed then.
 * Rejection counts are recomputed from the filtered tombstones so
 * the complement distribution is accurate for that point in time.
 */
export function sliceAtRound(map: VoidMap, round: number): VoidMap {
  if (round >= map.round) return map;
  if (round <= 0) return createVoidMap();

  const filtered = map.tombstones.filter((t) => t.round <= round);
  const counts = new Map<string, number>();
  for (const t of filtered) {
    counts.set(t.branchId, (counts.get(t.branchId) ?? 0) + 1);
  }

  return {
    tombstones: filtered,
    round,
    rejectionCounts: counts,
  };
}

/**
 * Slice a VoidMap to a range of rounds [fromRound, toRound].
 * Like `git log fromRound..toRound` -- see what happened in a window.
 */
export function sliceRange(
  map: VoidMap,
  fromRound: number,
  toRound: number
): VoidMap {
  const filtered = map.tombstones.filter(
    (t) => t.round >= fromRound && t.round <= toRound
  );
  const counts = new Map<string, number>();
  for (const t of filtered) {
    counts.set(t.branchId, (counts.get(t.branchId) ?? 0) + 1);
  }

  return {
    tombstones: filtered,
    round: Math.min(toRound, map.round),
    rejectionCounts: counts,
  };
}

// ---------------------------------------------------------------------------
// Temporal diff: the irreducible gap between two void states
// ---------------------------------------------------------------------------

export interface VoidDiff {
  /** Tombstones present in A but not B (by branchId+round key) */
  onlyInA: readonly Tombstone[];
  /** Tombstones present in B but not A */
  onlyInB: readonly Tombstone[];
  /** Tombstones present in both */
  shared: readonly Tombstone[];
  /** Per-branch rejection count delta: positive = A rejected more, negative = B rejected more */
  rejectionDelta: ReadonlyMap<string, number>;
  /** Complement distribution divergence (Jensen-Shannon) */
  jsDivergence: number;
  /** Whether the diff is mergeable (JS divergence below threshold) */
  mergeable: boolean;
}

const DEFAULT_MERGE_THRESHOLD = 0.5;

function tombstoneKey(t: Tombstone): string {
  return `${t.branchId}:${t.round}`;
}

/**
 * Jensen-Shannon divergence between two distributions.
 * Returns 0 for identical distributions, up to ln(2) for maximally different.
 * Normalized to [0, 1] range.
 */
function jensenShannonDivergence(p: number[], q: number[]): number {
  if (p.length === 0 && q.length === 0) return 0;
  if (p.length !== q.length) {
    // Pad shorter distribution with zeros
    const maxLen = Math.max(p.length, q.length);
    const pp = [...p, ...new Array(maxLen - p.length).fill(0)];
    const qq = [...q, ...new Array(maxLen - q.length).fill(0)];
    return jensenShannonDivergence(pp, qq);
  }

  const m = p.map((pi, i) => (pi + q[i]) / 2);
  let klPM = 0;
  let klQM = 0;

  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-10 && m[i] > 1e-10) {
      klPM += p[i] * Math.log2(p[i] / m[i]);
    }
    if (q[i] > 1e-10 && m[i] > 1e-10) {
      klQM += q[i] * Math.log2(q[i] / m[i]);
    }
  }

  return (klPM + klQM) / 2;
}

/**
 * Compute the diff between two VoidMaps.
 * Like `git diff branchA branchB` -- what diverged between two parallel voids.
 */
export function diffVoidMaps(
  a: VoidMap,
  b: VoidMap,
  options?: { eta?: number; mergeThreshold?: number }
): VoidDiff {
  const eta = options?.eta ?? 3.0;
  const mergeThreshold = options?.mergeThreshold ?? DEFAULT_MERGE_THRESHOLD;

  const aKeys = new Set(a.tombstones.map(tombstoneKey));
  const bKeys = new Set(b.tombstones.map(tombstoneKey));

  const onlyInA = a.tombstones.filter((t) => !bKeys.has(tombstoneKey(t)));
  const onlyInB = b.tombstones.filter((t) => !aKeys.has(tombstoneKey(t)));
  const shared = a.tombstones.filter((t) => bKeys.has(tombstoneKey(t)));

  // Rejection count delta
  const allBranches = new Set([
    ...a.rejectionCounts.keys(),
    ...b.rejectionCounts.keys(),
  ]);
  const rejectionDelta = new Map<string, number>();
  for (const branchId of allBranches) {
    const aCount = a.rejectionCounts.get(branchId) ?? 0;
    const bCount = b.rejectionCounts.get(branchId) ?? 0;
    rejectionDelta.set(branchId, aCount - bCount);
  }

  // Complement distribution divergence (aligned by branch ID)
  const allBranchIds = Array.from(allBranches).sort();
  function computeAligned(map: VoidMap): number[] {
    if (allBranchIds.length === 0) return [];
    const counts = allBranchIds.map((b) => map.rejectionCounts.get(b) ?? 0);
    const logits = counts.map((v) => -eta * v);
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }
  const distA = computeAligned(a);
  const distB = computeAligned(b);
  const jsDivergence = jensenShannonDivergence(distA, distB);

  return {
    onlyInA,
    onlyInB,
    shared,
    rejectionDelta,
    jsDivergence,
    mergeable: jsDivergence < mergeThreshold,
  };
}

// ---------------------------------------------------------------------------
// Fork: parallel universes of rejection
// ---------------------------------------------------------------------------

export interface VoidFork {
  id: string;
  label: string;
  forkRound: number;
  forkTimestamp: number;
  map: VoidMap;
}

export interface VoidTimeline {
  trunk: VoidMap;
  forks: readonly VoidFork[];
}

/**
 * Create a timeline from an initial VoidMap (the trunk).
 */
export function createTimeline(trunk?: VoidMap): VoidTimeline {
  return {
    trunk: trunk ?? createVoidMap(),
    forks: [],
  };
}

/**
 * Fork the trunk (or another fork) at a specific round.
 * The fork starts with the same rejection history up to that round,
 * then diverges independently. Like `git branch feature <commit>`.
 */
export function forkAt(
  timeline: VoidTimeline,
  id: string,
  label: string,
  round?: number
): VoidTimeline {
  const forkRound = round ?? timeline.trunk.round;
  const forkedMap = sliceAtRound(timeline.trunk, forkRound);

  const fork: VoidFork = {
    id,
    label,
    forkRound,
    forkTimestamp: Date.now(),
    map: forkedMap,
  };

  return {
    trunk: timeline.trunk,
    forks: [...timeline.forks, fork],
  };
}

/**
 * Fork from an existing fork (nested branching).
 * Like `git checkout feature-a && git branch feature-b`.
 */
export function forkFrom(
  timeline: VoidTimeline,
  sourceForkId: string,
  newId: string,
  newLabel: string,
  round?: number
): VoidTimeline {
  const source = timeline.forks.find((f) => f.id === sourceForkId);
  if (!source) return timeline;

  const forkRound = round ?? source.map.round;
  const forkedMap = sliceAtRound(source.map, forkRound);

  const fork: VoidFork = {
    id: newId,
    label: newLabel,
    forkRound,
    forkTimestamp: Date.now(),
    map: forkedMap,
  };

  return {
    trunk: timeline.trunk,
    forks: [...timeline.forks, fork],
  };
}

/**
 * Update the trunk with a new VoidMap (after recordFrame).
 */
export function updateTrunk(
  timeline: VoidTimeline,
  trunk: VoidMap
): VoidTimeline {
  return { trunk, forks: timeline.forks };
}

/**
 * Update a fork's VoidMap (after recordFrame with different parameters).
 */
export function updateFork(
  timeline: VoidTimeline,
  forkId: string,
  map: VoidMap
): VoidTimeline {
  return {
    trunk: timeline.trunk,
    forks: timeline.forks.map((f) =>
      f.id === forkId ? { ...f, map } : f
    ),
  };
}

/**
 * Attempt to merge a fork back into the trunk.
 * If the complement distributions have diverged beyond the merge threshold,
 * the merge is rejected and the diff is returned as an unmergeable witness.
 *
 * Like `git merge feature` -- but some branches are structurally unmergeable,
 * which is the proof that diversity is necessary (the pigeonhole witness).
 */
export function mergeToTrunk(
  timeline: VoidTimeline,
  forkId: string,
  options?: { eta?: number; mergeThreshold?: number }
): { timeline: VoidTimeline; diff: VoidDiff; merged: boolean } {
  const fork = timeline.forks.find((f) => f.id === forkId);
  if (!fork) {
    return {
      timeline,
      diff: {
        onlyInA: [],
        onlyInB: [],
        shared: [],
        rejectionDelta: new Map(),
        jsDivergence: 0,
        mergeable: false,
      },
      merged: false,
    };
  }

  const diff = diffVoidMaps(timeline.trunk, fork.map, options);

  if (!diff.mergeable) {
    return { timeline, diff, merged: false };
  }

  // Merge: union tombstones, combine rejection counts, take max round
  const trunkKeys = new Set(timeline.trunk.tombstones.map(tombstoneKey));
  const mergedTombstones = [
    ...timeline.trunk.tombstones,
    ...fork.map.tombstones.filter((t) => !trunkKeys.has(tombstoneKey(t))),
  ];

  const mergedCounts = new Map(timeline.trunk.rejectionCounts);
  for (const [branchId, count] of fork.map.rejectionCounts) {
    mergedCounts.set(
      branchId,
      Math.max(mergedCounts.get(branchId) ?? 0, count)
    );
  }

  const mergedTrunk: VoidMap = {
    tombstones: mergedTombstones,
    round: Math.max(timeline.trunk.round, fork.map.round),
    rejectionCounts: mergedCounts,
  };

  // Remove the merged fork
  return {
    timeline: {
      trunk: mergedTrunk,
      forks: timeline.forks.filter((f) => f.id !== forkId),
    },
    diff,
    merged: true,
  };
}

// ---------------------------------------------------------------------------
// Tombstone aging: opacity by temporal distance
// ---------------------------------------------------------------------------

/**
 * Compute opacity for a tombstone based on its temporal distance
 * from the current view round. Recent = 1.0, old = fading toward minOpacity.
 *
 * Uses exponential decay: opacity = minOpacity + (1 - minOpacity) * exp(-decay * age)
 */
export function tombstoneOpacity(
  tombstone: Tombstone,
  currentRound: number,
  options?: { decay?: number; minOpacity?: number }
): number {
  const decay = options?.decay ?? 0.05;
  const minOpacity = options?.minOpacity ?? 0.1;
  const age = Math.max(0, currentRound - tombstone.round);
  return minOpacity + (1 - minOpacity) * Math.exp(-decay * age);
}

/**
 * Compute per-tombstone opacities for an entire VoidMap at a given view round.
 * Returns a Float32Array suitable for a buffer attribute.
 */
export function computeOpacities(
  map: VoidMap,
  currentRound: number,
  options?: { decay?: number; minOpacity?: number }
): Float32Array {
  const opacities = new Float32Array(map.tombstones.length);
  for (let i = 0; i < map.tombstones.length; i++) {
    opacities[i] = tombstoneOpacity(map.tombstones[i], currentRound, options);
  }
  return opacities;
}

// ---------------------------------------------------------------------------
// Death trajectory: trail lines per branch across rounds
// ---------------------------------------------------------------------------

export interface DeathTrajectory {
  branchId: string;
  positions: readonly Vec3[];
  rounds: readonly number[];
  rejectionCount: number;
}

/**
 * Extract death trajectories: for each branch, the sequence of positions
 * where it was vented across rounds. Branches that die repeatedly trace
 * a path through the void -- their "death trajectory".
 */
export function extractTrajectories(map: VoidMap): readonly DeathTrajectory[] {
  const byBranch = new Map<string, Tombstone[]>();
  for (const t of map.tombstones) {
    let list = byBranch.get(t.branchId);
    if (!list) {
      list = [];
      byBranch.set(t.branchId, list);
    }
    list.push(t);
  }

  const trajectories: DeathTrajectory[] = [];
  for (const [branchId, tombstones] of byBranch) {
    if (tombstones.length < 2) continue;
    const sorted = [...tombstones].sort((a, b) => a.round - b.round);
    trajectories.push({
      branchId,
      positions: sorted.map((t) => t.position),
      rounds: sorted.map((t) => t.round),
      rejectionCount: sorted.length,
    });
  }

  return trajectories.sort((a, b) => b.rejectionCount - a.rejectionCount);
}

// ---------------------------------------------------------------------------
// Snapshot: content-addressed void state for ghost packaging
// ---------------------------------------------------------------------------

export interface VoidMapSnapshot {
  cid: string;
  round: number;
  tombstoneCount: number;
  stats: VoidBoundaryStats;
  serialized: string;
}

/**
 * Snapshot a VoidMap for ghost packaging.
 * Content-addressed by SHA-256 of the serialized state (when crypto available)
 * or a deterministic hash fallback.
 */
export async function snapshotVoidMap(
  map: VoidMap,
  options?: VoidMapOptions
): Promise<VoidMapSnapshot> {
  const stats = computeVoidBoundary(map, options);

  const serializable = {
    tombstones: map.tombstones,
    round: map.round,
    rejectionCounts: Array.from(map.rejectionCounts.entries()),
  };
  const serialized = JSON.stringify(serializable);

  let cid: string;
  if (typeof globalThis.crypto?.subtle?.digest === 'function') {
    const encoded = new TextEncoder().encode(serialized);
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoded);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    cid = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  } else {
    // Deterministic fallback: simple FNV-1a hash
    let hash = 0x811c9dc5;
    for (let i = 0; i < serialized.length; i++) {
      hash ^= serialized.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    cid = `fnv1a-${(hash >>> 0).toString(16).padStart(8, '0')}-r${map.round}-t${map.tombstones.length}`;
  }

  return {
    cid,
    round: map.round,
    tombstoneCount: map.tombstones.length,
    stats,
    serialized,
  };
}

/**
 * Restore a VoidMap from a snapshot.
 */
export function restoreFromSnapshot(snapshot: VoidMapSnapshot): VoidMap {
  const parsed = JSON.parse(snapshot.serialized);
  return {
    tombstones: parsed.tombstones,
    round: parsed.round,
    rejectionCounts: new Map(parsed.rejectionCounts),
  };
}

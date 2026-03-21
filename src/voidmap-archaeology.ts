/**
 * voidmap-archaeology.ts
 *
 * Tools for querying, searching, filtering, and bookmarking the void.
 * The void is a queryable database of rejection history rendered as
 * navigable 3D space. Scrub back to 2026 and see the first tombstones.
 */

import type { Vec3 } from './index.js';
import type {
  VoidMap,
  Tombstone,
  VoidBoundaryStats,
  VoidMapOptions,
} from './voidmap.js';
import { computeVoidBoundary } from './voidmap.js';
import type { VoidMapSnapshot } from './voidmap-temporal.js';
import { sliceAtRound, restoreFromSnapshot } from './voidmap-temporal.js';
import type { CentroidWaypoint, PhaseTransition } from './voidmap-precognition.js';
import { computeCentroidContrail, detectPhaseTransitions } from './voidmap-precognition.js';

// ---------------------------------------------------------------------------
// Search and filter
// ---------------------------------------------------------------------------

export interface TombstoneQuery {
  /** Filter by branch ID (exact match or pattern) */
  branchId?: string;
  /** Filter by branch ID prefix */
  branchIdPrefix?: string;
  /** Filter by rejection reason */
  reason?: Tombstone['reason'] | readonly Tombstone['reason'][];
  /** Filter by round range [min, max] */
  roundRange?: readonly [number, number];
  /** Filter by timestamp range [minMs, maxMs] */
  timestampRange?: readonly [number, number];
  /** Filter by coordinate region (AABB) */
  region?: {
    min: Vec3;
    max: Vec3;
  };
  /** Filter by minimum cost */
  minCost?: number;
  /** Filter by maximum quality */
  maxQuality?: number;
  /** Filter by minimum rejection count for the branch */
  minRejectionCount?: number;
  /** Limit results */
  limit?: number;
  /** Sort order */
  sortBy?: 'round-asc' | 'round-desc' | 'cost-desc' | 'quality-asc';
}

/**
 * Search tombstones with flexible query filters.
 */
export function searchTombstones(
  map: VoidMap,
  query: TombstoneQuery
): readonly Tombstone[] {
  let results: Tombstone[] = [];

  for (const t of map.tombstones) {
    // Branch ID filter
    if (query.branchId !== undefined && t.branchId !== query.branchId) continue;
    if (query.branchIdPrefix !== undefined && !t.branchId.startsWith(query.branchIdPrefix)) continue;

    // Reason filter
    if (query.reason !== undefined) {
      const reasons = Array.isArray(query.reason) ? query.reason : [query.reason];
      if (!reasons.includes(t.reason)) continue;
    }

    // Round range
    if (query.roundRange !== undefined) {
      if (t.round < query.roundRange[0] || t.round > query.roundRange[1]) continue;
    }

    // Timestamp range
    if (query.timestampRange !== undefined) {
      if (t.timestamp < query.timestampRange[0] || t.timestamp > query.timestampRange[1]) continue;
    }

    // Coordinate region (AABB containment)
    if (query.region !== undefined) {
      const r = query.region;
      if (
        t.position.x < r.min.x || t.position.x > r.max.x ||
        t.position.y < r.min.y || t.position.y > r.max.y ||
        t.position.z < r.min.z || t.position.z > r.max.z
      ) continue;
    }

    // Cost filter
    if (query.minCost !== undefined && t.cost < query.minCost) continue;

    // Quality filter
    if (query.maxQuality !== undefined && t.quality > query.maxQuality) continue;

    // Rejection count filter
    if (query.minRejectionCount !== undefined) {
      const count = map.rejectionCounts.get(t.branchId) ?? 0;
      if (count < query.minRejectionCount) continue;
    }

    results.push(t);
  }

  // Sort
  if (query.sortBy) {
    switch (query.sortBy) {
      case 'round-asc':
        results.sort((a, b) => a.round - b.round);
        break;
      case 'round-desc':
        results.sort((a, b) => b.round - a.round);
        break;
      case 'cost-desc':
        results.sort((a, b) => b.cost - a.cost);
        break;
      case 'quality-asc':
        results.sort((a, b) => a.quality - b.quality);
        break;
    }
  }

  // Limit
  if (query.limit !== undefined && results.length > query.limit) {
    results = results.slice(0, query.limit);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Era highlighting: color tombstones by time period
// ---------------------------------------------------------------------------

export interface Era {
  label: string;
  fromRound: number;
  toRound: number;
  color: string;
}

export interface HighlightedTombstone {
  tombstone: Tombstone;
  era: Era | null;
  /** 0-1 alpha based on era membership */
  highlight: number;
}

/**
 * Tag each tombstone with its era (or null if outside all eras).
 * Tombstones in an era get highlight=1, others get highlight based on dimFactor.
 */
export function highlightByEra(
  map: VoidMap,
  eras: readonly Era[],
  dimFactor: number = 0.15
): readonly HighlightedTombstone[] {
  return map.tombstones.map((t) => {
    const matchedEra = eras.find(
      (era) => t.round >= era.fromRound && t.round <= era.toRound
    ) ?? null;

    return {
      tombstone: t,
      era: matchedEra,
      highlight: matchedEra ? 1 : dimFactor,
    };
  });
}

// ---------------------------------------------------------------------------
// Tombstone density analysis
// ---------------------------------------------------------------------------

export interface DensityPoint {
  round: number;
  count: number;
  /** Cumulative count up to this round */
  cumulative: number;
}

/**
 * Compute tombstone density per round (how many tombstones were generated
 * at each round). This is the "error rate" view of the void.
 */
export function tombstoneDensityPerRound(map: VoidMap): readonly DensityPoint[] {
  if (map.tombstones.length === 0) return [];

  const countsByRound = new Map<number, number>();
  for (const t of map.tombstones) {
    countsByRound.set(t.round, (countsByRound.get(t.round) ?? 0) + 1);
  }

  const points: DensityPoint[] = [];
  let cumulative = 0;

  for (let r = 1; r <= map.round; r++) {
    const count = countsByRound.get(r) ?? 0;
    cumulative += count;
    points.push({ round: r, count, cumulative });
  }

  return points;
}

/**
 * Compute windowed density (moving average over windowSize rounds).
 */
export function windowedDensity(
  density: readonly DensityPoint[],
  windowSize: number
): readonly { round: number; averageDensity: number }[] {
  if (density.length < windowSize) return [];

  const result: { round: number; averageDensity: number }[] = [];
  let windowSum = 0;

  for (let i = 0; i < density.length; i++) {
    windowSum += density[i].count;
    if (i >= windowSize) {
      windowSum -= density[i - windowSize].count;
    }
    if (i >= windowSize - 1) {
      result.push({
        round: density[i].round,
        averageDensity: windowSum / windowSize,
      });
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Branch failure timeline
// ---------------------------------------------------------------------------

export interface BranchTimeline {
  branchId: string;
  totalRejections: number;
  firstRound: number;
  lastRound: number;
  /** Rounds at which this branch was rejected */
  rejectionRounds: readonly number[];
  /** Average cost at rejection */
  averageCost: number;
  /** Average quality at rejection */
  averageQuality: number;
  /** Gaps between rejections (round deltas) */
  intervals: readonly number[];
  /** Is the branch still being rejected recently? (within last 10% of rounds) */
  active: boolean;
}

/**
 * Build a complete timeline for a specific branch: when it failed,
 * how often, and whether it's still failing.
 */
export function branchFailureTimeline(
  map: VoidMap,
  branchId: string
): BranchTimeline | null {
  const tombstones = map.tombstones.filter((t) => t.branchId === branchId);
  if (tombstones.length === 0) return null;

  const sorted = [...tombstones].sort((a, b) => a.round - b.round);
  const rejectionRounds = sorted.map((t) => t.round);

  const intervals: number[] = [];
  for (let i = 1; i < rejectionRounds.length; i++) {
    intervals.push(rejectionRounds[i] - rejectionRounds[i - 1]);
  }

  const totalCost = sorted.reduce((sum, t) => sum + t.cost, 0);
  const totalQuality = sorted.reduce((sum, t) => sum + t.quality, 0);
  const recentThreshold = map.round * 0.9;

  return {
    branchId,
    totalRejections: sorted.length,
    firstRound: sorted[0].round,
    lastRound: sorted[sorted.length - 1].round,
    rejectionRounds,
    averageCost: totalCost / sorted.length,
    averageQuality: totalQuality / sorted.length,
    intervals,
    active: sorted[sorted.length - 1].round >= recentThreshold,
  };
}

/**
 * Get all branch timelines, sorted by rejection count (most rejected first).
 */
export function allBranchTimelines(map: VoidMap): readonly BranchTimeline[] {
  const branchIds = new Set(map.tombstones.map((t) => t.branchId));
  const timelines: BranchTimeline[] = [];

  for (const branchId of branchIds) {
    const timeline = branchFailureTimeline(map, branchId);
    if (timeline) timelines.push(timeline);
  }

  return timelines.sort((a, b) => b.totalRejections - a.totalRejections);
}

// ---------------------------------------------------------------------------
// Snapshot chain reconstruction
// ---------------------------------------------------------------------------

export interface ReconstructedHistory {
  /** Combined tombstones from all snapshots, deduplicated */
  tombstones: readonly Tombstone[];
  /** Snapshot CIDs in chronological order */
  snapshotChain: readonly string[];
  /** Round range covered */
  fromRound: number;
  toRound: number;
  /** Total unique tombstones recovered */
  totalRecovered: number;
  /** Tombstones that were lost (in gaps between snapshots) */
  estimatedLost: number;
}

/**
 * Reconstruct full void history from a chain of periodic VoidMapSnapshots.
 * Since the maxTombstones cap evicts old tombstones, the only way to see
 * the complete history is to chain snapshots taken at different times.
 *
 * Deduplicates tombstones by (branchId, round) key.
 */
export function reconstructFromSnapshotChain(
  snapshots: readonly VoidMapSnapshot[]
): ReconstructedHistory {
  if (snapshots.length === 0) {
    return {
      tombstones: [],
      snapshotChain: [],
      fromRound: 0,
      toRound: 0,
      totalRecovered: 0,
      estimatedLost: 0,
    };
  }

  // Sort snapshots by round
  const sorted = [...snapshots].sort((a, b) => a.round - b.round);
  const snapshotChain = sorted.map((s) => s.cid);

  // Collect all tombstones, deduplicate by (branchId, round)
  const seen = new Set<string>();
  const allTombstones: Tombstone[] = [];

  for (const snapshot of sorted) {
    const restored = restoreFromSnapshot(snapshot);
    for (const t of restored.tombstones) {
      const key = `${t.branchId}:${t.round}`;
      if (!seen.has(key)) {
        seen.add(key);
        allTombstones.push(t);
      }
    }
  }

  // Sort by round
  allTombstones.sort((a, b) => a.round - b.round);

  const fromRound = allTombstones.length > 0 ? allTombstones[0].round : 0;
  const toRound = sorted[sorted.length - 1].round;

  // Estimate lost tombstones: gaps in the round sequence where we'd expect
  // tombstones but have none (between snapshots, some may have been evicted
  // before the next snapshot was taken)
  const roundsWithTombstones = new Set(allTombstones.map((t) => t.round));
  let gapRounds = 0;
  for (let r = fromRound; r <= toRound; r++) {
    if (!roundsWithTombstones.has(r)) gapRounds++;
  }

  // Rough estimate: if average density is D tombstones/round, then
  // gapRounds * D is the estimated number of lost tombstones
  const coveredRounds = toRound - fromRound + 1 - gapRounds;
  const avgDensity = coveredRounds > 0 ? allTombstones.length / coveredRounds : 0;
  const estimatedLost = Math.round(gapRounds * avgDensity);

  return {
    tombstones: allTombstones,
    snapshotChain,
    fromRound,
    toRound,
    totalRecovered: allTombstones.length,
    estimatedLost,
  };
}

// ---------------------------------------------------------------------------
// Void bookmarks: saved view states at significant moments
// ---------------------------------------------------------------------------

export interface VoidBookmark {
  /** Unique bookmark ID */
  id: string;
  /** Human-readable label */
  label: string;
  /** Round this bookmark points to */
  round: number;
  /** Timestamp when the bookmark was created */
  createdAt: number;
  /** Camera position for restoring the 3D view */
  cameraPosition: Vec3;
  /** Camera look-at target */
  cameraTarget: Vec3;
  /** Which layers were visible */
  layers: {
    contrail: boolean;
    ghosts: boolean;
    cone: boolean;
    phases: boolean;
    trajectories: boolean;
  };
  /** Optional search query that was active */
  activeQuery?: TombstoneQuery;
  /** Stats at the bookmarked round */
  stats: {
    entropy: number;
    inverseBule: number;
    totalTombstones: number;
    homologyRank: number;
  };
  /** Why this bookmark was created */
  reason: 'manual' | 'phase-convergence' | 'phase-disruption' | 'milestone';
  /** Associated phase transition (if auto-created at a phase transition) */
  phaseTransition?: PhaseTransition;
}

export interface VoidBookmarkStore {
  bookmarks: readonly VoidBookmark[];
}

/**
 * Create a bookmark store.
 */
export function createBookmarkStore(): VoidBookmarkStore {
  return { bookmarks: [] };
}

/**
 * Add a bookmark to the store.
 */
export function addBookmark(
  store: VoidBookmarkStore,
  bookmark: VoidBookmark
): VoidBookmarkStore {
  return { bookmarks: [...store.bookmarks, bookmark] };
}

/**
 * Remove a bookmark by ID.
 */
export function removeBookmark(
  store: VoidBookmarkStore,
  bookmarkId: string
): VoidBookmarkStore {
  return {
    bookmarks: store.bookmarks.filter((b) => b.id !== bookmarkId),
  };
}

/**
 * Create a manual bookmark at the current view state.
 */
export function createManualBookmark(
  map: VoidMap,
  round: number,
  label: string,
  cameraPosition: Vec3,
  cameraTarget: Vec3,
  layers: VoidBookmark['layers'],
  activeQuery?: TombstoneQuery
): VoidBookmark {
  const sliced = sliceAtRound(map, round);
  const stats = computeVoidBoundary(sliced);

  return {
    id: `bk-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    label,
    round,
    createdAt: Date.now(),
    cameraPosition,
    cameraTarget,
    layers,
    activeQuery,
    stats: {
      entropy: stats.entropy,
      inverseBule: stats.inverseBule,
      totalTombstones: stats.totalTombstones,
      homologyRank: stats.homologyRank,
    },
    reason: 'manual',
  };
}

/**
 * Auto-generate bookmarks at every phase transition in the void.
 * Returns bookmarks for convergence and disruption events.
 */
export function autoBookmarkPhaseTransitions(
  map: VoidMap,
  options?: { eta?: number; entropyThreshold?: number }
): readonly VoidBookmark[] {
  const eta = options?.eta ?? 3.0;
  const contrail = computeCentroidContrail(map, { eta });
  const transitions = detectPhaseTransitions(contrail, {
    entropyThreshold: options?.entropyThreshold ?? 0.3,
  });

  return transitions.map((t) => {
    const sliced = sliceAtRound(map, t.round);
    const stats = computeVoidBoundary(sliced);
    const waypoint = contrail.find((w) => w.round === t.round);

    return {
      id: `auto-${t.type}-${t.round}`,
      label: `${t.type === 'convergence' ? 'Convergence' : 'Disruption'} at round ${t.round}`,
      round: t.round,
      createdAt: Date.now(),
      cameraPosition: waypoint
        ? { x: waypoint.position.x + 2, y: waypoint.position.y + 1, z: waypoint.position.z + 2 }
        : { x: 3, y: 2, z: 3 },
      cameraTarget: waypoint?.position ?? { x: 0, y: 0, z: 0 },
      layers: {
        contrail: true,
        ghosts: false,
        cone: false,
        phases: true,
        trajectories: true,
      },
      stats: {
        entropy: stats.entropy,
        inverseBule: stats.inverseBule,
        totalTombstones: stats.totalTombstones,
        homologyRank: stats.homologyRank,
      },
      reason: t.type === 'convergence' ? 'phase-convergence' as const : 'phase-disruption' as const,
      phaseTransition: t,
    };
  });
}

/**
 * Create milestone bookmarks at regular intervals (every N rounds).
 */
export function createMilestoneBookmarks(
  map: VoidMap,
  interval: number = 1000
): readonly VoidBookmark[] {
  const bookmarks: VoidBookmark[] = [];

  for (let r = interval; r <= map.round; r += interval) {
    const sliced = sliceAtRound(map, r);
    const stats = computeVoidBoundary(sliced);

    bookmarks.push({
      id: `milestone-${r}`,
      label: `Round ${r}`,
      round: r,
      createdAt: Date.now(),
      cameraPosition: { x: 3, y: 2, z: 3 },
      cameraTarget: stats.centroid,
      layers: {
        contrail: true,
        ghosts: false,
        cone: false,
        phases: false,
        trajectories: false,
      },
      stats: {
        entropy: stats.entropy,
        inverseBule: stats.inverseBule,
        totalTombstones: stats.totalTombstones,
        homologyRank: stats.homologyRank,
      },
      reason: 'milestone',
    });
  }

  return bookmarks;
}

// ---------------------------------------------------------------------------
// Void summary: high-level narrative of the rejection history
// ---------------------------------------------------------------------------

export interface VoidSummary {
  /** Total rounds processed */
  totalRounds: number;
  /** Total tombstones */
  totalTombstones: number;
  /** Number of distinct branches */
  distinctBranches: number;
  /** Most rejected branch */
  mostRejected: { branchId: string; count: number } | null;
  /** Least rejected branch */
  leastRejected: { branchId: string; count: number } | null;
  /** Number of phase transitions */
  phaseTransitionCount: number;
  /** Number of convergence events */
  convergenceCount: number;
  /** Number of disruption events */
  disruptionCount: number;
  /** Current entropy */
  entropy: number;
  /** Current Inverse Bule */
  inverseBule: number;
  /** Average tombstone density (tombstones per round) */
  averageDensity: number;
  /** Peak density (max tombstones in a single round) */
  peakDensity: number;
  /** Peak density round */
  peakDensityRound: number;
  /** Centroid velocity: is the void still moving? */
  currentCentroidSpeed: number;
  /** Is the void converged? (low velocity + high Inverse Bule) */
  converged: boolean;
}

/**
 * Generate a high-level summary of the void's rejection history.
 */
export function summarizeVoid(
  map: VoidMap,
  options?: { eta?: number }
): VoidSummary {
  const eta = options?.eta ?? 3.0;
  const stats = computeVoidBoundary(map, { eta });

  // Most/least rejected
  let mostRejected: { branchId: string; count: number } | null = null;
  let leastRejected: { branchId: string; count: number } | null = null;
  for (const [branchId, count] of map.rejectionCounts) {
    if (!mostRejected || count > mostRejected.count) {
      mostRejected = { branchId, count };
    }
    if (!leastRejected || count < leastRejected.count) {
      leastRejected = { branchId, count };
    }
  }

  // Phase transitions
  const contrail = computeCentroidContrail(map, { eta });
  const transitions = detectPhaseTransitions(contrail);
  const convergenceCount = transitions.filter((t) => t.type === 'convergence').length;
  const disruptionCount = transitions.filter((t) => t.type === 'disruption').length;

  // Density
  const density = tombstoneDensityPerRound(map);
  const avgDensity = map.round > 0 ? map.tombstones.length / map.round : 0;
  let peakDensity = 0;
  let peakDensityRound = 0;
  for (const d of density) {
    if (d.count > peakDensity) {
      peakDensity = d.count;
      peakDensityRound = d.round;
    }
  }

  // Centroid speed
  let currentCentroidSpeed = 0;
  if (contrail.length >= 2) {
    const last = contrail[contrail.length - 1].position;
    const prev = contrail[contrail.length - 2].position;
    const dx = last.x - prev.x;
    const dy = last.y - prev.y;
    const dz = last.z - prev.z;
    currentCentroidSpeed = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  return {
    totalRounds: map.round,
    totalTombstones: map.tombstones.length,
    distinctBranches: map.rejectionCounts.size,
    mostRejected,
    leastRejected,
    phaseTransitionCount: transitions.length,
    convergenceCount,
    disruptionCount,
    entropy: stats.entropy,
    inverseBule: stats.inverseBule,
    averageDensity: Number(avgDensity.toFixed(3)),
    peakDensity,
    peakDensityRound,
    currentCentroidSpeed,
    converged: currentCentroidSpeed < 0.01 && stats.inverseBule > 0.5,
  };
}

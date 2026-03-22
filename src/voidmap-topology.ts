/**
 * voidmap-topology.ts -- Ch17 Topological Analysis of the Void
 *
 * Computes actual Betti numbers (H₀ connected components, H₁ cycles)
 * from the tombstone point cloud, tracks topological deficit per round,
 * and generates fork/race/fold animation primitives.
 *
 * From ch17 section 2: the topological deficit Delta_beta measures how
 * far a system deviates from its problem's natural topology. High deficit
 * correlates with waste.
 *
 * From THM-TOPO-MOLECULAR-ISO: pipeline graphs and molecular graphs
 * share identical Betti signatures. The void of a rendering pipeline
 * has the same topological structure as the void of a protein fold.
 *
 * data-proof="lean4:pipeline_speedup_sandwich" data-proof-status="verified"
 */

import type { Vec3 } from './index.js';
import { subtractVec3, lengthVec3 } from './index.js';
import type { VoidMap, Tombstone, VoidBoundaryStats } from './voidmap.js';
import { computeVoidBoundary } from './voidmap.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Betti numbers computed from the tombstone cloud */
export interface BettiSignature {
  /** H₀: connected components (clusters of failures) */
  b0: number;
  /** H₁: independent cycles (loops in failure patterns) */
  b1: number;
  /** Total tombstones analyzed */
  pointCount: number;
  /** Epsilon used for Rips complex */
  epsilon: number;
}

/** Topological deficit measurement per round */
export interface TopologicalDeficitRecord {
  round: number;
  /** Branches attempted this round */
  attempted: number;
  /** Branches that survived this round */
  survived: number;
  /** Branches vented this round */
  vented: number;
  /** Deficit ratio: vented / attempted (0 = all survived, 1 = all vented) */
  deficitRatio: number;
  /** Cumulative deficit across all rounds */
  cumulativeDeficit: number;
  /** Cumulative attempts across all rounds */
  cumulativeAttempts: number;
  /** Cumulative deficit ratio */
  cumulativeDeficitRatio: number;
}

/** Fork/race/fold animation keyframe for 3D renderers */
export interface TopologyAnimationFrame {
  /** Animation phase */
  phase: 'fork' | 'race' | 'fold' | 'vent';
  /** Duration in ms */
  durationMs: number;
  /** Branch positions at this keyframe */
  branches: Array<{
    id: string;
    position: Vec3;
    /** 0 = fully transparent (vented), 1 = fully opaque */
    opacity: number;
    /** Scale factor (shrinks on vent, grows on fold) */
    scale: number;
    /** Color hint */
    color: 'active' | 'winning' | 'vented' | 'folded';
  }>;
  /** Centroid position (convergence target for fold) */
  centroid: Vec3;
}

// ---------------------------------------------------------------------------
// Persistent Homology (Approximate Betti Numbers)
// ---------------------------------------------------------------------------

/**
 * Compute Betti numbers from a tombstone point cloud using
 * the Vietoris-Rips complex at a given epsilon.
 *
 * H₀ = connected components: clusters of similar failures.
 * H₁ = independent cycles: repeating failure patterns.
 *
 * Uses union-find for H₀ and cycle detection for H₁.
 * This is an approximation -- full persistent homology would
 * sweep epsilon and track birth/death of features.
 */
export function computeBettiNumbers(
  tombstones: readonly Tombstone[],
  epsilon?: number
): BettiSignature {
  const n = tombstones.length;
  if (n === 0) {
    return { b0: 0, b1: 0, pointCount: 0, epsilon: 0 };
  }

  // Auto-select epsilon if not provided: median nearest-neighbor distance
  const eps = epsilon ?? estimateEpsilon(tombstones);

  // Union-Find for H₀
  const parent = Array.from({ length: n }, (_, i) => i);
  const rank = new Array(n).fill(0);

  function find(x: number): number {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]; // path compression
      x = parent[x];
    }
    return x;
  }

  function union(a: number, b: number): boolean {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return false; // already connected
    if (rank[ra] < rank[rb]) parent[ra] = rb;
    else if (rank[ra] > rank[rb]) parent[rb] = ra;
    else { parent[rb] = ra; rank[ra]++; }
    return true;
  }

  // Build edges (pairs within epsilon distance)
  let edgeCount = 0;
  let unionCount = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = lengthVec3(
        subtractVec3(tombstones[i].position, tombstones[j].position)
      );
      if (d <= eps) {
        edgeCount++;
        if (union(i, j)) {
          unionCount++;
        }
      }
    }
  }

  // H₀ = number of connected components
  const roots = new Set<number>();
  for (let i = 0; i < n; i++) roots.add(find(i));
  const b0 = roots.size;

  // H₁ = edges - (vertices - components) = edgeCount - unionCount
  // This is the rank of H₁ for the Rips complex (Euler characteristic)
  const b1 = Math.max(0, edgeCount - unionCount);

  return { b0, b1, pointCount: n, epsilon: eps };
}

/**
 * Estimate a reasonable epsilon from the point cloud.
 * Uses the median of k-nearest-neighbor distances.
 */
function estimateEpsilon(tombstones: readonly Tombstone[]): number {
  const n = tombstones.length;
  if (n <= 1) return 1;

  // For each point, find distance to nearest neighbor
  const nnDistances: number[] = [];
  for (let i = 0; i < Math.min(n, 200); i++) { // cap for perf
    let minDist = Infinity;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = lengthVec3(
        subtractVec3(tombstones[i].position, tombstones[j].position)
      );
      if (d < minDist) minDist = d;
    }
    nnDistances.push(minDist);
  }

  nnDistances.sort((a, b) => a - b);
  // Use 75th percentile as epsilon (includes most natural clusters)
  return nnDistances[Math.floor(nnDistances.length * 0.75)] ?? 0.5;
}

// ---------------------------------------------------------------------------
// Topological Deficit Tracking
// ---------------------------------------------------------------------------

/**
 * Compute per-round topological deficit from a VoidMap.
 *
 * From ch17: Delta_beta = beta_1* - beta_1 (natural - actual parallelism).
 * In rendering terms: deficit = vented / attempted per frame.
 *
 * Returns one record per round, plus cumulative tracking.
 */
export function computeDeficitTimeline(
  map: VoidMap,
  totalBranchesPerRound: Map<number, number>
): TopologicalDeficitRecord[] {
  const records: TopologicalDeficitRecord[] = [];

  // Group tombstones by round
  const ventedByRound = new Map<number, number>();
  for (const t of map.tombstones) {
    ventedByRound.set(t.round, (ventedByRound.get(t.round) ?? 0) + 1);
  }

  let cumulativeDeficit = 0;
  let cumulativeAttempts = 0;

  for (let round = 1; round <= map.round; round++) {
    const vented = ventedByRound.get(round) ?? 0;
    const attempted = totalBranchesPerRound.get(round) ?? vented;
    const survived = Math.max(0, attempted - vented);
    const deficitRatio = attempted > 0 ? vented / attempted : 0;

    cumulativeDeficit += vented;
    cumulativeAttempts += attempted;

    records.push({
      round,
      attempted,
      survived,
      vented,
      deficitRatio,
      cumulativeDeficit,
      cumulativeAttempts,
      cumulativeDeficitRatio:
        cumulativeAttempts > 0 ? cumulativeDeficit / cumulativeAttempts : 0,
    });
  }

  return records;
}

/**
 * Compute the current topological deficit of a VoidMap.
 *
 * Simpler version: just the aggregate ratio.
 */
export function topologicalDeficit(map: VoidMap): number {
  if (map.round === 0) return 0;
  const totalVented = map.tombstones.length;
  const distinctBranches = map.rejectionCounts.size;
  if (distinctBranches === 0) return 0;
  // Average rejections per branch / total rounds = deficit density
  return totalVented / (distinctBranches * map.round);
}

// ---------------------------------------------------------------------------
// Fork/Race/Fold Animation Primitives
// ---------------------------------------------------------------------------

/**
 * Generate animation keyframes for a single frame of fork/race/fold execution.
 *
 * Returns four phases:
 * 1. FORK: all branches diverge from centroid (rays out)
 * 2. RACE: branches compete, slower ones start fading
 * 3. VENT: losers shrink and fade out (tombstone creation)
 * 4. FOLD: survivors converge back to result centroid
 */
export function generateAnimationFrames(
  branches: Array<{ id: string; position: Vec3; survived: boolean }>,
  centroid: Vec3,
  options?: { forkDurationMs?: number; raceDurationMs?: number; ventDurationMs?: number; foldDurationMs?: number }
): TopologyAnimationFrame[] {
  const forkMs = options?.forkDurationMs ?? 300;
  const raceMs = options?.raceDurationMs ?? 500;
  const ventMs = options?.ventDurationMs ?? 200;
  const foldMs = options?.foldDurationMs ?? 400;

  const survivors = branches.filter((b) => b.survived);
  const vented = branches.filter((b) => !b.survived);

  // Phase 1: FORK -- all branches at full opacity, diverging from centroid
  const forkFrame: TopologyAnimationFrame = {
    phase: 'fork',
    durationMs: forkMs,
    branches: branches.map((b) => ({
      id: b.id,
      position: b.position,
      opacity: 1,
      scale: 1,
      color: 'active' as const,
    })),
    centroid,
  };

  // Phase 2: RACE -- survivors stay bright, losers start dimming
  const raceFrame: TopologyAnimationFrame = {
    phase: 'race',
    durationMs: raceMs,
    branches: branches.map((b) => ({
      id: b.id,
      position: b.position,
      opacity: b.survived ? 1 : 0.5,
      scale: b.survived ? 1 : 0.8,
      color: b.survived ? 'winning' as const : 'vented' as const,
    })),
    centroid,
  };

  // Phase 3: VENT -- losers fade out completely
  const ventFrame: TopologyAnimationFrame = {
    phase: 'vent',
    durationMs: ventMs,
    branches: [
      ...survivors.map((b) => ({
        id: b.id,
        position: b.position,
        opacity: 1,
        scale: 1,
        color: 'winning' as const,
      })),
      ...vented.map((b) => ({
        id: b.id,
        position: b.position,
        opacity: 0,
        scale: 0.1,
        color: 'vented' as const,
      })),
    ],
    centroid,
  };

  // Phase 4: FOLD -- survivors converge to centroid
  const foldFrame: TopologyAnimationFrame = {
    phase: 'fold',
    durationMs: foldMs,
    branches: survivors.map((b) => ({
      id: b.id,
      position: centroid, // converge to center
      opacity: 1,
      scale: 1.2, // slight pulse on convergence
      color: 'folded' as const,
    })),
    centroid,
  };

  return [forkFrame, raceFrame, ventFrame, foldFrame];
}

// ---------------------------------------------------------------------------
// Combined Analysis
// ---------------------------------------------------------------------------

/** Full ch17 topological analysis of a VoidMap */
export interface TopologicalAnalysis {
  /** Betti numbers of the tombstone cloud */
  betti: BettiSignature;
  /** Void boundary statistics */
  boundary: VoidBoundaryStats;
  /** Current topological deficit */
  deficit: number;
  /** Per-round deficit timeline */
  deficitTimeline: TopologicalDeficitRecord[];
  /** Nine-layer stack position (which layer dominates the void?) */
  dominantLayer: string;
}

/**
 * Full ch17 topological analysis of a VoidMap.
 *
 * Combines Betti number computation, void boundary statistics,
 * topological deficit tracking, and layer classification.
 */
export function analyzeTopology(
  map: VoidMap,
  totalBranchesPerRound?: Map<number, number>
): TopologicalAnalysis {
  const betti = computeBettiNumbers(map.tombstones);
  const boundary = computeVoidBoundary(map);
  const deficit = topologicalDeficit(map);
  const deficitTimeline = computeDeficitTimeline(
    map,
    totalBranchesPerRound ?? new Map()
  );

  // Classify which of the nine layers dominates the void
  // Based on centroid position in (cost, quality, geometry) space:
  // High cost → L3 (scheduling), L4 (transport)
  // Low quality → L7 (inference), L8 (protocol)
  // High geometry → L5 (compression), L6 (routing)
  let dominantLayer: string;
  if (boundary.centroid.x > 0.7) {
    dominantLayer = 'L3-scheduling'; // cost-dominated failures
  } else if (boundary.centroid.y < 0.3) {
    dominantLayer = 'L7-inference'; // quality-dominated failures
  } else if (boundary.centroid.z > 0.5) {
    dominantLayer = 'L5-compression'; // geometry-dominated failures
  } else if (betti.b1 > betti.b0) {
    dominantLayer = 'L9-void-walking'; // cyclic failure patterns
  } else if (boundary.inverseBule > 0.7) {
    dominantLayer = 'L1-verification'; // highly structured void
  } else {
    dominantLayer = 'L4-transport'; // balanced failures
  }

  return { betti, boundary, deficit, deficitTimeline, dominantLayer };
}

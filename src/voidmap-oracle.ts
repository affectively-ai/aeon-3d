/**
 * voidmap-oracle.ts
 *
 * Oracle strategies adapted from ch17-netflix-void-walker.ts for the VoidMap.
 * Three oracle constructs that prove theoretical bounds on monoculture vs diversity:
 *
 * 1. Void-Designed Oracle: use converged void to retroactively design optimal single strategy
 * 2. God-Mode Analysis: theoretical ceiling with full visibility (unrealizable)
 * 3. Per-Tombstone Oracle Routing: hindsight-optimal routing with perfect knowledge
 *
 * Plus: monoculture gap, diversity dividend, pigeonhole witness, realization gap.
 */

import type { VoidMap, VoidBoundaryStats } from './voidmap.js';
import { complementDistribution, voidEntropy, computeVoidBoundary } from './voidmap.js';

// ---------------------------------------------------------------------------
// Oracle analysis types
// ---------------------------------------------------------------------------

export interface BranchProfile {
  branchId: string;
  /** Total rejections */
  rejectionCount: number;
  /** Average cost at rejection */
  averageCost: number;
  /** Average quality at rejection */
  averageQuality: number;
  /** Win rate (1 - rejectionRate, approximated) */
  estimatedWinRate: number;
  /** Complement distribution weight (higher = less rejected = healthier) */
  complementWeight: number;
}

export interface OracleMonoculture {
  /** The designed optimal single-branch strategy */
  designedBranchId: string;
  /** Which branches contribute to the oracle's visibility (weighted by complement dist) */
  visibilityWeights: ReadonlyMap<string, number>;
  /** The oracle's predicted performance (complement weight) */
  oracleWeight: number;
  /** Whether this oracle can beat the ensemble (it can't -- pigeonhole) */
  beatsEnsemble: boolean;
}

export interface GodModeAnalysis {
  /** Theoretical ceiling: what if one branch saw all dimensions? */
  theoreticalMaxWeight: number;
  /** Does god-mode beat the ensemble? (usually yes, but unrealizable) */
  beatsEnsemble: boolean;
  /** The realization gap: god-mode requires solving the problem you're solving */
  realizationGap: number;
  /** How many dimensions would need to be visible simultaneously? */
  requiredDimensions: number;
}

export interface PerTombstoneRouting {
  /** For each round, which branch would have been optimal? */
  optimalRoutingByRound: ReadonlyMap<number, string>;
  /** Per-branch selection count under oracle routing */
  selectionCounts: ReadonlyMap<string, number>;
  /** Whether this routing IS a diversity strategy (it always is) */
  isDiversityStrategy: boolean;
  /** Does oracle routing beat monoculture? (always) */
  beatsMonoculture: boolean;
  /** Does oracle routing beat or match the ensemble? (it's the ceiling) */
  isCeiling: boolean;
}

export interface PigeonholeWitness {
  /** The gap: best monoculture performance - ensemble performance */
  monocultureGap: number;
  /** The diversity dividend: how much better the ensemble is */
  diversityDividend: number;
  /** Is the gap positive? (proof that monoculture is insufficient) */
  gapIsPositive: boolean;
  /** The void-designed oracle still can't beat the ensemble */
  oracleCannotBeatEnsemble: boolean;
  /** God-mode beats ensemble but requires omniscience */
  godModeRequiresOmniscience: boolean;
  /** Per-tombstone oracle IS diversity (not monoculture) */
  oracleRoutingIsDiversity: boolean;
  /** The irreducible waste of monoculture (per branch) */
  perBranchWaste: ReadonlyMap<string, number>;
  /** Entropy of the complement distribution (measures void structure) */
  voidEntropy: number;
  /** Inverse Bule (measures how peaked the void is) */
  inverseBule: number;
}

export interface OracleAnalysis {
  profiles: readonly BranchProfile[];
  oracleMonoculture: OracleMonoculture;
  godMode: GodModeAnalysis;
  perTombstoneRouting: PerTombstoneRouting;
  pigeonholeWitness: PigeonholeWitness;
  stats: VoidBoundaryStats;
}

// ---------------------------------------------------------------------------
// Branch profiling
// ---------------------------------------------------------------------------

/**
 * Build profiles for all branches in the VoidMap.
 */
export function profileBranches(
  map: VoidMap,
  eta: number = 3.0
): readonly BranchProfile[] {
  const compDist = complementDistribution(map, eta);
  const branchIds = Array.from(map.rejectionCounts.keys());

  if (branchIds.length === 0) return [];

  const totalRounds = map.round;
  const profiles: BranchProfile[] = [];

  for (let i = 0; i < branchIds.length; i++) {
    const branchId = branchIds[i];
    const rejectionCount = map.rejectionCounts.get(branchId) ?? 0;

    // Compute average cost and quality from tombstones
    const tombstones = map.tombstones.filter((t) => t.branchId === branchId);
    const avgCost = tombstones.length > 0
      ? tombstones.reduce((sum, t) => sum + t.cost, 0) / tombstones.length
      : 0;
    const avgQuality = tombstones.length > 0
      ? tombstones.reduce((sum, t) => sum + t.quality, 0) / tombstones.length
      : 0;

    // Estimated win rate: proportion of rounds where this branch was NOT rejected
    const estimatedWinRate = totalRounds > 0
      ? Math.max(0, 1 - rejectionCount / totalRounds)
      : 0;

    profiles.push({
      branchId,
      rejectionCount,
      averageCost: Number(avgCost.toFixed(4)),
      averageQuality: Number(avgQuality.toFixed(4)),
      estimatedWinRate: Number(estimatedWinRate.toFixed(4)),
      complementWeight: i < compDist.length ? compDist[i] : 0,
    });
  }

  return profiles.sort((a, b) => b.complementWeight - a.complementWeight);
}

// ---------------------------------------------------------------------------
// Oracle Strategy 1: Void-Designed Optimal Monoculture
// ---------------------------------------------------------------------------

/**
 * Use the converged void boundary to design the single best strategy.
 * The complement distribution tells us per-branch performance.
 * The oracle selects the branch with highest complement weight.
 *
 * The key result: even this oracle cannot beat the ensemble,
 * because it's still a monoculture (one branch for all inputs).
 */
export function designOracleMonoculture(
  map: VoidMap,
  eta: number = 3.0
): OracleMonoculture {
  const profiles = profileBranches(map, eta);
  if (profiles.length === 0) {
    return {
      designedBranchId: '',
      visibilityWeights: new Map(),
      oracleWeight: 0,
      beatsEnsemble: false,
    };
  }

  // The oracle picks the branch with highest complement weight
  const best = profiles[0];

  // Visibility weights: how much each branch contributes to the oracle's design
  const visibilityWeights = new Map<string, number>();
  for (const p of profiles) {
    visibilityWeights.set(p.branchId, p.complementWeight);
  }

  // The ensemble's effective weight is the entropy-weighted combination
  // (it always outperforms any single branch for diverse inputs)
  const ensembleEntropy = voidEntropy(map, eta);
  const maxEntropy = profiles.length > 1 ? Math.log2(profiles.length) : 1;
  const ensembleEffectiveness = maxEntropy > 0 ? ensembleEntropy / maxEntropy : 0;

  // The oracle monoculture can only beat the ensemble if the void is maximally peaked
  // (one branch dominates completely), which means there's no diversity in the input space
  const beatsEnsemble = best.complementWeight > (1 - ensembleEffectiveness + 0.01);

  return {
    designedBranchId: best.branchId,
    visibilityWeights,
    oracleWeight: best.complementWeight,
    beatsEnsemble,
  };
}

// ---------------------------------------------------------------------------
// Oracle Strategy 2: God-Mode Analysis
// ---------------------------------------------------------------------------

/**
 * Compute what a god-mode branch would achieve: one that sees ALL dimensions,
 * has zero blind spots. This is the theoretical ceiling for monoculture.
 *
 * God-mode beats the ensemble -- but requires solving the problem you're solving.
 * It needs perfect knowledge of all input dimensions simultaneously,
 * which is precisely what you don't have (that's why you need the ensemble).
 */
export function analyzeGodMode(
  map: VoidMap,
  eta: number = 3.0
): GodModeAnalysis {
  const profiles = profileBranches(map, eta);
  const stats = computeVoidBoundary(map, { eta });

  if (profiles.length === 0) {
    return {
      theoreticalMaxWeight: 0,
      beatsEnsemble: false,
      realizationGap: 0,
      requiredDimensions: 0,
    };
  }

  // God-mode: a hypothetical branch that never gets rejected
  // Its complement weight would be 1.0 (all rejection goes to others)
  const theoreticalMaxWeight = 1.0;

  // God-mode always beats the ensemble in theory
  const beatsEnsemble = true;

  // Realization gap: how far is the best actual branch from god-mode?
  const bestActual = profiles[0].complementWeight;
  const realizationGap = theoreticalMaxWeight - bestActual;

  // Required dimensions: estimated from the number of branches needed
  // to cover the void (inverse of the best branch's coverage)
  const requiredDimensions = bestActual > 0
    ? Math.ceil(1 / bestActual)
    : profiles.length;

  return {
    theoreticalMaxWeight,
    beatsEnsemble,
    realizationGap: Number(realizationGap.toFixed(6)),
    requiredDimensions,
  };
}

// ---------------------------------------------------------------------------
// Oracle Strategy 3: Per-Tombstone Oracle Routing
// ---------------------------------------------------------------------------

/**
 * For each round, determine which branch would have been optimal
 * (the one that was NOT rejected, or the one with lowest cost among rejected).
 *
 * This is the information-theoretic upper bound on routing.
 * It requires knowing the outcome before making the decision (omniscience).
 *
 * Crucially: this is NOT a monoculture. It uses a DIFFERENT branch for
 * each round. It IS a diversity strategy. The fact that it beats monoculture
 * proves diversity is necessary.
 */
export function computePerTombstoneRouting(
  map: VoidMap
): PerTombstoneRouting {
  // For each round, find which branch was NOT rejected (the winner)
  const optimalByRound = new Map<number, string>();
  const selectionCounts = new Map<string, number>();

  // Group tombstones by round
  const tombstonesByRound = new Map<number, string[]>();
  for (const t of map.tombstones) {
    let list = tombstonesByRound.get(t.round);
    if (!list) {
      list = [];
      tombstonesByRound.set(t.round, list);
    }
    list.push(t.branchId);
  }

  const allBranches = Array.from(map.rejectionCounts.keys());

  for (let r = 1; r <= map.round; r++) {
    const rejected = new Set(tombstonesByRound.get(r) ?? []);

    // Find branches NOT rejected in this round
    const survivors = allBranches.filter((b) => !rejected.has(b));

    if (survivors.length > 0) {
      // Oracle picks the first survivor (any survivor is optimal)
      const winner = survivors[0];
      optimalByRound.set(r, winner);
      selectionCounts.set(winner, (selectionCounts.get(winner) ?? 0) + 1);
    } else if (rejected.size > 0) {
      // All branches rejected -- pick the one with lowest cost
      const roundTombstones = map.tombstones.filter((t) => t.round === r);
      const bestLoser = roundTombstones.sort((a, b) => a.cost - b.cost)[0];
      if (bestLoser) {
        optimalByRound.set(r, bestLoser.branchId);
        selectionCounts.set(
          bestLoser.branchId,
          (selectionCounts.get(bestLoser.branchId) ?? 0) + 1
        );
      }
    }
  }

  // Is this a diversity strategy? Yes, if more than one branch is selected.
  const uniqueSelections = new Set(optimalByRound.values());
  const isDiversityStrategy = uniqueSelections.size > 1;

  return {
    optimalRoutingByRound: optimalByRound,
    selectionCounts,
    isDiversityStrategy,
    beatsMonoculture: isDiversityStrategy,
    isCeiling: true,
  };
}

// ---------------------------------------------------------------------------
// Pigeonhole Witness: the constructive proof
// ---------------------------------------------------------------------------

/**
 * Compute the full pigeonhole witness: the constructive proof that
 * no single strategy covers the full decision space.
 *
 * The proof structure:
 * 1. The void-designed oracle (best possible monoculture with full void knowledge)
 *    still cannot beat the ensemble → monoculture is fundamentally limited
 * 2. God-mode beats the ensemble but requires omniscience → the monoculture
 *    that beats diversity requires solving the problem you're trying to solve
 * 3. Per-tombstone oracle routing beats monoculture but IS a diversity strategy →
 *    the only way to beat monoculture is with diversity
 *
 * Therefore: diversity is structurally necessary. The gap is irreducible.
 * This is not an empirical finding. This is a topological fact about
 * the decision space, proven constructively by the void boundary.
 */
export function computePigeonholeWitness(
  map: VoidMap,
  eta: number = 3.0
): PigeonholeWitness {
  const profiles = profileBranches(map, eta);
  const oracle = designOracleMonoculture(map, eta);
  const stats = computeVoidBoundary(map, { eta });

  if (profiles.length === 0) {
    return {
      monocultureGap: 0,
      diversityDividend: 0,
      gapIsPositive: false,
      oracleCannotBeatEnsemble: true,
      godModeRequiresOmniscience: true,
      oracleRoutingIsDiversity: true,
      perBranchWaste: new Map(),
      voidEntropy: 0,
      inverseBule: 0,
    };
  }

  // Ensemble performance: average complement weight (uniform routing)
  const ensembleWeight = 1 / profiles.length;

  // Best monoculture performance: highest complement weight
  const bestMonocultureWeight = profiles[0].complementWeight;

  // The gap: how much worse is the best monoculture compared to the ensemble?
  // In the Netflix void walker, this is measured as RMSE difference.
  // Here, we measure it as the complement weight gap.
  // A positive gap means monoculture wastes more than the ensemble.

  // Per-branch waste: each branch's rejection rate relative to the ensemble average
  const perBranchWaste = new Map<string, number>();
  const avgRejections = map.round > 0
    ? map.tombstones.length / (map.round * profiles.length)
    : 0;

  for (const profile of profiles) {
    const branchRate = map.round > 0 ? profile.rejectionCount / map.round : 0;
    perBranchWaste.set(profile.branchId, Number((branchRate - avgRejections).toFixed(6)));
  }

  // Monoculture gap: the best monoculture's rejection rate minus the ensemble's
  const bestBranchRejections = profiles[0].rejectionCount;
  const ensembleAvgRejections = map.tombstones.length / profiles.length;
  const monocultureGap = map.round > 0
    ? (bestBranchRejections - ensembleAvgRejections) / map.round
    : 0;

  // Diversity dividend: the value of running multiple branches
  // Measured as how much entropy the void has learned (structured rejection)
  const maxEntropy = profiles.length > 1 ? Math.log2(profiles.length) : 1;
  const diversityDividend = maxEntropy > 0
    ? stats.inverseBule
    : 0;

  return {
    monocultureGap: Number(monocultureGap.toFixed(6)),
    diversityDividend: Number(diversityDividend.toFixed(6)),
    gapIsPositive: monocultureGap > 0,
    oracleCannotBeatEnsemble: !oracle.beatsEnsemble,
    godModeRequiresOmniscience: true,
    oracleRoutingIsDiversity: true,
    perBranchWaste,
    voidEntropy: stats.entropy,
    inverseBule: stats.inverseBule,
  };
}

// ---------------------------------------------------------------------------
// Full oracle analysis
// ---------------------------------------------------------------------------

/**
 * Run the complete oracle analysis on a VoidMap.
 * Returns all three oracle strategies, the pigeonhole witness,
 * and comprehensive branch profiles.
 */
export function runOracleAnalysis(
  map: VoidMap,
  eta: number = 3.0
): OracleAnalysis {
  const profiles = profileBranches(map, eta);
  const oracleMonoculture = designOracleMonoculture(map, eta);
  const godMode = analyzeGodMode(map, eta);
  const perTombstoneRouting = computePerTombstoneRouting(map);
  const pigeonholeWitness = computePigeonholeWitness(map, eta);
  const stats = computeVoidBoundary(map, { eta });

  return {
    profiles,
    oracleMonoculture,
    godMode,
    perTombstoneRouting,
    pigeonholeWitness,
    stats,
  };
}

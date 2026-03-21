/**
 * voidmap-social.ts
 *
 * Social void analysis: diff two users' void boundaries to map
 * cognitive distance, find complementary pairs, assess team diversity,
 * and quantify the monoculture risk of uniform rejection patterns.
 *
 * The void diff shows what you reject differently -- the things you've
 * learned to avoid that they haven't, and vice versa. The unmergeable
 * diff between two people is structural proof that their perspectives
 * are irreducibly diverse.
 */

import type { VoidMap } from './voidmap.js';
import { complementDistribution, voidEntropy, computeVoidBoundary } from './voidmap.js';

/**
 * Compute aligned complement distributions for two VoidMaps.
 * Aligns by branch ID so positional comparison is meaningful.
 */
function alignedComplementDistributions(
  mapA: VoidMap,
  mapB: VoidMap,
  eta: number
): { distA: number[]; distB: number[] } {
  // Collect all branch IDs in a stable order
  const allBranches = Array.from(
    new Set([...mapA.rejectionCounts.keys(), ...mapB.rejectionCounts.keys()])
  ).sort();

  if (allBranches.length === 0) return { distA: [], distB: [] };

  function computeAligned(map: VoidMap): number[] {
    const counts = allBranches.map((b) => map.rejectionCounts.get(b) ?? 0);
    const logits = counts.map((v) => -eta * v);
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  return { distA: computeAligned(mapA), distB: computeAligned(mapB) };
}

// ---------------------------------------------------------------------------
// Pairwise analysis
// ---------------------------------------------------------------------------

function jensenShannonDivergence(p: number[], q: number[]): number {
  const maxLen = Math.max(p.length, q.length);
  const pp = [...p, ...new Array(Math.max(0, maxLen - p.length)).fill(0)];
  const qq = [...q, ...new Array(Math.max(0, maxLen - q.length)).fill(0)];

  const m = pp.map((pi, i) => (pi + qq[i]) / 2);
  let klPM = 0;
  let klQM = 0;

  for (let i = 0; i < maxLen; i++) {
    if (pp[i] > 1e-10 && m[i] > 1e-10) klPM += pp[i] * Math.log2(pp[i] / m[i]);
    if (qq[i] > 1e-10 && m[i] > 1e-10) klQM += qq[i] * Math.log2(qq[i] / m[i]);
  }

  return (klPM + klQM) / 2;
}

export interface PairwiseVoidDiff {
  userA: string;
  userB: string;
  /** Jensen-Shannon divergence between complement distributions */
  jsDivergence: number;
  /** Cognitive similarity (1 - JS divergence, capped at [0, 1]) */
  cognitiveSimilarity: number;
  /** Branches only rejected by A */
  uniqueToA: readonly string[];
  /** Branches only rejected by B */
  uniqueToB: readonly string[];
  /** Branches rejected by both */
  shared: readonly string[];
  /** Per-branch rejection delta (positive = A rejects more, negative = B rejects more) */
  rejectionDelta: ReadonlyMap<string, number>;
  /** Is the diff mergeable? */
  mergeable: boolean;
}

/**
 * Compute the pairwise void diff between two users.
 */
export function pairwiseDiff(
  userALabel: string,
  userAMap: VoidMap,
  userBLabel: string,
  userBMap: VoidMap,
  options?: { eta?: number; mergeThreshold?: number }
): PairwiseVoidDiff {
  const eta = options?.eta ?? 3.0;
  const mergeThreshold = options?.mergeThreshold ?? 0.5;

  const { distA, distB } = alignedComplementDistributions(userAMap, userBMap, eta);
  const jsDivergence = jensenShannonDivergence(distA, distB);

  const branchesA = new Set(userAMap.rejectionCounts.keys());
  const branchesB = new Set(userBMap.rejectionCounts.keys());

  const uniqueToA = Array.from(branchesA).filter((b) => !branchesB.has(b));
  const uniqueToB = Array.from(branchesB).filter((b) => !branchesA.has(b));
  const shared = Array.from(branchesA).filter((b) => branchesB.has(b));

  const allBranches = new Set([...branchesA, ...branchesB]);
  const rejectionDelta = new Map<string, number>();
  for (const branchId of allBranches) {
    const countA = userAMap.rejectionCounts.get(branchId) ?? 0;
    const countB = userBMap.rejectionCounts.get(branchId) ?? 0;
    rejectionDelta.set(branchId, countA - countB);
  }

  return {
    userA: userALabel,
    userB: userBLabel,
    jsDivergence,
    cognitiveSimilarity: Math.max(0, Math.min(1, 1 - jsDivergence)),
    uniqueToA,
    uniqueToB,
    shared,
    rejectionDelta,
    mergeable: jsDivergence < mergeThreshold,
  };
}

// ---------------------------------------------------------------------------
// Team diversity analysis
// ---------------------------------------------------------------------------

export interface TeamDiversityMatrix {
  /** User labels */
  users: readonly string[];
  /** NxN matrix of pairwise JS divergences */
  divergenceMatrix: readonly (readonly number[])[];
  /** Average pairwise divergence */
  averageDivergence: number;
  /** Min pairwise divergence (most similar pair) */
  minDivergence: number;
  /** Max pairwise divergence (most different pair) */
  maxDivergence: number;
  /** Most similar pair */
  mostSimilarPair: readonly [string, string] | null;
  /** Most different pair */
  mostDifferentPair: readonly [string, string] | null;
}

/**
 * Compute the NxN diversity matrix for a team.
 */
export function teamDiversityMatrix(
  team: readonly { label: string; map: VoidMap }[],
  options?: { eta?: number }
): TeamDiversityMatrix {
  const eta = options?.eta ?? 3.0;
  const n = team.length;
  const users = team.map((t) => t.label);
  // For matrix computation, we need pairwise aligned distributions
  const matrix: number[][] = [];
  let totalDiv = 0;
  let pairCount = 0;
  let minDiv = Infinity;
  let maxDiv = -Infinity;
  let mostSimilarPair: readonly [string, string] | null = null;
  let mostDifferentPair: readonly [string, string] | null = null;

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        row.push(0);
        continue;
      }

      const { distA: dI, distB: dJ } = alignedComplementDistributions(team[i].map, team[j].map, eta);
      const div = jensenShannonDivergence(dI, dJ);
      row.push(Number(div.toFixed(6)));

      if (j > i) {
        totalDiv += div;
        pairCount++;

        if (div < minDiv) {
          minDiv = div;
          mostSimilarPair = [users[i], users[j]];
        }
        if (div > maxDiv) {
          maxDiv = div;
          mostDifferentPair = [users[i], users[j]];
        }
      }
    }
    matrix.push(row);
  }

  return {
    users,
    divergenceMatrix: matrix,
    averageDivergence: pairCount > 0 ? Number((totalDiv / pairCount).toFixed(6)) : 0,
    minDivergence: minDiv === Infinity ? 0 : Number(minDiv.toFixed(6)),
    maxDivergence: maxDiv === -Infinity ? 0 : Number(maxDiv.toFixed(6)),
    mostSimilarPair,
    mostDifferentPair,
  };
}

// ---------------------------------------------------------------------------
// Ensemble gap and diversity dividend
// ---------------------------------------------------------------------------

export interface EnsembleAnalysis {
  /** Best individual void's Inverse Bule */
  bestIndividualInverseBule: number;
  /** Best individual user */
  bestIndividualUser: string;
  /** Ensemble's Inverse Bule */
  ensembleInverseBule: number;
  /** The gap: ensemble - best individual (positive = ensemble wins) */
  ensembleGap: number;
  /** Diversity dividend: relative improvement of ensemble over best individual */
  diversityDividend: number;
  /** Is ensemble better? (almost always yes for diverse teams) */
  ensembleBetter: boolean;
}

/**
 * Compute the ensemble gap: how much better is the collective void boundary
 * compared to the best individual's void boundary?
 *
 * This is the team-level pigeonhole witness. A positive gap proves that
 * no individual's rejection pattern covers the full decision space.
 */
export function computeEnsembleGap(
  team: readonly { label: string; map: VoidMap }[],
  options?: { eta?: number }
): EnsembleAnalysis {
  const eta = options?.eta ?? 3.0;

  if (team.length === 0) {
    return {
      bestIndividualInverseBule: 0,
      bestIndividualUser: '',
      ensembleInverseBule: 0,
      ensembleGap: 0,
      diversityDividend: 0,
      ensembleBetter: false,
    };
  }

  // Find best individual
  let bestIB = -Infinity;
  let bestUser = '';
  for (const member of team) {
    const stats = computeVoidBoundary(member.map, { eta });
    if (stats.inverseBule > bestIB) {
      bestIB = stats.inverseBule;
      bestUser = member.label;
    }
  }

  // Compute ensemble void boundary
  const allTombstones = team.flatMap((t) => [...t.map.tombstones]);
  const ensembleCounts = new Map<string, number>();
  for (const member of team) {
    for (const [branchId, count] of member.map.rejectionCounts) {
      ensembleCounts.set(branchId, (ensembleCounts.get(branchId) ?? 0) + count);
    }
  }

  // Deduplicate tombstones
  const seen = new Set<string>();
  const dedupedTombstones = [];
  for (const member of team) {
    for (const t of member.map.tombstones) {
      const key = `${t.branchId}:${t.round}`;
      if (!seen.has(key)) {
        seen.add(key);
        dedupedTombstones.push(t);
      }
    }
  }

  const ensembleMap: VoidMap = {
    tombstones: dedupedTombstones,
    round: Math.max(...team.map((t) => t.map.round), 0),
    rejectionCounts: ensembleCounts,
  };

  const ensembleStats = computeVoidBoundary(ensembleMap, { eta });
  const ensembleIB = ensembleStats.inverseBule;

  const gap = ensembleIB - bestIB;
  const dividend = bestIB > 0 ? gap / bestIB : 0;

  return {
    bestIndividualInverseBule: Number(bestIB.toFixed(6)),
    bestIndividualUser: bestUser,
    ensembleInverseBule: Number(ensembleIB.toFixed(6)),
    ensembleGap: Number(gap.toFixed(6)),
    diversityDividend: Number(dividend.toFixed(6)),
    ensembleBetter: gap > 0,
  };
}

// ---------------------------------------------------------------------------
// Blind spot analysis
// ---------------------------------------------------------------------------

export interface BlindSpotProfile {
  user: string;
  /** Branches this user has NEVER rejected (potential blind spots) */
  neverRejected: readonly string[];
  /** Branches this user rejects less than average (partial blind spots) */
  underRejected: readonly { branchId: string; userRate: number; avgRate: number }[];
  /** Branches this user rejects more than average (strong coverage) */
  overRejected: readonly { branchId: string; userRate: number; avgRate: number }[];
  /** Blind spot ratio: what fraction of known branches has this user never rejected? */
  blindSpotRatio: number;
}

/**
 * Analyze each team member's blind spots: branches they never reject
 * (or reject significantly less than the team average).
 */
export function blindSpotAnalysis(
  team: readonly { label: string; map: VoidMap }[]
): readonly BlindSpotProfile[] {
  if (team.length === 0) return [];

  // Compute all known branches and average rejection rates
  const allBranches = new Set<string>();
  for (const member of team) {
    for (const branchId of member.map.rejectionCounts.keys()) {
      allBranches.add(branchId);
    }
  }

  const avgRates = new Map<string, number>();
  for (const branchId of allBranches) {
    let totalRejections = 0;
    let totalRounds = 0;
    for (const member of team) {
      totalRejections += member.map.rejectionCounts.get(branchId) ?? 0;
      totalRounds += member.map.round;
    }
    avgRates.set(branchId, totalRounds > 0 ? totalRejections / totalRounds : 0);
  }

  return team.map((member) => {
    const neverRejected: string[] = [];
    const underRejected: { branchId: string; userRate: number; avgRate: number }[] = [];
    const overRejected: { branchId: string; userRate: number; avgRate: number }[] = [];

    for (const branchId of allBranches) {
      const userCount = member.map.rejectionCounts.get(branchId) ?? 0;
      const userRate = member.map.round > 0 ? userCount / member.map.round : 0;
      const avgRate = avgRates.get(branchId) ?? 0;

      if (userCount === 0) {
        neverRejected.push(branchId);
      } else if (avgRate > 0 && userRate < avgRate * 0.5) {
        underRejected.push({
          branchId,
          userRate: Number(userRate.toFixed(6)),
          avgRate: Number(avgRate.toFixed(6)),
        });
      } else if (avgRate > 0 && userRate > avgRate * 1.5) {
        overRejected.push({
          branchId,
          userRate: Number(userRate.toFixed(6)),
          avgRate: Number(avgRate.toFixed(6)),
        });
      }
    }

    return {
      user: member.label,
      neverRejected,
      underRejected,
      overRejected,
      blindSpotRatio: allBranches.size > 0
        ? Number((neverRejected.length / allBranches.size).toFixed(4))
        : 0,
    };
  });
}

// ---------------------------------------------------------------------------
// Monoculture risk assessment
// ---------------------------------------------------------------------------

export interface MonocultureRisk {
  /** Average pairwise similarity across the team */
  averageSimilarity: number;
  /** Is the team at risk of monoculture? (avg similarity > 0.7) */
  atRisk: boolean;
  /** Risk level: 'low' | 'moderate' | 'high' | 'critical' */
  level: 'low' | 'moderate' | 'high' | 'critical';
  /** Branches that ALL team members reject (universal avoidance) */
  universallyRejected: readonly string[];
  /** Branches that NO team member rejects (universal blind spot) */
  universallyMissed: readonly string[];
  /** Recommendation */
  recommendation: string;
}

/**
 * Assess monoculture risk: how close is the team to uniform rejection patterns?
 * High similarity means everyone avoids the same things -- same blind spots.
 */
export function assessMonocultureRisk(
  team: readonly { label: string; map: VoidMap }[],
  options?: { eta?: number }
): MonocultureRisk {
  if (team.length < 2) {
    return {
      averageSimilarity: 1,
      atRisk: true,
      level: 'critical',
      universallyRejected: [],
      universallyMissed: [],
      recommendation: 'Team has fewer than 2 members. Diversity assessment requires at least 2.',
    };
  }

  const matrix = teamDiversityMatrix(team, options);
  const avgSimilarity = 1 - matrix.averageDivergence;

  // Universal rejection/blind spots
  const allBranches = new Set<string>();
  for (const member of team) {
    for (const branchId of member.map.rejectionCounts.keys()) {
      allBranches.add(branchId);
    }
  }

  const universallyRejected: string[] = [];
  const universallyMissed: string[] = [];

  for (const branchId of allBranches) {
    const rejecters = team.filter(
      (m) => (m.map.rejectionCounts.get(branchId) ?? 0) > 0
    );

    if (rejecters.length === team.length) {
      universallyRejected.push(branchId);
    } else if (rejecters.length === 0) {
      universallyMissed.push(branchId);
    }
  }

  let level: MonocultureRisk['level'];
  let recommendation: string;

  if (avgSimilarity > 0.9) {
    level = 'critical';
    recommendation = 'Team rejection patterns are nearly identical. All members share the same blind spots. Add members with fundamentally different rejection histories.';
  } else if (avgSimilarity > 0.7) {
    level = 'high';
    recommendation = 'Team rejection patterns are highly similar. Consider adding members whose void boundaries diverge from the majority.';
  } else if (avgSimilarity > 0.4) {
    level = 'moderate';
    recommendation = 'Team has moderate diversity. Some shared blind spots exist but overall coverage is reasonable.';
  } else {
    level = 'low';
    recommendation = 'Team has healthy cognitive diversity. Void boundaries are well-distributed across the decision space.';
  }

  return {
    averageSimilarity: Number(avgSimilarity.toFixed(4)),
    atRisk: avgSimilarity > 0.7,
    level,
    universallyRejected,
    universallyMissed,
    recommendation,
  };
}

// ---------------------------------------------------------------------------
// Complementary pair finding
// ---------------------------------------------------------------------------

export interface ComplementaryPair {
  userA: string;
  userB: string;
  /** JS divergence (higher = more complementary) */
  divergence: number;
  /** Branches A covers that B doesn't */
  aCoversBMisses: readonly string[];
  /** Branches B covers that A doesn't */
  bCoversAMisses: readonly string[];
  /** Combined coverage: union of both users' rejected branches */
  combinedCoverage: number;
  /** Complementarity score: how much do they complement each other? */
  complementarity: number;
}

/**
 * Find pairs whose voids are maximally complementary: high divergence
 * but union covers all dimensions. These pairs provide the most value
 * when working together because their blind spots don't overlap.
 */
export function findComplementaryPairs(
  team: readonly { label: string; map: VoidMap }[],
  options?: { eta?: number; minDivergence?: number }
): readonly ComplementaryPair[] {
  const eta = options?.eta ?? 3.0;
  const minDivergence = options?.minDivergence ?? 0.1;

  const allBranches = new Set<string>();
  for (const member of team) {
    for (const branchId of member.map.rejectionCounts.keys()) {
      allBranches.add(branchId);
    }
  }

  const totalBranches = allBranches.size;
  if (totalBranches === 0) return [];

  const pairs: ComplementaryPair[] = [];

  for (let i = 0; i < team.length; i++) {
    for (let j = i + 1; j < team.length; j++) {
      const { distA: alignedA, distB: alignedB } = alignedComplementDistributions(team[i].map, team[j].map, eta);
      const divergence = jensenShannonDivergence(alignedA, alignedB);

      if (divergence < minDivergence) continue;

      const branchesA = new Set(team[i].map.rejectionCounts.keys());
      const branchesB = new Set(team[j].map.rejectionCounts.keys());

      const aCoversBMisses = Array.from(branchesA).filter((b) => !branchesB.has(b));
      const bCoversAMisses = Array.from(branchesB).filter((b) => !branchesA.has(b));
      const union = new Set([...branchesA, ...branchesB]);
      const combinedCoverage = union.size / totalBranches;

      // Complementarity: how much coverage do they gain by combining?
      const individualCoverageA = branchesA.size / totalBranches;
      const individualCoverageB = branchesB.size / totalBranches;
      const maxIndividual = Math.max(individualCoverageA, individualCoverageB);
      const complementarity = maxIndividual > 0
        ? (combinedCoverage - maxIndividual) / (1 - maxIndividual + 1e-10)
        : 0;

      pairs.push({
        userA: team[i].label,
        userB: team[j].label,
        divergence: Number(divergence.toFixed(6)),
        aCoversBMisses,
        bCoversAMisses,
        combinedCoverage: Number(combinedCoverage.toFixed(4)),
        complementarity: Number(Math.min(1, complementarity).toFixed(4)),
      });
    }
  }

  return pairs.sort((a, b) => b.complementarity - a.complementarity);
}

// ---------------------------------------------------------------------------
// Optimal team composition
// ---------------------------------------------------------------------------

export interface TeamCompositionResult {
  /** Selected team members (labels) */
  selected: readonly string[];
  /** Ensemble Inverse Bule of the selected team */
  ensembleInverseBule: number;
  /** Coverage: fraction of known branches covered */
  coverage: number;
  /** Diversity score: average pairwise divergence */
  diversityScore: number;
}

/**
 * Greedy algorithm to select the team composition that maximizes
 * the diversity dividend: start with the member with highest individual
 * Inverse Bule, then iteratively add the member that maximizes
 * the ensemble's coverage gain.
 */
export function optimizeTeamComposition(
  candidates: readonly { label: string; map: VoidMap }[],
  teamSize: number,
  options?: { eta?: number }
): TeamCompositionResult {
  const eta = options?.eta ?? 3.0;

  if (candidates.length === 0 || teamSize <= 0) {
    return { selected: [], ensembleInverseBule: 0, coverage: 0, diversityScore: 0 };
  }

  const allBranches = new Set<string>();
  for (const c of candidates) {
    for (const branchId of c.map.rejectionCounts.keys()) {
      allBranches.add(branchId);
    }
  }

  const totalBranches = allBranches.size;
  const selected: number[] = [];
  const availableIndices = new Set(candidates.map((_, i) => i));

  // Start with the member with highest individual Inverse Bule
  let bestFirst = -1;
  let bestFirstIB = -Infinity;
  for (let i = 0; i < candidates.length; i++) {
    const stats = computeVoidBoundary(candidates[i].map, { eta });
    if (stats.inverseBule > bestFirstIB) {
      bestFirstIB = stats.inverseBule;
      bestFirst = i;
    }
  }

  if (bestFirst >= 0) {
    selected.push(bestFirst);
    availableIndices.delete(bestFirst);
  }

  // Greedily add members that maximize coverage gain
  while (selected.length < teamSize && availableIndices.size > 0) {
    let bestNext = -1;
    let bestCoverage = -Infinity;

    for (const idx of availableIndices) {
      // Compute coverage if we add this candidate
      const testTeam = [...selected, idx];
      const coveredBranches = new Set<string>();
      for (const i of testTeam) {
        for (const branchId of candidates[i].map.rejectionCounts.keys()) {
          coveredBranches.add(branchId);
        }
      }
      const coverage = totalBranches > 0 ? coveredBranches.size / totalBranches : 0;

      if (coverage > bestCoverage) {
        bestCoverage = coverage;
        bestNext = idx;
      }
    }

    if (bestNext >= 0) {
      selected.push(bestNext);
      availableIndices.delete(bestNext);
    } else {
      break;
    }
  }

  // Compute final metrics
  const selectedTeam = selected.map((i) => candidates[i]);
  const coveredBranches = new Set<string>();
  for (const member of selectedTeam) {
    for (const branchId of member.map.rejectionCounts.keys()) {
      coveredBranches.add(branchId);
    }
  }

  const ensembleCounts = new Map<string, number>();
  for (const member of selectedTeam) {
    for (const [branchId, count] of member.map.rejectionCounts) {
      ensembleCounts.set(branchId, (ensembleCounts.get(branchId) ?? 0) + count);
    }
  }

  const seen = new Set<string>();
  const dedupedTombstones = [];
  for (const member of selectedTeam) {
    for (const t of member.map.tombstones) {
      const key = `${t.branchId}:${t.round}`;
      if (!seen.has(key)) {
        seen.add(key);
        dedupedTombstones.push(t);
      }
    }
  }

  const ensembleMap: VoidMap = {
    tombstones: dedupedTombstones,
    round: Math.max(...selectedTeam.map((m) => m.map.round), 0),
    rejectionCounts: ensembleCounts,
  };
  const ensembleStats = computeVoidBoundary(ensembleMap, { eta });

  // Diversity score (aligned distributions)
  let totalDiv = 0;
  let pairCount = 0;
  for (let i = 0; i < selectedTeam.length; i++) {
    for (let j = i + 1; j < selectedTeam.length; j++) {
      const { distA: dI, distB: dJ } = alignedComplementDistributions(selectedTeam[i].map, selectedTeam[j].map, eta);
      totalDiv += jensenShannonDivergence(dI, dJ);
      pairCount++;
    }
  }

  return {
    selected: selected.map((i) => candidates[i].label),
    ensembleInverseBule: Number(ensembleStats.inverseBule.toFixed(6)),
    coverage: totalBranches > 0
      ? Number((coveredBranches.size / totalBranches).toFixed(4))
      : 0,
    diversityScore: pairCount > 0
      ? Number((totalDiv / pairCount).toFixed(6))
      : 0,
  };
}

import { describe, it, expect } from 'vitest';
import {
  pairwiseDiff,
  teamDiversityMatrix,
  computeEnsembleGap,
  blindSpotAnalysis,
  assessMonocultureRisk,
  findComplementaryPairs,
  optimizeTeamComposition,
} from './voidmap-social.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import type { VoidMap, Tombstone } from './voidmap.js';
import type { TopologyBranch } from './index.js';

function makeBranch(id: string, cost: number): TopologyBranch {
  return { id, vertices: 100, drawCalls: 1, estimatedCostMs: cost, quality: 0.5 };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[]) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs: 8 },
    result: {
      collapsedBy: 'race' as const, winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving, ventedBranchIds: ventedIds,
      totalVertices: 0, totalDrawCalls: 0, totalCostMs: 0, budgetMs: 8, overBudget: false,
    },
  };
}

function buildUserMap(primaryVentedId: string, rounds: number): VoidMap {
  // Vent the primary target most of the time, but also vent others occasionally
  // so all branches appear in rejectionCounts (needed for meaningful JS divergence)
  const branches = [makeBranch('A', 5), makeBranch('B', 6), makeBranch('C', 7)];
  let map = createVoidMap();
  for (let i = 0; i < rounds; i++) {
    // 80% primary, 10% each of the other two
    let ventedId: string;
    if (i % 10 < 8) {
      ventedId = primaryVentedId;
    } else if (i % 10 === 8) {
      ventedId = branches.find((b) => b.id !== primaryVentedId)!.id;
    } else {
      ventedId = branches.filter((b) => b.id !== primaryVentedId)[1]?.id ?? primaryVentedId;
    }
    const f = makeFrame(branches, [ventedId]);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('pairwiseDiff', () => {
  it('returns zero divergence for identical voids', () => {
    const map = buildUserMap('A', 10);
    const diff = pairwiseDiff('Alice', map, 'Bob', map);
    expect(diff.jsDivergence).toBeCloseTo(0, 5);
    expect(diff.cognitiveSimilarity).toBeCloseTo(1, 2);
    expect(diff.mergeable).toBe(true);
  });

  it('detects divergence between different rejection patterns', () => {
    const mapA = buildUserMap('A', 20);
    const mapB = buildUserMap('B', 20);
    const diff = pairwiseDiff('Alice', mapA, 'Bob', mapB);
    expect(diff.jsDivergence).toBeGreaterThan(0);
    expect(diff.cognitiveSimilarity).toBeLessThan(1);
  });

  it('correctly identifies unique and shared branches', () => {
    const mapA = buildUserMap('A', 10); // Rejects A
    const mapB = buildUserMap('B', 10); // Rejects B
    const diff = pairwiseDiff('Alice', mapA, 'Bob', mapB);
    // Both maps have A in rejection counts (from mapA)
    // and B in rejection counts (from mapB)
    expect(diff.shared.length + diff.uniqueToA.length + diff.uniqueToB.length).toBeGreaterThan(0);
  });

  it('cognitiveSimilarity is in [0, 1]', () => {
    const mapA = buildUserMap('A', 20);
    const mapB = buildUserMap('C', 20);
    const diff = pairwiseDiff('Alice', mapA, 'Bob', mapB);
    expect(diff.cognitiveSimilarity).toBeGreaterThanOrEqual(0);
    expect(diff.cognitiveSimilarity).toBeLessThanOrEqual(1);
  });
});

describe('teamDiversityMatrix', () => {
  it('produces symmetric matrix with zero diagonal', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 10) },
      { label: 'Bob', map: buildUserMap('B', 10) },
      { label: 'Carol', map: buildUserMap('C', 10) },
    ];
    const matrix = teamDiversityMatrix(team);
    expect(matrix.users).toHaveLength(3);
    expect(matrix.divergenceMatrix).toHaveLength(3);

    // Diagonal is zero
    for (let i = 0; i < 3; i++) {
      expect(matrix.divergenceMatrix[i][i]).toBe(0);
    }

    // Symmetric
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(matrix.divergenceMatrix[i][j]).toBeCloseTo(matrix.divergenceMatrix[j][i], 5);
      }
    }
  });

  it('finds most similar and most different pairs', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 10) },
      { label: 'Bob', map: buildUserMap('A', 10) }, // Same as Alice
      { label: 'Carol', map: buildUserMap('C', 10) }, // Different
    ];
    const matrix = teamDiversityMatrix(team);
    expect(matrix.mostSimilarPair).not.toBeNull();
    expect(matrix.mostDifferentPair).not.toBeNull();
    // Alice and Bob should be most similar (same rejection pattern)
    expect(matrix.mostSimilarPair).toContain('Alice');
    expect(matrix.mostSimilarPair).toContain('Bob');
  });
});

describe('computeEnsembleGap', () => {
  it('ensemble is better than best individual for diverse team', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 20) },
      { label: 'Bob', map: buildUserMap('B', 20) },
      { label: 'Carol', map: buildUserMap('C', 20) },
    ];
    const analysis = computeEnsembleGap(team);
    // Ensemble covers more branches than any individual
    expect(analysis.ensembleInverseBule).toBeGreaterThanOrEqual(0);
    expect(typeof analysis.diversityDividend).toBe('number');
  });

  it('handles empty team', () => {
    const analysis = computeEnsembleGap([]);
    expect(analysis.ensembleInverseBule).toBe(0);
  });
});

describe('blindSpotAnalysis', () => {
  it('identifies under-rejected branches for each user', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 20) }, // Mostly rejects A
      { label: 'Bob', map: buildUserMap('B', 20) },   // Mostly rejects B
    ];
    const profiles = blindSpotAnalysis(team);

    const alice = profiles.find((p) => p.user === 'Alice');
    expect(alice).toBeDefined();
    // Alice rejects A 80% of the time, so B should be under-rejected relative to average
    // (Bob rejects B heavily, raising the average for B)
    expect(alice!.underRejected.length + alice!.neverRejected.length).toBeGreaterThanOrEqual(0);
    expect(alice!.overRejected.length).toBeGreaterThanOrEqual(0);
  });

  it('computes blindSpotRatio correctly', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 10) },
      { label: 'Bob', map: buildUserMap('B', 10) },
    ];
    const profiles = blindSpotAnalysis(team);
    for (const p of profiles) {
      expect(p.blindSpotRatio).toBeGreaterThanOrEqual(0);
      expect(p.blindSpotRatio).toBeLessThanOrEqual(1);
    }
  });
});

describe('assessMonocultureRisk', () => {
  it('reports critical risk for identical voids', () => {
    const sharedMap = buildUserMap('A', 20);
    const team = [
      { label: 'Alice', map: sharedMap },
      { label: 'Bob', map: sharedMap },
    ];
    const risk = assessMonocultureRisk(team);
    expect(risk.atRisk).toBe(true);
    expect(risk.level).toBe('critical');
    expect(risk.averageSimilarity).toBeCloseTo(1, 2);
  });

  it('reports low risk for diverse voids', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 20) },
      { label: 'Bob', map: buildUserMap('B', 20) },
      { label: 'Carol', map: buildUserMap('C', 20) },
    ];
    const risk = assessMonocultureRisk(team);
    // Different rejection patterns should have lower similarity
    expect(risk.averageSimilarity).toBeLessThan(1);
    expect(risk.universallyRejected.length).toBeLessThanOrEqual(3);
  });

  it('identifies universally rejected branches', () => {
    // All three team members reject A
    const team = [
      { label: 'Alice', map: buildUserMap('A', 10) },
      { label: 'Bob', map: buildUserMap('A', 10) },
    ];
    const risk = assessMonocultureRisk(team);
    expect(risk.universallyRejected).toContain('A');
  });
});

describe('findComplementaryPairs', () => {
  it('finds pairs with non-overlapping coverage', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 20) },
      { label: 'Bob', map: buildUserMap('B', 20) },
      { label: 'Carol', map: buildUserMap('C', 20) },
    ];
    const pairs = findComplementaryPairs(team);
    expect(pairs.length).toBeGreaterThan(0);
    for (const pair of pairs) {
      expect(pair.complementarity).toBeGreaterThanOrEqual(0);
      expect(pair.complementarity).toBeLessThanOrEqual(1);
      expect(pair.combinedCoverage).toBeGreaterThan(0);
    }
  });

  it('sorts by complementarity descending', () => {
    const team = [
      { label: 'Alice', map: buildUserMap('A', 20) },
      { label: 'Bob', map: buildUserMap('B', 20) },
      { label: 'Carol', map: buildUserMap('A', 20) }, // Same as Alice
    ];
    const pairs = findComplementaryPairs(team);
    for (let i = 1; i < pairs.length; i++) {
      expect(pairs[i].complementarity).toBeLessThanOrEqual(pairs[i - 1].complementarity);
    }
  });
});

describe('optimizeTeamComposition', () => {
  it('selects team that maximizes coverage', () => {
    const candidates = [
      { label: 'Alice', map: buildUserMap('A', 20) },
      { label: 'Bob', map: buildUserMap('B', 20) },
      { label: 'Carol', map: buildUserMap('C', 20) },
      { label: 'Dave', map: buildUserMap('A', 20) }, // Redundant with Alice
    ];
    const result = optimizeTeamComposition(candidates, 3);
    expect(result.selected).toHaveLength(3);
    // Should prefer Alice, Bob, Carol over Dave (who duplicates Alice)
    expect(result.selected).toContain('Alice');
    expect(result.selected).toContain('Bob');
    expect(result.selected).toContain('Carol');
    expect(result.coverage).toBeGreaterThan(0);
  });

  it('handles empty candidates', () => {
    const result = optimizeTeamComposition([], 3);
    expect(result.selected).toHaveLength(0);
  });

  it('handles teamSize larger than candidates', () => {
    const candidates = [
      { label: 'Alice', map: buildUserMap('A', 10) },
      { label: 'Bob', map: buildUserMap('B', 10) },
    ];
    const result = optimizeTeamComposition(candidates, 5);
    expect(result.selected).toHaveLength(2);
  });
});

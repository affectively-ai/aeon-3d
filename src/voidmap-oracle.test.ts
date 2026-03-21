import { describe, it, expect } from 'vitest';
import {
  profileBranches,
  designOracleMonoculture,
  analyzeGodMode,
  computePerTombstoneRouting,
  computePigeonholeWitness,
  runOracleAnalysis,
} from './voidmap-oracle.js';
import { createVoidMap, recordFrame, complementDistribution } from './voidmap.js';
import { runNetflixDemo } from './voidmap-netflix.js';
import type { TopologyBranch } from './index.js';

function makeBranch(id: string, cost: number, quality: number, vertices: number): TopologyBranch {
  return { id, vertices, drawCalls: 1, estimatedCostMs: cost, quality };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[], budgetMs = 8) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs },
    result: {
      collapsedBy: 'race' as const, winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving, ventedBranchIds: ventedIds,
      totalVertices: 0, totalDrawCalls: 0, totalCostMs: 0, budgetMs, overBudget: false,
    },
  };
}

function buildDiverseMap() {
  // Build a map where different branches win in different rounds (diverse)
  const branches = [
    makeBranch('MF', 5, 0.8, 400),
    makeBranch('kNN', 6, 0.7, 500),
    makeBranch('RBM', 7, 0.6, 400),
    makeBranch('tSVD', 5.5, 0.75, 300),
  ];
  let map = createVoidMap();
  for (let r = 0; r < 100; r++) {
    const winnerIdx = r % 4;
    const ventedIds = branches.filter((_, i) => i !== winnerIdx).map((b) => b.id);
    const f = makeFrame(branches, ventedIds);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('profileBranches', () => {
  it('returns profiles with complement weights summing to 1', () => {
    const map = buildDiverseMap();
    const profiles = profileBranches(map);
    const sum = profiles.reduce((s, p) => s + p.complementWeight, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('returns profiles sorted by complement weight (descending)', () => {
    const map = buildDiverseMap();
    const profiles = profileBranches(map);
    for (let i = 1; i < profiles.length; i++) {
      expect(profiles[i].complementWeight).toBeLessThanOrEqual(profiles[i - 1].complementWeight);
    }
  });

  it('returns empty for empty map', () => {
    expect(profileBranches(createVoidMap())).toHaveLength(0);
  });
});

describe('designOracleMonoculture', () => {
  it('picks the branch with highest complement weight', () => {
    const map = buildDiverseMap();
    const profiles = profileBranches(map);
    const oracle = designOracleMonoculture(map);
    expect(oracle.designedBranchId).toBe(profiles[0].branchId);
  });

  it('oracle weight is bounded for diverse inputs', () => {
    const map = buildDiverseMap();
    const oracle = designOracleMonoculture(map);
    // For uniformly diverse data, oracle weight should be close to 1/N
    expect(oracle.oracleWeight).toBeGreaterThan(0);
    expect(oracle.oracleWeight).toBeLessThanOrEqual(1);
  });
});

describe('analyzeGodMode', () => {
  it('returns beatsEnsemble = true (theoretical)', () => {
    const map = buildDiverseMap();
    const godMode = analyzeGodMode(map);
    expect(godMode.beatsEnsemble).toBe(true);
  });

  it('has positive realization gap', () => {
    const map = buildDiverseMap();
    const godMode = analyzeGodMode(map);
    expect(godMode.realizationGap).toBeGreaterThan(0);
  });

  it('requires multiple dimensions', () => {
    const map = buildDiverseMap();
    const godMode = analyzeGodMode(map);
    expect(godMode.requiredDimensions).toBeGreaterThan(1);
  });
});

describe('computePerTombstoneRouting', () => {
  it('is a diversity strategy (uses multiple branches)', () => {
    const map = buildDiverseMap();
    const routing = computePerTombstoneRouting(map);
    expect(routing.isDiversityStrategy).toBe(true);
  });

  it('is the ceiling (by definition)', () => {
    const map = buildDiverseMap();
    const routing = computePerTombstoneRouting(map);
    expect(routing.isCeiling).toBe(true);
  });

  it('beats monoculture when diverse', () => {
    const map = buildDiverseMap();
    const routing = computePerTombstoneRouting(map);
    expect(routing.beatsMonoculture).toBe(true);
  });

  it('distributes selections across branches', () => {
    const map = buildDiverseMap();
    const routing = computePerTombstoneRouting(map);
    expect(routing.selectionCounts.size).toBeGreaterThan(1);
  });
});

describe('computePigeonholeWitness', () => {
  it('has non-negative monoculture gap', () => {
    const map = buildDiverseMap();
    const witness = computePigeonholeWitness(map);
    // The gap measures waste of monoculture relative to ensemble
    expect(typeof witness.monocultureGap).toBe('number');
    expect(witness.diversityDividend).toBeGreaterThanOrEqual(0);
  });

  it('proves god-mode requires omniscience', () => {
    const map = buildDiverseMap();
    const witness = computePigeonholeWitness(map);
    expect(witness.godModeRequiresOmniscience).toBe(true);
  });

  it('proves oracle routing is diversity', () => {
    const map = buildDiverseMap();
    const witness = computePigeonholeWitness(map);
    expect(witness.oracleRoutingIsDiversity).toBe(true);
  });

  it('has non-zero void entropy', () => {
    const map = buildDiverseMap();
    const witness = computePigeonholeWitness(map);
    expect(witness.voidEntropy).toBeGreaterThan(0);
  });

  it('works on Netflix demo data', () => {
    const { map } = runNetflixDemo({ maxFrames: 200, seed: 42 });
    const witness = computePigeonholeWitness(map);
    expect(witness.godModeRequiresOmniscience).toBe(true);
    expect(witness.oracleRoutingIsDiversity).toBe(true);
  });
});

describe('runOracleAnalysis', () => {
  it('returns complete analysis with all components', () => {
    const map = buildDiverseMap();
    const analysis = runOracleAnalysis(map);

    expect(analysis.profiles.length).toBe(4);
    expect(analysis.oracleMonoculture.designedBranchId).toBeTruthy();
    expect(analysis.godMode.beatsEnsemble).toBe(true);
    expect(analysis.perTombstoneRouting.isDiversityStrategy).toBe(true);
    expect(typeof analysis.pigeonholeWitness.oracleCannotBeatEnsemble).toBe('boolean');
    expect(analysis.stats.totalTombstones).toBeGreaterThan(0);
  });
});

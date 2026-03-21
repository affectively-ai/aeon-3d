import { describe, it, expect } from 'vitest';
import {
  createVoidMap,
  recordFrame,
  branchToPosition,
  complementDistribution,
  voidEntropy,
  computeVoidBoundary,
} from './voidmap.js';
import type { TopologyFrameInput, TopologyFrameResult, TopologyBranch } from './index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeBranch(id: string, cost: number, quality: number, vertices: number, vent = false): TopologyBranch {
  return { id, vertices, drawCalls: 1, estimatedCostMs: cost, quality, vent };
}

function makeFrame(
  branches: TopologyBranch[],
  ventedIds: string[],
  budgetMs = 8
): { input: TopologyFrameInput; result: TopologyFrameResult } {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race', budgetMs },
    result: {
      collapsedBy: 'race',
      winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving,
      ventedBranchIds: ventedIds,
      totalVertices: surviving.reduce((s, b) => s + b.vertices, 0),
      totalDrawCalls: surviving.length,
      totalCostMs: surviving.reduce((s, b) => s + b.estimatedCostMs, 0),
      budgetMs,
      overBudget: false,
    },
  };
}

function buildTestMap(rounds: number): ReturnType<typeof createVoidMap> {
  const branches = [
    makeBranch('MF', 5, 0.8, 400),
    makeBranch('kNN', 6, 0.7, 500),
    makeBranch('RBM', 7, 0.6, 400),
  ];
  let map = createVoidMap();
  for (let r = 0; r < rounds; r++) {
    // Rotate which branch wins each round
    const winnerIdx = r % 3;
    const ventedIds = branches.filter((_, i) => i !== winnerIdx).map((b) => b.id);
    const frame = makeFrame(branches, ventedIds);
    map = recordFrame(map, frame.input, frame.result);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('createVoidMap', () => {
  it('returns empty map', () => {
    const map = createVoidMap();
    expect(map.tombstones).toHaveLength(0);
    expect(map.round).toBe(0);
    expect(map.rejectionCounts.size).toBe(0);
  });
});

describe('branchToPosition', () => {
  it('maps cost/budget to X, quality to Y, log2(vertices)/20 to Z', () => {
    const branch = makeBranch('test', 4, 0.75, 1024);
    const pos = branchToPosition(branch, 1, 8);
    expect(pos.x).toBeCloseTo(0.5, 5);
    expect(pos.y).toBeCloseTo(0.75, 5);
    expect(pos.z).toBeCloseTo(Math.log2(1024) / 20, 5);
  });

  it('handles zero budget', () => {
    const branch = makeBranch('test', 4, 0.5, 100);
    const pos = branchToPosition(branch, 1, 0);
    expect(pos.x).toBe(0);
  });
});

describe('recordFrame', () => {
  it('appends tombstones for vented branches', () => {
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 10, 0.3, 200)];
    const frame = makeFrame(branches, ['B']);
    let map = createVoidMap();
    map = recordFrame(map, frame.input, frame.result);

    expect(map.tombstones).toHaveLength(1);
    expect(map.tombstones[0].branchId).toBe('B');
    expect(map.round).toBe(1);
    expect(map.rejectionCounts.get('B')).toBe(1);
  });

  it('classifies reason as explicit-vent for vent=true branches', () => {
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 10, 0.3, 200, true)];
    const frame = makeFrame(branches, ['B']);
    let map = createVoidMap();
    map = recordFrame(map, frame.input, frame.result);
    expect(map.tombstones[0].reason).toBe('explicit-vent');
  });

  it('classifies reason as over-budget for high-cost branches', () => {
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 20, 0.3, 200)];
    const frame = makeFrame(branches, ['B'], 8);
    let map = createVoidMap();
    map = recordFrame(map, frame.input, frame.result);
    expect(map.tombstones[0].reason).toBe('over-budget');
  });

  it('respects maxTombstones cap', () => {
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200)];
    let map = createVoidMap();
    for (let i = 0; i < 20; i++) {
      const frame = makeFrame(branches, ['B']);
      map = recordFrame(map, frame.input, frame.result, { maxTombstones: 10 });
    }
    expect(map.tombstones.length).toBeLessThanOrEqual(10);
    expect(map.round).toBe(20);
  });

  it('increments rejection counts across rounds', () => {
    const map = buildTestMap(9);
    // Each branch loses 6 out of 9 rounds (wins 3)
    expect(map.rejectionCounts.get('MF')).toBe(6);
    expect(map.rejectionCounts.get('kNN')).toBe(6);
    expect(map.rejectionCounts.get('RBM')).toBe(6);
  });
});

describe('complementDistribution', () => {
  it('returns empty array for empty map', () => {
    const map = createVoidMap();
    expect(complementDistribution(map)).toEqual([]);
  });

  it('sums to 1.0', () => {
    const map = buildTestMap(30);
    const dist = complementDistribution(map);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 10);
  });

  it('matches ch17 softmax(-eta * v) formula', () => {
    // Build a map with known rejection counts
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200)];
    let map = createVoidMap();
    // A loses 3 times, B loses 7 times
    for (let i = 0; i < 10; i++) {
      const ventedIds = i < 3 ? ['A'] : ['B'];
      const frame = makeFrame(branches, ventedIds);
      map = recordFrame(map, frame.input, frame.result);
    }

    const eta = 3.0;
    const dist = complementDistribution(map, eta);
    const countsA = map.rejectionCounts.get('A')!;
    const countsB = map.rejectionCounts.get('B')!;

    // Manual computation: softmax(-eta * [countsA, countsB])
    const logitA = -eta * countsA;
    const logitB = -eta * countsB;
    const maxLogit = Math.max(logitA, logitB);
    const expA = Math.exp(logitA - maxLogit);
    const expB = Math.exp(logitB - maxLogit);
    const sumExp = expA + expB;

    expect(dist[0]).toBeCloseTo(expA / sumExp, 10);
    expect(dist[1]).toBeCloseTo(expB / sumExp, 10);
  });

  it('gives higher weight to less-rejected branches', () => {
    // Both branches must be rejected at least once to appear in rejectionCounts
    const branches = [makeBranch('less-rejected', 2, 0.9, 100), makeBranch('more-rejected', 10, 0.2, 200)];
    let map = createVoidMap();
    // less-rejected loses 2 times, more-rejected loses 18 times
    for (let i = 0; i < 20; i++) {
      const ventedId = i < 2 ? 'less-rejected' : 'more-rejected';
      const frame = makeFrame(branches, [ventedId]);
      map = recordFrame(map, frame.input, frame.result);
    }
    const dist = complementDistribution(map);
    expect(dist).toHaveLength(2);
    // Less-rejected (2) should have higher weight than more-rejected (18)
    expect(dist[0]).toBeGreaterThan(dist[1]);
  });
});

describe('voidEntropy', () => {
  it('returns 0 for empty map', () => {
    expect(voidEntropy(createVoidMap())).toBe(0);
  });

  it('returns maximum entropy for uniform rejection counts', () => {
    const map = buildTestMap(9); // All three branches rejected equally (6 each)
    const entropy = voidEntropy(map);
    const maxEntropy = Math.log2(3);
    expect(entropy).toBeCloseTo(maxEntropy, 5);
  });

  it('returns lower entropy for peaked distribution', () => {
    const branches = [makeBranch('A', 2, 0.9, 100), makeBranch('B', 10, 0.2, 200)];
    let map = createVoidMap();
    for (let i = 0; i < 50; i++) {
      const frame = makeFrame(branches, ['B']);
      map = recordFrame(map, frame.input, frame.result);
    }
    const entropy = voidEntropy(map);
    const maxEntropy = Math.log2(2);
    expect(entropy).toBeLessThan(maxEntropy);
  });
});

describe('computeVoidBoundary', () => {
  it('returns zeros for empty map', () => {
    const stats = computeVoidBoundary(createVoidMap());
    expect(stats.totalTombstones).toBe(0);
    expect(stats.entropy).toBe(0);
    expect(stats.inverseBule).toBe(0);
    expect(stats.centroid).toEqual({ x: 0, y: 0, z: 0 });
  });

  it('computes correct centroid', () => {
    const branches = [makeBranch('A', 4, 1.0, 1)];
    const frame = makeFrame(branches, ['A'], 8);
    let map = createVoidMap();
    map = recordFrame(map, frame.input, frame.result);

    const stats = computeVoidBoundary(map);
    const expectedPos = branchToPosition(branches[0], 1, 8);
    expect(stats.centroid.x).toBeCloseTo(expectedPos.x, 5);
    expect(stats.centroid.y).toBeCloseTo(expectedPos.y, 5);
    expect(stats.centroid.z).toBeCloseTo(expectedPos.z, 5);
  });

  it('computes Inverse Bule correctly', () => {
    // Uniform distribution: inverseBule should be near 0
    const uniformMap = buildTestMap(9);
    const uniformStats = computeVoidBoundary(uniformMap);
    expect(uniformStats.inverseBule).toBeCloseTo(0, 1);

    // Peaked distribution: inverseBule should be positive
    const branches = [makeBranch('A', 2, 0.9, 100), makeBranch('B', 10, 0.2, 200)];
    let peakedMap = createVoidMap();
    for (let i = 0; i < 50; i++) {
      const frame = makeFrame(branches, ['B']);
      peakedMap = recordFrame(peakedMap, frame.input, frame.result);
    }
    const peakedStats = computeVoidBoundary(peakedMap);
    expect(peakedStats.inverseBule).toBeGreaterThan(0.1);
  });

  it('counts distinct branches for homologyRank', () => {
    const map = buildTestMap(10);
    const stats = computeVoidBoundary(map);
    expect(stats.homologyRank).toBe(3); // MF, kNN, RBM
  });
});

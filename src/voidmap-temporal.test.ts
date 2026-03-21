import { describe, it, expect } from 'vitest';
import {
  sliceAtRound,
  sliceRange,
  diffVoidMaps,
  createTimeline,
  forkAt,
  updateTrunk,
  updateFork,
  mergeToTrunk,
  tombstoneOpacity,
  computeOpacities,
  extractTrajectories,
  snapshotVoidMap,
  restoreFromSnapshot,
} from './voidmap-temporal.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import type { TopologyBranch, TopologyFrameInput, TopologyFrameResult } from './index.js';

function makeBranch(id: string, cost: number, quality: number, vertices: number): TopologyBranch {
  return { id, vertices, drawCalls: 1, estimatedCostMs: cost, quality };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[], budgetMs = 8) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs },
    result: {
      collapsedBy: 'race' as const,
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

function buildMap(rounds: number) {
  const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200), makeBranch('C', 7, 0.6, 300)];
  let map = createVoidMap();
  for (let r = 0; r < rounds; r++) {
    const winnerIdx = r % 3;
    const ventedIds = branches.filter((_, i) => i !== winnerIdx).map((b) => b.id);
    map = recordFrame(map, makeFrame(branches, ventedIds).input, makeFrame(branches, ventedIds).result);
  }
  return map;
}

describe('sliceAtRound', () => {
  it('returns full map when round >= map.round', () => {
    const map = buildMap(10);
    const sliced = sliceAtRound(map, 100);
    expect(sliced).toBe(map);
  });

  it('returns empty map for round <= 0', () => {
    const map = buildMap(10);
    const sliced = sliceAtRound(map, 0);
    expect(sliced.tombstones).toHaveLength(0);
    expect(sliced.round).toBe(0);
  });

  it('filters tombstones by round and recomputes counts', () => {
    const map = buildMap(10);
    const sliced = sliceAtRound(map, 5);
    expect(sliced.round).toBe(5);
    for (const t of sliced.tombstones) {
      expect(t.round).toBeLessThanOrEqual(5);
    }
    // Verify rejection counts are recomputed
    let totalFromCounts = 0;
    for (const count of sliced.rejectionCounts.values()) {
      totalFromCounts += count;
    }
    expect(totalFromCounts).toBe(sliced.tombstones.length);
  });
});

describe('sliceRange', () => {
  it('filters to [fromRound, toRound]', () => {
    const map = buildMap(20);
    const sliced = sliceRange(map, 5, 10);
    for (const t of sliced.tombstones) {
      expect(t.round).toBeGreaterThanOrEqual(5);
      expect(t.round).toBeLessThanOrEqual(10);
    }
  });
});

describe('diffVoidMaps', () => {
  it('returns zero divergence for identical maps', () => {
    const map = buildMap(10);
    const diff = diffVoidMaps(map, map);
    expect(diff.jsDivergence).toBeCloseTo(0, 5);
    expect(diff.onlyInA).toHaveLength(0);
    expect(diff.onlyInB).toHaveLength(0);
    expect(diff.mergeable).toBe(true);
  });

  it('detects tombstones unique to each side', () => {
    const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200)];
    let mapA = createVoidMap();
    let mapB = createVoidMap();

    // A has tombstones for B, B has tombstones for A
    mapA = recordFrame(mapA, makeFrame(branches, ['B']).input, makeFrame(branches, ['B']).result);
    mapB = recordFrame(mapB, makeFrame(branches, ['A']).input, makeFrame(branches, ['A']).result);

    const diff = diffVoidMaps(mapA, mapB);
    expect(diff.onlyInA.length).toBeGreaterThan(0);
    expect(diff.onlyInB.length).toBeGreaterThan(0);
  });

  it('computes rejection delta', () => {
    const branches = [makeBranch('X', 5, 0.8, 100)];
    let mapA = createVoidMap();
    let mapB = createVoidMap();

    for (let i = 0; i < 5; i++) {
      mapA = recordFrame(mapA, makeFrame(branches, ['X']).input, makeFrame(branches, ['X']).result);
    }
    for (let i = 0; i < 3; i++) {
      mapB = recordFrame(mapB, makeFrame(branches, ['X']).input, makeFrame(branches, ['X']).result);
    }

    const diff = diffVoidMaps(mapA, mapB);
    expect(diff.rejectionDelta.get('X')).toBe(2); // A has 5, B has 3, delta = 2
  });
});

describe('fork/merge', () => {
  it('forkAt creates a fork with sliced trunk', () => {
    const map = buildMap(10);
    let timeline = createTimeline(map);
    timeline = forkAt(timeline, 'fork-1', 'Experiment', 5);

    expect(timeline.forks).toHaveLength(1);
    expect(timeline.forks[0].id).toBe('fork-1');
    expect(timeline.forks[0].forkRound).toBe(5);
    expect(timeline.forks[0].map.round).toBe(5);
  });

  it('mergeToTrunk succeeds when voids are similar', () => {
    const map = buildMap(10);
    let timeline = createTimeline(map);
    timeline = forkAt(timeline, 'fork-1', 'Test', 10);

    // Fork has same data as trunk (no divergence)
    const result = mergeToTrunk(timeline, 'fork-1');
    expect(result.merged).toBe(true);
    expect(result.diff.mergeable).toBe(true);
    expect(result.timeline.forks).toHaveLength(0);
  });

  it('mergeToTrunk fails when divergence exceeds threshold', () => {
    // Both maps must share the same branch IDs for JS divergence to be meaningful.
    // Trunk: A rejected 50 times, B rejected 0 times
    // Fork: A rejected 0 times, B rejected 50 times
    const allBranches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200)];

    let trunk = createVoidMap();
    for (let i = 0; i < 50; i++) {
      trunk = recordFrame(trunk, makeFrame(allBranches, ['A']).input, makeFrame(allBranches, ['A']).result);
    }

    let timeline = createTimeline(trunk);
    timeline = forkAt(timeline, 'divergent', 'Divergent', 0);

    // Fork rejects B instead of A -- opposite rejection pattern
    let forkMap = createVoidMap();
    for (let i = 0; i < 50; i++) {
      forkMap = recordFrame(forkMap, makeFrame(allBranches, ['B']).input, makeFrame(allBranches, ['B']).result);
    }
    timeline = updateFork(timeline, 'divergent', forkMap);

    const result = mergeToTrunk(timeline, 'divergent', { mergeThreshold: 0.01 });
    expect(result.merged).toBe(false);
    expect(result.diff.mergeable).toBe(false);
  });
});

describe('tombstoneOpacity', () => {
  it('returns 1.0 for current-round tombstone', () => {
    const t = { branchId: 'A', round: 10, cost: 5, quality: 0.8, vertices: 100, drawCalls: 1, reason: 'race-loser' as const, position: { x: 0, y: 0, z: 0 }, timestamp: 0 };
    expect(tombstoneOpacity(t, 10)).toBeCloseTo(1.0, 5);
  });

  it('decays toward minOpacity for old tombstones', () => {
    const t = { branchId: 'A', round: 1, cost: 5, quality: 0.8, vertices: 100, drawCalls: 1, reason: 'race-loser' as const, position: { x: 0, y: 0, z: 0 }, timestamp: 0 };
    const opacity = tombstoneOpacity(t, 100, { decay: 0.05, minOpacity: 0.1 });
    expect(opacity).toBeGreaterThan(0.1);
    expect(opacity).toBeLessThan(0.5);
  });
});

describe('extractTrajectories', () => {
  it('returns trajectories for branches with 2+ deaths', () => {
    const map = buildMap(10);
    const trajectories = extractTrajectories(map);
    expect(trajectories.length).toBeGreaterThan(0);
    for (const traj of trajectories) {
      expect(traj.rejectionCount).toBeGreaterThanOrEqual(2);
      expect(traj.positions.length).toBe(traj.rejectionCount);
    }
  });

  it('returns empty for single-tombstone branches', () => {
    const branches = [makeBranch('A', 5, 0.8, 100)];
    let map = createVoidMap();
    map = recordFrame(map, makeFrame(branches, ['A']).input, makeFrame(branches, ['A']).result);
    const trajectories = extractTrajectories(map);
    expect(trajectories).toHaveLength(0);
  });
});

describe('snapshotVoidMap / restoreFromSnapshot', () => {
  it('round-trips correctly', async () => {
    const map = buildMap(10);
    const snapshot = await snapshotVoidMap(map);

    expect(snapshot.round).toBe(map.round);
    expect(snapshot.tombstoneCount).toBe(map.tombstones.length);
    expect(snapshot.cid).toBeTruthy();

    const restored = restoreFromSnapshot(snapshot);
    expect(restored.round).toBe(map.round);
    expect(restored.tombstones.length).toBe(map.tombstones.length);
    expect(restored.rejectionCounts.size).toBe(map.rejectionCounts.size);

    for (const [key, value] of map.rejectionCounts) {
      expect(restored.rejectionCounts.get(key)).toBe(value);
    }
  });
});

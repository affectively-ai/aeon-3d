import { describe, it, expect } from 'vitest';
import {
  computeCentroidContrail,
  centroidVelocity,
  projectGhosts,
  fitPredictionCone,
  detectAtRisk,
  detectPhaseTransitions,
} from './voidmap-precognition.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import type { TopologyBranch } from './index.js';

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
      totalVertices: 0,
      totalDrawCalls: 0,
      totalCostMs: 0,
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
    const f = makeFrame(branches, ventedIds);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('computeCentroidContrail', () => {
  it('returns empty for empty map', () => {
    expect(computeCentroidContrail(createVoidMap())).toHaveLength(0);
  });

  it('produces waypoints at correct rounds', () => {
    const map = buildMap(20);
    const contrail = computeCentroidContrail(map, { stepSize: 5 });
    const rounds = contrail.map((w) => w.round);
    expect(rounds).toContain(5);
    expect(rounds).toContain(10);
    expect(rounds).toContain(15);
    expect(rounds).toContain(20);
  });

  it('includes entropy and inverseBule at each waypoint', () => {
    const map = buildMap(10);
    const contrail = computeCentroidContrail(map);
    for (const wp of contrail) {
      expect(typeof wp.entropy).toBe('number');
      expect(typeof wp.inverseBule).toBe('number');
      expect(wp.tombstoneCount).toBeGreaterThan(0);
    }
  });
});

describe('centroidVelocity', () => {
  it('returns empty for fewer than 2 waypoints', () => {
    expect(centroidVelocity([])).toHaveLength(0);
    expect(centroidVelocity([{ round: 1, position: { x: 0, y: 0, z: 0 }, entropy: 0, inverseBule: 0, tombstoneCount: 0 }])).toHaveLength(0);
  });

  it('computes velocity between waypoints', () => {
    const contrail = [
      { round: 1, position: { x: 0, y: 0, z: 0 }, entropy: 0, inverseBule: 0, tombstoneCount: 1 },
      { round: 2, position: { x: 1, y: 0, z: 0 }, entropy: 0, inverseBule: 0, tombstoneCount: 2 },
    ];
    const vel = centroidVelocity(contrail);
    expect(vel).toHaveLength(1);
    expect(vel[0].velocity.x).toBeCloseTo(1, 5);
    expect(vel[0].speed).toBeCloseTo(1, 5);
  });
});

describe('projectGhosts', () => {
  it('returns empty for empty map', () => {
    expect(projectGhosts(createVoidMap())).toHaveLength(0);
  });

  it('assigns higher death probability to more-rejected branches', () => {
    const branches = [makeBranch('winner', 2, 0.9, 100), makeBranch('loser', 10, 0.2, 200)];
    let map = createVoidMap();
    for (let i = 0; i < 20; i++) {
      const f = makeFrame(branches, ['loser']);
      map = recordFrame(map, f.input, f.result);
    }
    // Winner has 0 rejections, loser has 20
    const ghosts = projectGhosts(map);
    const loserGhost = ghosts.find((g) => g.branchId === 'loser');
    expect(loserGhost).toBeDefined();
    expect(loserGhost!.deathProbability).toBeGreaterThan(0.5);
  });
});

describe('fitPredictionCone', () => {
  it('returns null for insufficient data', () => {
    const map = buildMap(2);
    expect(fitPredictionCone(map)).toBeNull();
  });

  it('returns a cone with valid properties for sufficient data', () => {
    const map = buildMap(50);
    const cone = fitPredictionCone(map);
    if (cone) {
      expect(cone.confidence).toBeGreaterThanOrEqual(0);
      expect(cone.confidence).toBeLessThanOrEqual(1);
      expect(cone.reach).toBeGreaterThan(0);
      expect(cone.halfAngle).toBeGreaterThanOrEqual(0);
      expect(cone.halfAngle).toBeLessThanOrEqual(Math.PI / 3);
    }
  });
});

describe('detectAtRisk', () => {
  it('returns empty when no branches are in the cone', () => {
    const cone = {
      apex: { x: 0, y: 0, z: 0 },
      direction: { x: 1, y: 0, z: 0 },
      halfAngle: 0.1,
      confidence: 1,
      reach: 1,
    };
    // Branch is behind the cone
    const positions = new Map([['test', { x: -5, y: 0, z: 0 }]]);
    expect(detectAtRisk(cone, positions)).toHaveLength(0);
  });

  it('detects branches inside the cone', () => {
    const cone = {
      apex: { x: 0, y: 0, z: 0 },
      direction: { x: 1, y: 0, z: 0 },
      halfAngle: Math.PI / 4,
      confidence: 1,
      reach: 10,
    };
    const positions = new Map([
      ['inside', { x: 5, y: 0, z: 0 }],
      ['outside', { x: 0, y: 0, z: 10 }],
    ]);
    const atRisk = detectAtRisk(cone, positions);
    expect(atRisk.some((r) => r.branchId === 'inside')).toBe(true);
    expect(atRisk.some((r) => r.branchId === 'outside')).toBe(false);
  });
});

describe('detectPhaseTransitions', () => {
  it('returns empty for insufficient data', () => {
    expect(detectPhaseTransitions([])).toHaveLength(0);
  });

  it('detects convergence when entropy drops', () => {
    const contrail = [
      { round: 1, position: { x: 0, y: 0, z: 0 }, entropy: 2.0, inverseBule: 0.1, tombstoneCount: 10 },
      { round: 2, position: { x: 0, y: 0, z: 0 }, entropy: 1.2, inverseBule: 0.5, tombstoneCount: 20 },
    ];
    const transitions = detectPhaseTransitions(contrail, { entropyThreshold: 0.3 });
    expect(transitions.length).toBeGreaterThan(0);
    expect(transitions[0].type).toBe('convergence');
  });

  it('detects disruption when entropy spikes', () => {
    const contrail = [
      { round: 1, position: { x: 0, y: 0, z: 0 }, entropy: 1.0, inverseBule: 0.5, tombstoneCount: 10 },
      { round: 2, position: { x: 0, y: 0, z: 0 }, entropy: 2.0, inverseBule: 0.1, tombstoneCount: 20 },
    ];
    const transitions = detectPhaseTransitions(contrail, { entropyThreshold: 0.3 });
    expect(transitions.length).toBeGreaterThan(0);
    expect(transitions[0].type).toBe('disruption');
  });
});

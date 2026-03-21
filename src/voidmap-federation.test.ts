import { describe, it, expect } from 'vitest';
import {
  parseVoidAddress,
  createVoidAddress,
  createFederatedNode,
  updateLocalMap,
  createGossipMessage,
  receiveGossip,
  gossipRound,
  aggregateVoidBoundaries,
  computeNodeAggregate,
  crossRegionDiff,
  computeGlobalMetrics,
  collectiveDeathTrajectories,
  globalCentroidVelocity,
  civilizationalPredictionCone,
} from './voidmap-federation.js';
import { createVoidMap, recordFrame } from './voidmap.js';
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

function buildMap(branches: TopologyBranch[], ventedId: string, rounds: number) {
  let map = createVoidMap();
  for (let i = 0; i < rounds; i++) {
    const f = makeFrame(branches, [ventedId]);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('Void addressing', () => {
  it('creates and parses addresses', () => {
    const addr = createVoidAddress('earth', 'node-42', 'abc123def');
    expect(addr.uri).toBe('aeon://rhizome/earth/node-42/void/abc123def');

    const parsed = parseVoidAddress(addr.uri);
    expect(parsed).not.toBeNull();
    expect(parsed!.world).toBe('earth');
    expect(parsed!.nodeId).toBe('node-42');
    expect(parsed!.voidCid).toBe('abc123def');
  });

  it('returns null for invalid URIs', () => {
    expect(parseVoidAddress('https://example.com')).toBeNull();
    expect(parseVoidAddress('aeon://rhizome/world')).toBeNull();
  });
});

describe('Federated nodes', () => {
  it('creates a node with empty local map', () => {
    const node = createFederatedNode('node-1', 'earth');
    expect(node.nodeId).toBe('node-1');
    expect(node.localMap.tombstones).toHaveLength(0);
    expect(node.neighborSnapshots.size).toBe(0);
  });

  it('updates local map', () => {
    const branches = [makeBranch('A', 5)];
    const map = buildMap(branches, 'A', 5);
    const node = updateLocalMap(createFederatedNode('node-1', 'earth'), map);
    expect(node.localMap.round).toBe(5);
    expect(node.aggregate).toBeNull(); // Invalidated
  });
});

describe('Gossip protocol', () => {
  it('creates gossip messages', async () => {
    const branches = [makeBranch('A', 5)];
    const map = buildMap(branches, 'A', 3);
    const node = updateLocalMap(createFederatedNode('n1', 'earth'), map);
    const msg = await createGossipMessage(node);
    expect(msg.fromNodeId).toBe('n1');
    expect(msg.snapshot.round).toBe(3);
  });

  it('receives gossip and updates neighbor snapshots', async () => {
    const node1 = updateLocalMap(
      createFederatedNode('n1', 'earth'),
      buildMap([makeBranch('A', 5)], 'A', 5)
    );
    const msg = await createGossipMessage(node1);

    let node2 = createFederatedNode('n2', 'earth');
    node2 = receiveGossip(node2, msg);
    expect(node2.neighborSnapshots.size).toBe(1);
    expect(node2.neighborSnapshots.has('n1')).toBe(true);
  });

  it('ignores messages from self', async () => {
    const node = updateLocalMap(
      createFederatedNode('n1', 'earth'),
      buildMap([makeBranch('A', 5)], 'A', 5)
    );
    const msg = await createGossipMessage(node);
    const updated = receiveGossip(node, msg);
    expect(updated.neighborSnapshots.size).toBe(0);
  });

  it('performs a full gossip round', async () => {
    const node1 = updateLocalMap(
      createFederatedNode('n1', 'earth'),
      buildMap([makeBranch('A', 5)], 'A', 5)
    );
    const node2 = updateLocalMap(
      createFederatedNode('n2', 'earth'),
      buildMap([makeBranch('B', 6)], 'B', 5)
    );

    const msg1 = await createGossipMessage(node1);
    const msg2 = await createGossipMessage(node2);

    const { updatedNode: n1Updated } = await gossipRound(node1, [msg2]);
    const { updatedNode: n2Updated } = await gossipRound(node2, [msg1]);

    expect(n1Updated.neighborSnapshots.has('n2')).toBe(true);
    expect(n2Updated.neighborSnapshots.has('n1')).toBe(true);
  });
});

describe('Aggregation', () => {
  it('unions tombstones across maps', () => {
    const mapA = buildMap([makeBranch('A', 5)], 'A', 5);
    const mapB = buildMap([makeBranch('B', 6)], 'B', 5);
    const aggregate = aggregateVoidBoundaries([mapA, mapB]);

    expect(aggregate.rejectionCounts.has('A')).toBe(true);
    expect(aggregate.rejectionCounts.has('B')).toBe(true);
    expect(aggregate.tombstones.length).toBe(10); // 5 + 5
  });

  it('deduplicates tombstones with same key', () => {
    const map = buildMap([makeBranch('A', 5)], 'A', 5);
    const aggregate = aggregateVoidBoundaries([map, map]);
    // Same tombstones, should deduplicate
    expect(aggregate.tombstones.length).toBe(5);
  });

  it('respects maxTombstones', () => {
    const map = buildMap([makeBranch('A', 5)], 'A', 20);
    const aggregate = aggregateVoidBoundaries([map], { maxTombstones: 10 });
    expect(aggregate.tombstones.length).toBeLessThanOrEqual(10);
  });
});

describe('Cross-region diff', () => {
  it('returns zero divergence for identical regions', () => {
    const map = buildMap([makeBranch('A', 5)], 'A', 10);
    const diff = crossRegionDiff('US', map, 'EU', map);
    expect(diff.jsDivergence).toBeCloseTo(0, 5);
    expect(diff.mergeable).toBe(true);
  });

  it('detects divergence between different regions', () => {
    // Both regions must have both branches in rejectionCounts for meaningful divergence
    const branches = [makeBranch('A', 5), makeBranch('B', 6)];
    let mapA = createVoidMap();
    let mapB = createVoidMap();
    // Region A: mostly rejects A (16 times), occasionally B (4 times)
    for (let i = 0; i < 20; i++) {
      const ventedId = i < 16 ? 'A' : 'B';
      const f = makeFrame(branches, [ventedId]);
      mapA = recordFrame(mapA, f.input, f.result);
    }
    // Region B: mostly rejects B (16 times), occasionally A (4 times)
    for (let i = 0; i < 20; i++) {
      const ventedId = i < 16 ? 'B' : 'A';
      const f = makeFrame(branches, [ventedId]);
      mapB = recordFrame(mapB, f.input, f.result);
    }
    const diff = crossRegionDiff('US', mapA, 'EU', mapB);
    expect(diff.jsDivergence).toBeGreaterThan(0);
    expect(diff.cognitiveDistance).toBeGreaterThan(0);
  });
});

describe('Global metrics', () => {
  it('computes metrics from multiple nodes', () => {
    const nodes = [
      updateLocalMap(createFederatedNode('n1', 'earth'), buildMap([makeBranch('A', 5)], 'A', 10)),
      updateLocalMap(createFederatedNode('n2', 'earth'), buildMap([makeBranch('B', 6)], 'B', 10)),
    ];
    const metrics = computeGlobalMetrics(nodes);
    expect(metrics.nodeCount).toBe(2);
    expect(metrics.totalTombstones).toBeGreaterThan(0);
    expect(metrics.totalBranches).toBe(2);
  });
});

describe('Collective death trajectories', () => {
  it('finds branches rejected across multiple nodes', () => {
    const branches = [makeBranch('shared', 5), makeBranch('B', 6)];
    const nodes = [
      updateLocalMap(createFederatedNode('n1', 'earth'), buildMap(branches, 'shared', 10)),
      updateLocalMap(createFederatedNode('n2', 'earth'), buildMap(branches, 'shared', 10)),
      updateLocalMap(createFederatedNode('n3', 'earth'), buildMap([makeBranch('C', 7)], 'C', 10)),
    ];
    const collective = collectiveDeathTrajectories(nodes, 2);
    const sharedTraj = collective.find((t) => t.branchId === 'shared');
    expect(sharedTraj).toBeDefined();
    expect(sharedTraj!.nodeCount).toBe(2);
  });
});

describe('Global centroid velocity', () => {
  it('computes velocity from snapshots', () => {
    const snapshots = [
      { round: 1, centroid: { x: 0, y: 0, z: 0 } },
      { round: 2, centroid: { x: 1, y: 0, z: 0 } },
      { round: 3, centroid: { x: 2, y: 0, z: 0 } },
    ];
    const vel = globalCentroidVelocity(snapshots);
    expect(vel).toHaveLength(2);
    expect(vel[0].speed).toBeCloseTo(1, 5);
  });
});

describe('Civilizational prediction cone', () => {
  it('returns null for insufficient data', () => {
    expect(civilizationalPredictionCone([], { x: 0, y: 0, z: 0 })).toBeNull();
  });

  it('returns a cone for consistent velocity', () => {
    const velocities = Array.from({ length: 10 }, (_, i) => ({
      round: i + 1,
      velocity: { x: 1, y: 0, z: 0 },
      speed: 1,
    }));
    const cone = civilizationalPredictionCone(velocities, { x: 0, y: 0, z: 0 });
    expect(cone).not.toBeNull();
    expect(cone!.confidence).toBeGreaterThan(0.5);
    expect(cone!.direction.x).toBeCloseTo(1, 2);
  });
});

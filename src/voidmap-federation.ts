/**
 * voidmap-federation.ts
 *
 * Decentralized void aggregation across the rhizome network.
 * Every node maintains its local void boundary. Aggregation happens
 * through gossip -- nodes share snapshots with neighbors, who compute
 * local aggregates, who share with their neighbors. The global void
 * emerges from local rejection without any node having a complete picture.
 */

import type { Vec3 } from './index.js';
import { addVec3, scaleVec3, subtractVec3, lengthVec3 } from './index.js';
import type { VoidMap, VoidBoundaryStats, Tombstone } from './voidmap.js';
import {
  createVoidMap,
  complementDistribution,
  voidEntropy,
  computeVoidBoundary,
} from './voidmap.js';
import type { VoidMapSnapshot } from './voidmap-temporal.js';
import {
  snapshotVoidMap,
  restoreFromSnapshot,
  extractTrajectories,
} from './voidmap-temporal.js';

// ---------------------------------------------------------------------------
// Addressing
// ---------------------------------------------------------------------------

export interface VoidAddress {
  /** Full address: aeon://rhizome/<world>/<node>/void/<cid> */
  uri: string;
  world: string;
  nodeId: string;
  voidCid: string;
}

/**
 * Parse a void address URI.
 */
export function parseVoidAddress(uri: string): VoidAddress | null {
  const match = uri.match(
    /^aeon:\/\/rhizome\/([^/]+)\/([^/]+)\/void\/([a-f0-9]+(?:-[a-z0-9-]+)?)$/
  );
  if (!match) return null;
  return {
    uri,
    world: match[1],
    nodeId: match[2],
    voidCid: match[3],
  };
}

/**
 * Construct a void address URI.
 */
export function createVoidAddress(
  world: string,
  nodeId: string,
  voidCid: string
): VoidAddress {
  return {
    uri: `aeon://rhizome/${world}/${nodeId}/void/${voidCid}`,
    world,
    nodeId,
    voidCid,
  };
}

// ---------------------------------------------------------------------------
// Federated node
// ---------------------------------------------------------------------------

export interface FederatedVoidNode {
  /** This node's ID */
  nodeId: string;
  /** This node's world */
  world: string;
  /** This node's local void boundary */
  localMap: VoidMap;
  /** Neighbors: nodeId -> most recent snapshot received */
  neighborSnapshots: ReadonlyMap<string, VoidMapSnapshot>;
  /** Aggregate computed from local + neighbor data */
  aggregate: VoidMap | null;
  /** Last gossip round */
  lastGossipRound: number;
  /** Gossip history: rounds at which gossip occurred */
  gossipHistory: readonly number[];
}

/**
 * Create a federated void node.
 */
export function createFederatedNode(
  nodeId: string,
  world: string,
  localMap?: VoidMap
): FederatedVoidNode {
  return {
    nodeId,
    world,
    localMap: localMap ?? createVoidMap(),
    neighborSnapshots: new Map(),
    aggregate: null,
    lastGossipRound: 0,
    gossipHistory: [],
  };
}

/**
 * Update the node's local VoidMap.
 */
export function updateLocalMap(
  node: FederatedVoidNode,
  localMap: VoidMap
): FederatedVoidNode {
  return { ...node, localMap, aggregate: null };
}

// ---------------------------------------------------------------------------
// Gossip protocol
// ---------------------------------------------------------------------------

export interface GossipMessage {
  fromNodeId: string;
  world: string;
  snapshot: VoidMapSnapshot;
  round: number;
  timestamp: number;
}

/**
 * Create a gossip message from this node's local void.
 */
export async function createGossipMessage(
  node: FederatedVoidNode
): Promise<GossipMessage> {
  const snapshot = await snapshotVoidMap(node.localMap);
  return {
    fromNodeId: node.nodeId,
    world: node.world,
    snapshot,
    round: node.localMap.round,
    timestamp: Date.now(),
  };
}

/**
 * Receive a gossip message from a neighbor.
 * Updates the neighbor's snapshot and invalidates the aggregate.
 */
export function receiveGossip(
  node: FederatedVoidNode,
  message: GossipMessage
): FederatedVoidNode {
  if (message.fromNodeId === node.nodeId) return node;

  const neighborSnapshots = new Map(node.neighborSnapshots);

  const existing = neighborSnapshots.get(message.fromNodeId);
  if (existing && existing.round >= message.snapshot.round) {
    return node;
  }

  neighborSnapshots.set(message.fromNodeId, message.snapshot);

  return {
    ...node,
    neighborSnapshots,
    aggregate: null,
    lastGossipRound: message.round,
    gossipHistory: [...node.gossipHistory, message.round],
  };
}

/**
 * Perform a gossip round: create a message and process incoming messages.
 */
export async function gossipRound(
  node: FederatedVoidNode,
  incomingMessages: readonly GossipMessage[]
): Promise<{
  updatedNode: FederatedVoidNode;
  outgoingMessage: GossipMessage;
}> {
  let updatedNode = node;
  for (const msg of incomingMessages) {
    updatedNode = receiveGossip(updatedNode, msg);
  }

  const outgoingMessage = await createGossipMessage(updatedNode);

  return { updatedNode, outgoingMessage };
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/**
 * Aggregate multiple void boundaries into collective stats.
 * Union tombstones (deduped by branchId+round), aggregate rejection counts,
 * compute collective entropy and Inverse Bule.
 */
export function aggregateVoidBoundaries(
  maps: readonly VoidMap[],
  options?: { eta?: number; maxTombstones?: number }
): VoidMap {
  const maxTombstones = options?.maxTombstones ?? 50_000;

  const seen = new Set<string>();
  const allTombstones: Tombstone[] = [];
  const aggregateCounts = new Map<string, number>();
  let maxRound = 0;

  for (const map of maps) {
    for (const t of map.tombstones) {
      const key = `${t.branchId}:${t.round}`;
      if (!seen.has(key)) {
        seen.add(key);
        allTombstones.push(t);
      }
    }

    for (const [branchId, count] of map.rejectionCounts) {
      aggregateCounts.set(
        branchId,
        (aggregateCounts.get(branchId) ?? 0) + count
      );
    }

    if (map.round > maxRound) maxRound = map.round;
  }

  // Sort by round and cap
  allTombstones.sort((a, b) => a.round - b.round);
  const capped = allTombstones.length > maxTombstones
    ? allTombstones.slice(allTombstones.length - maxTombstones)
    : allTombstones;

  return {
    tombstones: capped,
    round: maxRound,
    rejectionCounts: aggregateCounts,
  };
}

/**
 * Compute the aggregate for a federated node (local + all neighbors).
 */
export function computeNodeAggregate(
  node: FederatedVoidNode,
  options?: { eta?: number; maxTombstones?: number }
): FederatedVoidNode {
  const maps: VoidMap[] = [node.localMap];

  for (const snapshot of node.neighborSnapshots.values()) {
    maps.push(restoreFromSnapshot(snapshot));
  }

  const aggregate = aggregateVoidBoundaries(maps, options);

  return { ...node, aggregate };
}

// ---------------------------------------------------------------------------
// Cross-region analysis
// ---------------------------------------------------------------------------

export interface CrossRegionDiff {
  regionA: string;
  regionB: string;
  /** Jensen-Shannon divergence between complement distributions */
  jsDivergence: number;
  /** Branches only rejected in region A */
  uniqueToA: readonly string[];
  /** Branches only rejected in region B */
  uniqueToB: readonly string[];
  /** Branches rejected in both */
  shared: readonly string[];
  /** Is the diff mergeable? */
  mergeable: boolean;
  /** Cognitive distance metric (0 = identical rejection, 1 = maximally different) */
  cognitiveDistance: number;
}

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

/**
 * Compute the diff between two regions' aggregate void boundaries.
 */
export function crossRegionDiff(
  regionALabel: string,
  regionAMap: VoidMap,
  regionBLabel: string,
  regionBMap: VoidMap,
  options?: { eta?: number; mergeThreshold?: number }
): CrossRegionDiff {
  const eta = options?.eta ?? 3.0;
  const mergeThreshold = options?.mergeThreshold ?? 0.5;

  // Align distributions by branch ID for meaningful comparison
  const allBranchIds = Array.from(
    new Set([...regionAMap.rejectionCounts.keys(), ...regionBMap.rejectionCounts.keys()])
  ).sort();

  function computeAligned(map: VoidMap): number[] {
    if (allBranchIds.length === 0) return [];
    const counts = allBranchIds.map((b) => map.rejectionCounts.get(b) ?? 0);
    const logits = counts.map((v) => -eta * v);
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  const distA = computeAligned(regionAMap);
  const distB = computeAligned(regionBMap);
  const jsDivergence = jensenShannonDivergence(distA, distB);

  const branchesA = new Set(regionAMap.rejectionCounts.keys());
  const branchesB = new Set(regionBMap.rejectionCounts.keys());

  const uniqueToA = Array.from(branchesA).filter((b) => !branchesB.has(b));
  const uniqueToB = Array.from(branchesB).filter((b) => !branchesA.has(b));
  const shared = Array.from(branchesA).filter((b) => branchesB.has(b));

  return {
    regionA: regionALabel,
    regionB: regionBLabel,
    jsDivergence,
    uniqueToA,
    uniqueToB,
    shared,
    mergeable: jsDivergence < mergeThreshold,
    cognitiveDistance: Math.min(1, jsDivergence),
  };
}

// ---------------------------------------------------------------------------
// Global metrics (computed from aggregate)
// ---------------------------------------------------------------------------

export interface GlobalVoidMetrics {
  /** Total nodes contributing */
  nodeCount: number;
  /** Aggregate entropy */
  entropy: number;
  /** Aggregate Inverse Bule */
  inverseBule: number;
  /** Total unique tombstones across all nodes */
  totalTombstones: number;
  /** Total unique branches */
  totalBranches: number;
  /** Centroid of the aggregate void */
  centroid: Vec3;
  /** Spread of the aggregate void */
  spread: number;
}

/**
 * Compute global void metrics from a set of federated nodes.
 */
export function computeGlobalMetrics(
  nodes: readonly FederatedVoidNode[],
  options?: { eta?: number }
): GlobalVoidMetrics {
  const maps = nodes.map((n) => n.aggregate ?? n.localMap);
  const aggregate = aggregateVoidBoundaries(maps, options);
  const stats = computeVoidBoundary(aggregate, options);

  return {
    nodeCount: nodes.length,
    entropy: stats.entropy,
    inverseBule: stats.inverseBule,
    totalTombstones: stats.totalTombstones,
    totalBranches: stats.homologyRank,
    centroid: stats.centroid,
    spread: stats.spread,
  };
}

/**
 * Compute global centroid velocity from multiple time-series snapshots
 * of the global aggregate.
 */
export function globalCentroidVelocity(
  snapshots: readonly { round: number; centroid: Vec3 }[]
): readonly { round: number; velocity: Vec3; speed: number }[] {
  if (snapshots.length < 2) return [];

  const velocities: { round: number; velocity: Vec3; speed: number }[] = [];
  for (let i = 1; i < snapshots.length; i++) {
    const dt = snapshots[i].round - snapshots[i - 1].round;
    if (dt === 0) continue;
    const delta = subtractVec3(snapshots[i].centroid, snapshots[i - 1].centroid);
    const velocity = scaleVec3(delta, 1 / dt);
    velocities.push({
      round: snapshots[i].round,
      velocity,
      speed: lengthVec3(velocity),
    });
  }
  return velocities;
}

/**
 * Find death trajectories that appear across multiple nodes (civilizational patterns).
 * A strategy rejected by many nodes is a civilizational blind alley.
 */
export function collectiveDeathTrajectories(
  nodes: readonly FederatedVoidNode[],
  minNodeCount: number = 2
): readonly {
  branchId: string;
  nodeCount: number;
  totalRejections: number;
  nodeIds: readonly string[];
}[] {
  const branchNodeMap = new Map<string, Set<string>>();
  const branchRejections = new Map<string, number>();

  for (const node of nodes) {
    const map = node.aggregate ?? node.localMap;
    for (const [branchId, count] of map.rejectionCounts) {
      let nodes = branchNodeMap.get(branchId);
      if (!nodes) {
        nodes = new Set();
        branchNodeMap.set(branchId, nodes);
      }
      nodes.add(node.nodeId);
      branchRejections.set(branchId, (branchRejections.get(branchId) ?? 0) + count);
    }
  }

  const results: {
    branchId: string;
    nodeCount: number;
    totalRejections: number;
    nodeIds: readonly string[];
  }[] = [];

  for (const [branchId, nodeSet] of branchNodeMap) {
    if (nodeSet.size >= minNodeCount) {
      results.push({
        branchId,
        nodeCount: nodeSet.size,
        totalRejections: branchRejections.get(branchId) ?? 0,
        nodeIds: Array.from(nodeSet),
      });
    }
  }

  return results.sort((a, b) => b.nodeCount - a.nodeCount);
}

/**
 * Compute a civilizational prediction cone from global aggregate velocity.
 * The direction the global void is expanding is the direction humanity
 * is moving away from.
 */
export function civilizationalPredictionCone(
  velocityHistory: readonly { round: number; velocity: Vec3; speed: number }[],
  currentCentroid: Vec3,
  options?: { lookback?: number }
): {
  direction: Vec3;
  confidence: number;
  reach: number;
  convergenceRate: number;
} | null {
  const lookback = options?.lookback ?? 10;
  const recent = velocityHistory.slice(-lookback);
  if (recent.length < 2) return null;

  // Average velocity
  let avgVel: Vec3 = { x: 0, y: 0, z: 0 };
  for (const v of recent) {
    avgVel = addVec3(avgVel, v.velocity);
  }
  avgVel = scaleVec3(avgVel, 1 / recent.length);

  const speed = lengthVec3(avgVel);
  if (speed < 1e-10) return null;

  const direction = scaleVec3(avgVel, 1 / speed);

  // Confidence from consistency
  let cosSum = 0;
  for (const v of recent) {
    const vSpeed = lengthVec3(v.velocity);
    if (vSpeed < 1e-8) continue;
    const vNorm = scaleVec3(v.velocity, 1 / vSpeed);
    cosSum += direction.x * vNorm.x + direction.y * vNorm.y + direction.z * vNorm.z;
  }
  const confidence = Math.max(0, cosSum / recent.length);

  // Convergence rate: is speed decreasing? (negative = converging)
  const speeds = recent.map((v) => v.speed);
  let speedDelta = 0;
  for (let i = 1; i < speeds.length; i++) {
    speedDelta += speeds[i] - speeds[i - 1];
  }
  const convergenceRate = speeds.length > 1
    ? speedDelta / (speeds.length - 1)
    : 0;

  return {
    direction,
    confidence,
    reach: speed * lookback,
    convergenceRate,
  };
}

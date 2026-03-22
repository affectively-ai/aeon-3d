import type { Vec3 } from './index.js';
import { addVec3, scaleVec3, subtractVec3, lengthVec3, normalizeVec3 } from './index.js';
import type { VoidMap, VoidBoundaryStats, Tombstone } from './voidmap.js';
import {
  complementDistribution,
  computeVoidBoundary,
} from './voidmap.js';
import { sliceAtRound } from './voidmap-temporal.js';

// ---------------------------------------------------------------------------
// Centroid contrail: the white trail of the void's center of mass
// ---------------------------------------------------------------------------

export interface CentroidWaypoint {
  round: number;
  position: Vec3;
  entropy: number;
  inverseBule: number;
  tombstoneCount: number;
}

/**
 * Compute the centroid at every Nth round, producing a trail that shows
 * how the void's center of mass drifts as it accumulates rejections.
 * The trail itself is meaningful -- sudden jumps reveal phase transitions
 * in the rejection pattern.
 */
export function computeCentroidContrail(
  map: VoidMap,
  options?: { stepSize?: number; eta?: number }
): readonly CentroidWaypoint[] {
  const step = options?.stepSize ?? 1;
  const eta = options?.eta ?? 3.0;

  if (map.tombstones.length === 0 || map.round === 0) return [];

  const waypoints: CentroidWaypoint[] = [];

  for (let r = step; r <= map.round; r += step) {
    const slice = sliceAtRound(map, r);
    if (slice.tombstones.length === 0) continue;

    const stats = computeVoidBoundary(slice, { eta });
    waypoints.push({
      round: r,
      position: stats.centroid,
      entropy: stats.entropy,
      inverseBule: stats.inverseBule,
      tombstoneCount: stats.totalTombstones,
    });
  }

  // Always include the final round if not already there
  if (waypoints.length === 0 || waypoints[waypoints.length - 1].round !== map.round) {
    const stats = computeVoidBoundary(map, { eta });
    waypoints.push({
      round: map.round,
      position: stats.centroid,
      entropy: stats.entropy,
      inverseBule: stats.inverseBule,
      tombstoneCount: stats.totalTombstones,
    });
  }

  return waypoints;
}

/**
 * Compute centroid velocity (drift direction and magnitude per round).
 * Large velocity = the void is shifting, the rejection pattern is changing.
 * Near-zero velocity = convergence, the void has found its shape.
 */
export function centroidVelocity(
  contrail: readonly CentroidWaypoint[]
): readonly { round: number; velocity: Vec3; speed: number }[] {
  if (contrail.length < 2) return [];

  const velocities: { round: number; velocity: Vec3; speed: number }[] = [];
  for (let i = 1; i < contrail.length; i++) {
    const dt = contrail[i].round - contrail[i - 1].round;
    if (dt === 0) continue;
    const delta = subtractVec3(contrail[i].position, contrail[i - 1].position);
    const velocity = scaleVec3(delta, 1 / dt);
    velocities.push({
      round: contrail[i].round,
      velocity,
      speed: lengthVec3(velocity),
    });
  }
  return velocities;
}

// ---------------------------------------------------------------------------
// Ghost projections: predicting where future tombstones will land
// ---------------------------------------------------------------------------

export interface GhostPoint {
  branchId: string;
  position: Vec3;
  /** Probability this branch dies next round (from complement dist inverse) */
  deathProbability: number;
  /** How many times already rejected */
  priorRejections: number;
}

/**
 * Project ghost points: translucent predictions of where the next
 * tombstones will appear. Uses the complement distribution inverted --
 * branches with high rejection counts have high death probability
 * (the void predicts they'll fail again).
 *
 * Ghost positions are the branch's last known death position, nudged
 * forward by the centroid velocity.
 */
export function projectGhosts(
  map: VoidMap,
  options?: { eta?: number; minProbability?: number }
): readonly GhostPoint[] {
  const eta = options?.eta ?? 3.0;
  const minProbability = options?.minProbability ?? 0.05;

  if (map.tombstones.length === 0) return [];

  // Complement distribution: high weight = low rejection count = HEALTHY.
  // Invert it: death probability is proportional to rejection count.
  const entries = Array.from(map.rejectionCounts.entries());
  if (entries.length === 0) return [];

  const counts = entries.map(([, c]) => c);
  const maxCount = Math.max(...counts);
  if (maxCount === 0) return [];

  // Death probability: softmax(eta * count) -- opposite of complement
  const logits = counts.map((c) => eta * c);
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  const deathProbs = exps.map((e) => e / sum);

  // Find last known position for each branch
  const lastPosition = new Map<string, Vec3>();
  for (const t of map.tombstones) {
    lastPosition.set(t.branchId, t.position);
  }

  const ghosts: GhostPoint[] = [];
  for (let i = 0; i < entries.length; i++) {
    const [branchId, count] = entries[i];
    const prob = deathProbs[i];
    if (prob < minProbability) continue;

    const pos = lastPosition.get(branchId);
    if (!pos) continue;

    ghosts.push({
      branchId,
      position: pos,
      deathProbability: prob,
      priorRejections: count,
    });
  }

  return ghosts.sort((a, b) => b.deathProbability - a.deathProbability);
}

// ---------------------------------------------------------------------------
// Prediction cone: extrapolating the void boundary's expansion
// ---------------------------------------------------------------------------

export interface PredictionCone {
  /** Origin of the cone (current centroid) */
  apex: Vec3;
  /** Normalized direction the void is expanding toward */
  direction: Vec3;
  /** Half-angle in radians */
  halfAngle: number;
  /** Confidence: 0 = no data, 1 = very consistent expansion direction */
  confidence: number;
  /** Predicted reach distance (how far the void will extend) */
  reach: number;
}

/**
 * Fit a prediction cone showing where future rejections will likely land.
 * Uses the centroid contrail's velocity to determine expansion direction,
 * and spread of recent tombstones to determine cone angle.
 */
export function fitPredictionCone(
  map: VoidMap,
  options?: { eta?: number; lookbackRounds?: number }
): PredictionCone | null {
  const lookback = options?.lookbackRounds ?? 10;
  const eta = options?.eta ?? 3.0;

  const contrail = computeCentroidContrail(map, { eta });
  if (contrail.length < 3) return null;

  const velocities = centroidVelocity(contrail);
  if (velocities.length < 2) return null;

  // Average recent velocity to get expansion direction
  const recentVelocities = velocities.slice(-lookback);
  let avgVel: Vec3 = { x: 0, y: 0, z: 0 };
  for (const v of recentVelocities) {
    avgVel = addVec3(avgVel, v.velocity);
  }
  avgVel = scaleVec3(avgVel, 1 / recentVelocities.length);

  const speed = lengthVec3(avgVel);
  if (speed < 1e-8) return null;

  const direction = normalizeVec3(avgVel);

  // Confidence: how consistent is the direction?
  // Measured as average cosine similarity of velocities to the mean.
  let cosSum = 0;
  for (const v of recentVelocities) {
    const vSpeed = lengthVec3(v.velocity);
    if (vSpeed < 1e-8) continue;
    const vNorm = normalizeVec3(v.velocity);
    const cos = direction.x * vNorm.x + direction.y * vNorm.y + direction.z * vNorm.z;
    cosSum += cos;
  }
  const confidence = Math.max(0, cosSum / recentVelocities.length);

  // Half-angle: based on spread of recent tombstones relative to direction.
  // Wider spread = wider cone.
  const stats = computeVoidBoundary(map, { eta });
  const halfAngle = Math.atan2(stats.spread, speed * lookback + 1e-8);

  // Reach: extrapolate speed forward
  const reach = speed * lookback;

  return {
    apex: stats.centroid,
    direction,
    halfAngle: Math.min(halfAngle, Math.PI / 3), // cap at 60 degrees
    confidence,
    reach,
  };
}

// ---------------------------------------------------------------------------
// At-risk detection: which surviving branches are in the cone's path?
// ---------------------------------------------------------------------------

export interface AtRiskBranch {
  branchId: string;
  position: Vec3;
  /** Distance from the prediction cone's axis */
  coneDistance: number;
  /** 0 = on axis, 1 = at cone edge */
  riskFactor: number;
}

/**
 * Check which positions (surviving branch positions) fall inside
 * the prediction cone. These are the branches the void is heading toward --
 * the "precrime" moment: the void predicts their future rejection.
 */
export function detectAtRisk(
  cone: PredictionCone,
  branchPositions: ReadonlyMap<string, Vec3>
): readonly AtRiskBranch[] {
  const atRisk: AtRiskBranch[] = [];

  for (const [branchId, pos] of branchPositions) {
    // Vector from cone apex to branch
    const toPoint = subtractVec3(pos, cone.apex);
    const dist = lengthVec3(toPoint);
    if (dist < 1e-8) continue;

    // Project onto cone axis
    const axisProjection =
      toPoint.x * cone.direction.x +
      toPoint.y * cone.direction.y +
      toPoint.z * cone.direction.z;

    // Only consider points ahead of the apex and within reach
    if (axisProjection <= 0 || axisProjection > cone.reach) continue;

    // Perpendicular distance from axis
    const projectedPoint = scaleVec3(cone.direction, axisProjection);
    const perpendicular = subtractVec3(toPoint, projectedPoint);
    const perpDist = lengthVec3(perpendicular);

    // Cone radius at this depth
    const coneRadius = axisProjection * Math.tan(cone.halfAngle);
    if (perpDist > coneRadius) continue;

    const riskFactor = 1 - perpDist / (coneRadius + 1e-8);

    atRisk.push({
      branchId,
      position: pos,
      coneDistance: perpDist,
      riskFactor: Math.min(1, riskFactor * cone.confidence),
    });
  }

  return atRisk.sort((a, b) => b.riskFactor - a.riskFactor);
}

// ---------------------------------------------------------------------------
// Echo boundary bridge: convert EchoBoundary to GhostPoints
// ---------------------------------------------------------------------------

/**
 * Shape of an EchoBoundary from aeon-echo.
 * Inlined to avoid a hard dependency on @a0n/aeon-echo.
 */
export interface EchoBoundaryLike {
  readonly rejections: ReadonlyMap<string, number>;
  readonly totalMass: number;
  readonly dimensions: number;
}

/**
 * Convert an EchoBoundary into GhostPoints compatible with projectGhosts.
 * Each rejection dimension becomes a point -- coordinates derived from
 * a simple hash of the dimension name to distribute across 3D space.
 */
export function echoBoundaryToVoidPoints(boundary: EchoBoundaryLike): GhostPoint[] {
  const points: GhostPoint[] = [];
  const total = boundary.totalMass || 1;

  for (const [dim, count] of boundary.rejections) {
    // Simple string hash -> 3D position
    let h = 0;
    for (let i = 0; i < dim.length; i++) {
      h = ((h << 5) - h + dim.charCodeAt(i)) | 0;
    }
    const x = ((h & 0xff) / 128) - 1;
    const y = (((h >> 8) & 0xff) / 128) - 1;
    const z = (((h >> 16) & 0xff) / 128) - 1;

    points.push({
      branchId: dim,
      position: { x, y, z },
      deathProbability: count / total,
      priorRejections: count,
    });
  }

  return points.sort((a, b) => b.deathProbability - a.deathProbability);
}

// ---------------------------------------------------------------------------
// Phase transition detection: when the void's character changes
// ---------------------------------------------------------------------------

export interface PhaseTransition {
  round: number;
  /** Entropy before the transition */
  entropyBefore: number;
  /** Entropy after */
  entropyAfter: number;
  /** Magnitude of the entropy jump */
  entropyDelta: number;
  /** Centroid displacement */
  centroidShift: number;
  /** Type: 'convergence' = entropy dropping, 'disruption' = entropy spiking */
  type: 'convergence' | 'disruption';
}

/**
 * Detect phase transitions: moments where the void's entropy or centroid
 * undergoes a discontinuous jump. These are the "plot twists" in the
 * rejection history -- a new algorithm family suddenly starts failing,
 * or the budget changes and everything shifts.
 */
export function detectPhaseTransitions(
  contrail: readonly CentroidWaypoint[],
  options?: { entropyThreshold?: number; shiftThreshold?: number }
): readonly PhaseTransition[] {
  const entropyThreshold = options?.entropyThreshold ?? 0.3;
  const shiftThreshold = options?.shiftThreshold ?? 0.1;

  if (contrail.length < 2) return [];

  const transitions: PhaseTransition[] = [];
  for (let i = 1; i < contrail.length; i++) {
    const prev = contrail[i - 1];
    const curr = contrail[i];
    const entropyDelta = Math.abs(curr.entropy - prev.entropy);
    const centroidShift = lengthVec3(subtractVec3(curr.position, prev.position));

    if (entropyDelta > entropyThreshold || centroidShift > shiftThreshold) {
      transitions.push({
        round: curr.round,
        entropyBefore: prev.entropy,
        entropyAfter: curr.entropy,
        entropyDelta,
        centroidShift,
        type: curr.entropy < prev.entropy ? 'convergence' : 'disruption',
      });
    }
  }

  return transitions;
}

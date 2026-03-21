import React, { useMemo, useRef } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import type { Group } from 'three';

import type { Vec3 } from './index.js';
import { addVec3, scaleVec3 } from './index.js';
import type { VoidMap } from './voidmap.js';
import type {
  CentroidWaypoint,
  GhostPoint,
  PredictionCone,
  AtRiskBranch,
  PhaseTransition,
} from './voidmap-precognition.js';
import {
  computeCentroidContrail,
  projectGhosts,
  fitPredictionCone,
  detectAtRisk,
  detectPhaseTransitions,
} from './voidmap-precognition.js';

// ---------------------------------------------------------------------------
// Centroid contrail: the white trail
// ---------------------------------------------------------------------------

export interface CentroidContrailProps {
  voidMap: VoidMap;
  /** Compute centroid every N rounds (default 1) */
  stepSize?: number;
  color?: string;
  lineWidth?: number;
  /** Color the trail by entropy: low entropy = white, high = yellow */
  colorByEntropy?: boolean;
  scale?: number;
  eta?: number;
}

export function CentroidContrail({
  voidMap,
  stepSize = 1,
  color = '#ffffff',
  lineWidth = 2,
  colorByEntropy = false,
  scale = 1,
  eta = 3.0,
}: CentroidContrailProps) {
  const contrail = useMemo(
    () => computeCentroidContrail(voidMap, { stepSize, eta }),
    [voidMap, stepSize, eta]
  );

  const points = useMemo(
    () =>
      contrail.map(
        (wp) =>
          [wp.position.x * scale, wp.position.y * scale, wp.position.z * scale] as [
            number,
            number,
            number,
          ]
      ),
    [contrail, scale]
  );

  const vertexColors = useMemo(() => {
    if (!colorByEntropy || contrail.length === 0) return undefined;
    const maxEntropy = Math.max(...contrail.map((w) => w.entropy), 1e-8);
    return contrail.map((wp) => {
      const t = wp.entropy / maxEntropy;
      // White (low entropy, converged) -> Yellow (high entropy, uncertain)
      return `rgb(255, 255, ${Math.round(255 * (1 - t))})`;
    });
  }, [contrail, colorByEntropy]);

  if (points.length < 2) return null;

  return (
    <Line
      points={points}
      color={colorByEntropy ? undefined : color}
      vertexColors={colorByEntropy ? (vertexColors as unknown as Array<[number, number, number]>) : undefined}
      lineWidth={lineWidth}
    />
  );
}

// ---------------------------------------------------------------------------
// Ghost projections: translucent predicted tombstones
// ---------------------------------------------------------------------------

export interface GhostProjectionsProps {
  voidMap: VoidMap;
  pointSize?: number;
  /** Ghost color -- rendered translucent */
  color?: string;
  /** Only show ghosts above this death probability */
  minProbability?: number;
  scale?: number;
  eta?: number;
  /** Pulse animation: ghosts breathe in and out */
  animate?: boolean;
}

export function GhostProjections({
  voidMap,
  pointSize = 4,
  color = '#ef4444',
  minProbability = 0.05,
  scale = 1,
  eta = 3.0,
  animate = true,
}: GhostProjectionsProps) {
  const ghosts = useMemo(
    () => projectGhosts(voidMap, { eta, minProbability }),
    [voidMap, eta, minProbability]
  );

  const groupRef = useRef<Group>(null);
  const phaseRef = useRef(0);

  useFrame((_, delta) => {
    if (!animate || !groupRef.current) return;
    phaseRef.current += delta * 2;
    // Pulse opacity via scale
    const pulse = 0.7 + 0.3 * Math.sin(phaseRef.current);
    groupRef.current.scale.setScalar(pulse);
  });

  const positions = useMemo(() => {
    const pos = new Float32Array(ghosts.length * 3);
    for (let i = 0; i < ghosts.length; i++) {
      const p = ghosts[i].position;
      pos[i * 3] = p.x * scale;
      pos[i * 3 + 1] = p.y * scale;
      pos[i * 3 + 2] = p.z * scale;
    }
    return pos;
  }, [ghosts, scale]);

  if (ghosts.length === 0) return null;

  return (
    <group ref={groupRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color={color}
          size={pointSize}
          sizeAttenuation={false}
          opacity={0.4}
          transparent
        />
      </points>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Prediction cone: the precrime visualization
// ---------------------------------------------------------------------------

export interface PredictionConeProps {
  voidMap: VoidMap;
  color?: string;
  /** Number of segments for the cone wireframe */
  segments?: number;
  scale?: number;
  eta?: number;
  lookbackRounds?: number;
}

export function PredictionConeView({
  voidMap,
  color = '#ef444440',
  segments = 16,
  scale = 1,
  eta = 3.0,
  lookbackRounds = 10,
}: PredictionConeProps) {
  const cone = useMemo(
    () => fitPredictionCone(voidMap, { eta, lookbackRounds }),
    [voidMap, eta, lookbackRounds]
  );

  const coneLines = useMemo(() => {
    if (!cone) return [];

    const apex: [number, number, number] = [
      cone.apex.x * scale,
      cone.apex.y * scale,
      cone.apex.z * scale,
    ];

    // Generate ring points at the cone's base
    const baseCenter = addVec3(cone.apex, scaleVec3(cone.direction, cone.reach));
    const radius = cone.reach * Math.tan(cone.halfAngle);

    // Find two perpendicular vectors to the direction
    const up: Vec3 =
      Math.abs(cone.direction.y) < 0.9
        ? { x: 0, y: 1, z: 0 }
        : { x: 1, y: 0, z: 0 };
    const right: Vec3 = {
      x: cone.direction.y * up.z - cone.direction.z * up.y,
      y: cone.direction.z * up.x - cone.direction.x * up.z,
      z: cone.direction.x * up.y - cone.direction.y * up.x,
    };
    const rLen = Math.sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    const rNorm: Vec3 = rLen > 0
      ? { x: right.x / rLen, y: right.y / rLen, z: right.z / rLen }
      : { x: 1, y: 0, z: 0 };
    const upPerp: Vec3 = {
      x: cone.direction.y * rNorm.z - cone.direction.z * rNorm.y,
      y: cone.direction.z * rNorm.x - cone.direction.x * rNorm.z,
      z: cone.direction.x * rNorm.y - cone.direction.y * rNorm.x,
    };

    const lines: [number, number, number][][] = [];
    const ringPoints: [number, number, number][] = [];

    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const p: [number, number, number] = [
        (baseCenter.x + rNorm.x * cos * radius + upPerp.x * sin * radius) * scale,
        (baseCenter.y + rNorm.y * cos * radius + upPerp.y * sin * radius) * scale,
        (baseCenter.z + rNorm.z * cos * radius + upPerp.z * sin * radius) * scale,
      ];
      ringPoints.push(p);
      // Line from apex to ring point
      lines.push([apex, p]);
    }

    // Close the ring
    for (let i = 0; i < segments; i++) {
      lines.push([ringPoints[i], ringPoints[(i + 1) % segments]]);
    }

    return lines;
  }, [cone, scale, segments]);

  if (!cone || coneLines.length === 0) return null;

  return (
    <group>
      {coneLines.map((edge, i) => (
        <Line
          key={`cone-${i}`}
          points={edge}
          color={color}
          lineWidth={1}
          opacity={cone.confidence}
          transparent
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// At-risk indicators: halos around doomed branches
// ---------------------------------------------------------------------------

export interface AtRiskIndicatorsProps {
  voidMap: VoidMap;
  /** Map of surviving branch IDs to their 3D positions */
  branchPositions: ReadonlyMap<string, Vec3>;
  color?: string;
  /** Ring radius multiplier */
  ringSize?: number;
  scale?: number;
  eta?: number;
  lookbackRounds?: number;
  /** Pulse animation */
  animate?: boolean;
}

export function AtRiskIndicators({
  voidMap,
  branchPositions,
  color = '#ef4444',
  ringSize = 0.05,
  scale = 1,
  eta = 3.0,
  lookbackRounds = 10,
  animate = true,
}: AtRiskIndicatorsProps) {
  const atRisk = useMemo(() => {
    const cone = fitPredictionCone(voidMap, { eta, lookbackRounds });
    if (!cone) return [];
    return detectAtRisk(cone, branchPositions);
  }, [voidMap, branchPositions, eta, lookbackRounds]);

  const groupRef = useRef<Group>(null);
  const phaseRef = useRef(0);

  useFrame((_, delta) => {
    if (!animate || !groupRef.current) return;
    phaseRef.current += delta * 3;
  });

  if (atRisk.length === 0) return null;

  return (
    <group ref={groupRef}>
      {atRisk.map((branch) => {
        const opacity = 0.3 + branch.riskFactor * 0.5;
        const size = ringSize * (1 + branch.riskFactor);

        return (
          <mesh
            key={branch.branchId}
            position={[
              branch.position.x * scale,
              branch.position.y * scale,
              branch.position.z * scale,
            ]}
          >
            <ringGeometry args={[size * 0.8, size, 16]} />
            <meshBasicMaterial
              color={color}
              opacity={opacity}
              transparent
              side={2} // DoubleSide
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Phase transition markers: vertical lines at disruption/convergence points
// ---------------------------------------------------------------------------

export interface PhaseTransitionMarkersProps {
  voidMap: VoidMap;
  convergenceColor?: string;
  disruptionColor?: string;
  /** Height of the marker lines */
  markerHeight?: number;
  scale?: number;
  eta?: number;
  entropyThreshold?: number;
}

export function PhaseTransitionMarkers({
  voidMap,
  convergenceColor = '#22c55e',
  disruptionColor = '#f59e0b',
  markerHeight = 1,
  scale = 1,
  eta = 3.0,
  entropyThreshold = 0.3,
}: PhaseTransitionMarkersProps) {
  const { contrail, transitions } = useMemo(() => {
    const c = computeCentroidContrail(voidMap, { eta });
    const t = detectPhaseTransitions(c, { entropyThreshold });
    return { contrail: c, transitions: t };
  }, [voidMap, eta, entropyThreshold]);

  // Map rounds to centroid positions
  const roundToPos = useMemo(() => {
    const m = new Map<number, Vec3>();
    for (const wp of contrail) m.set(wp.round, wp.position);
    return m;
  }, [contrail]);

  if (transitions.length === 0) return null;

  return (
    <group>
      {transitions.map((t) => {
        const pos = roundToPos.get(t.round);
        if (!pos) return null;

        const color = t.type === 'convergence' ? convergenceColor : disruptionColor;
        const x = pos.x * scale;
        const y = pos.y * scale;
        const z = pos.z * scale;

        return (
          <Line
            key={`phase-${t.round}`}
            points={[
              [x, y - markerHeight * 0.5, z],
              [x, y + markerHeight * 0.5, z],
            ]}
            color={color}
            lineWidth={2}
          />
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Composite: all precognition layers in one component
// ---------------------------------------------------------------------------

export interface VoidPrecognitionProps {
  voidMap: VoidMap;
  /** Surviving branch positions for at-risk detection */
  branchPositions?: ReadonlyMap<string, Vec3>;
  showContrail?: boolean;
  showGhosts?: boolean;
  showCone?: boolean;
  showAtRisk?: boolean;
  showPhaseTransitions?: boolean;
  scale?: number;
  eta?: number;
  animate?: boolean;
}

export function VoidPrecognition({
  voidMap,
  branchPositions,
  showContrail = true,
  showGhosts = true,
  showCone = true,
  showAtRisk = true,
  showPhaseTransitions = true,
  scale = 1,
  eta = 3.0,
  animate = true,
}: VoidPrecognitionProps) {
  return (
    <group>
      {showContrail && (
        <CentroidContrail
          voidMap={voidMap}
          colorByEntropy
          scale={scale}
          eta={eta}
        />
      )}
      {showGhosts && (
        <GhostProjections
          voidMap={voidMap}
          scale={scale}
          eta={eta}
          animate={animate}
        />
      )}
      {showCone && (
        <PredictionConeView
          voidMap={voidMap}
          scale={scale}
          eta={eta}
        />
      )}
      {showAtRisk && branchPositions && (
        <AtRiskIndicators
          voidMap={voidMap}
          branchPositions={branchPositions}
          scale={scale}
          eta={eta}
          animate={animate}
        />
      )}
      {showPhaseTransitions && (
        <PhaseTransitionMarkers
          voidMap={voidMap}
          scale={scale}
          eta={eta}
        />
      )}
    </group>
  );
}

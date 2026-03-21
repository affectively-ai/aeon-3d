import React, { useMemo, useCallback, useRef, useState } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

import type { Vec3, TopologyFrameInput, TopologyFrameResult } from './index.js';
import type {
  VoidMap,
  VoidMapOptions,
  Tombstone,
  VoidBoundaryStats,
} from './voidmap.js';
import { createVoidMap, recordFrame, computeVoidBoundary } from './voidmap.js';
import type { VoidTimeline, VoidFork, VoidDiff, DeathTrajectory } from './voidmap-temporal.js';
import {
  sliceAtRound,
  computeOpacities,
  extractTrajectories,
  diffVoidMaps,
  createTimeline,
  forkAt,
  updateTrunk,
  updateFork,
  mergeToTrunk,
} from './voidmap-temporal.js';

// ---------------------------------------------------------------------------
// Temporal VoidMap3D: scrubbing + aging + trajectories
// ---------------------------------------------------------------------------

export interface TemporalVoidMap3DProps {
  voidMap: VoidMap;
  /** Current scrub position. Defaults to map.round (latest). */
  viewRound?: number;
  pointSize?: number;
  /** Color for fresh tombstones */
  pointColor?: string;
  /** Color for aged-out tombstones */
  fadedColor?: string;
  showCentroid?: boolean;
  centroidColor?: string;
  showBoundary?: boolean;
  boundaryColor?: string;
  /** Show death trajectory lines for repeat-rejected branches */
  showTrajectories?: boolean;
  trajectoryColor?: string;
  /** Min number of rejections before a trajectory is shown */
  trajectoryMinRejections?: number;
  /** Opacity decay rate. Higher = faster fade. */
  decayRate?: number;
  /** Minimum opacity for oldest tombstones */
  minOpacity?: number;
  scale?: number;
  onTombstoneHover?: (tombstone: Tombstone | null) => void;
}

export function TemporalVoidMap3D({
  voidMap,
  viewRound,
  pointSize = 3,
  pointColor = '#ef4444',
  fadedColor = '#666666',
  showCentroid = true,
  centroidColor = '#ffffff',
  showBoundary = false,
  boundaryColor = '#ef444440',
  showTrajectories = true,
  trajectoryColor = '#ef444480',
  trajectoryMinRejections = 3,
  decayRate = 0.05,
  minOpacity = 0.1,
  scale = 1,
  onTombstoneHover,
}: TemporalVoidMap3DProps) {
  const currentRound = viewRound ?? voidMap.round;

  const { slicedMap, positions, colors, stats, boundaryLines, trajectories } =
    useMemo(() => {
      const sliced = sliceAtRound(voidMap, currentRound);
      const tombstones = sliced.tombstones;
      const len = tombstones.length;

      const pos = new Float32Array(len * 3);
      const col = new Float32Array(len * 3);

      // Parse hex colors
      const fresh = hexToRgb(pointColor);
      const faded = hexToRgb(fadedColor);

      for (let i = 0; i < len; i++) {
        const t = tombstones[i];
        const p = t.position;
        pos[i * 3] = p.x * scale;
        pos[i * 3 + 1] = p.y * scale;
        pos[i * 3 + 2] = p.z * scale;

        // Opacity-based color interpolation (lerp fresh -> faded by age)
        const age = Math.max(0, currentRound - t.round);
        const alpha =
          minOpacity + (1 - minOpacity) * Math.exp(-decayRate * age);
        col[i * 3] = faded.r + (fresh.r - faded.r) * alpha;
        col[i * 3 + 1] = faded.g + (fresh.g - faded.g) * alpha;
        col[i * 3 + 2] = faded.b + (fresh.b - faded.b) * alpha;
      }

      const s = computeVoidBoundary(sliced);

      // AABB
      let lines: [number, number, number][][] = [];
      if (len > 0) {
        let minX = Infinity,
          minY = Infinity,
          minZ = Infinity;
        let maxX = -Infinity,
          maxY = -Infinity,
          maxZ = -Infinity;
        for (let i = 0; i < len; i++) {
          const px = pos[i * 3],
            py = pos[i * 3 + 1],
            pz = pos[i * 3 + 2];
          if (px < minX) minX = px;
          if (py < minY) minY = py;
          if (pz < minZ) minZ = pz;
          if (px > maxX) maxX = px;
          if (py > maxY) maxY = py;
          if (pz > maxZ) maxZ = pz;
        }
        const c: [number, number, number][] = [
          [minX, minY, minZ],
          [maxX, minY, minZ],
          [maxX, maxY, minZ],
          [minX, maxY, minZ],
          [minX, minY, maxZ],
          [maxX, minY, maxZ],
          [maxX, maxY, maxZ],
          [minX, maxY, maxZ],
        ];
        lines = [
          [c[0], c[1]], [c[1], c[2]], [c[2], c[3]], [c[3], c[0]],
          [c[4], c[5]], [c[5], c[6]], [c[6], c[7]], [c[7], c[4]],
          [c[0], c[4]], [c[1], c[5]], [c[2], c[6]], [c[3], c[7]],
        ];
      }

      // Death trajectories (only from the sliced view)
      const trajs = showTrajectories
        ? extractTrajectories(sliced).filter(
            (t) => t.rejectionCount >= trajectoryMinRejections
          )
        : [];

      return {
        slicedMap: sliced,
        positions: pos,
        colors: col,
        stats: s,
        boundaryLines: lines,
        trajectories: trajs,
      };
    }, [
      voidMap,
      currentRound,
      pointColor,
      fadedColor,
      decayRate,
      minOpacity,
      scale,
      showTrajectories,
      trajectoryMinRejections,
    ]);

  const handlePointerMove = useCallback(
    (event: { index?: number }) => {
      if (!onTombstoneHover || typeof event.index !== 'number') return;
      const idx = event.index;
      if (idx >= 0 && idx < slicedMap.tombstones.length) {
        onTombstoneHover(slicedMap.tombstones[idx]);
      }
    },
    [onTombstoneHover, slicedMap.tombstones]
  );

  const handlePointerLeave = useCallback(() => {
    onTombstoneHover?.(null);
  }, [onTombstoneHover]);

  if (slicedMap.tombstones.length === 0) return null;

  return (
    <group>
      {/* Aged tombstone point cloud with per-vertex color */}
      <points
        onPointerMove={onTombstoneHover ? handlePointerMove : undefined}
        onPointerLeave={onTombstoneHover ? handlePointerLeave : undefined}
      >
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          size={pointSize}
          sizeAttenuation={false}
        />
      </points>

      {/* Centroid */}
      {showCentroid && (
        <mesh
          position={[
            stats.centroid.x * scale,
            stats.centroid.y * scale,
            stats.centroid.z * scale,
          ]}
        >
          <sphereGeometry args={[pointSize * 0.02, 8, 8]} />
          <meshBasicMaterial color={centroidColor} />
        </mesh>
      )}

      {/* Boundary wireframe */}
      {showBoundary &&
        boundaryLines.map((edge, i) => (
          <Line
            key={`boundary-${i}`}
            points={edge}
            color={boundaryColor}
            lineWidth={1}
          />
        ))}

      {/* Death trajectories */}
      {trajectories.map((traj) => (
        <Line
          key={`traj-${traj.branchId}`}
          points={traj.positions.map(
            (p) => [p.x * scale, p.y * scale, p.z * scale] as [number, number, number]
          )}
          color={trajectoryColor}
          lineWidth={1.5}
          dashed
          dashSize={0.02}
          gapSize={0.01}
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Parallel universe split view: two forks side by side
// ---------------------------------------------------------------------------

export interface VoidDiffViewProps {
  left: VoidMap;
  right: VoidMap;
  leftLabel?: string;
  rightLabel?: string;
  /** Horizontal separation between the two clouds */
  separation?: number;
  leftColor?: string;
  rightColor?: string;
  sharedColor?: string;
  showDiffLines?: boolean;
  scale?: number;
  eta?: number;
  mergeThreshold?: number;
}

export function VoidDiffView({
  left,
  right,
  separation = 2,
  leftColor = '#ef4444',
  rightColor = '#3b82f6',
  sharedColor = '#a855f7',
  showDiffLines = true,
  scale = 1,
  eta = 3.0,
  mergeThreshold = 0.5,
}: VoidDiffViewProps) {
  const { diff, leftPositions, leftColors, rightPositions, rightColors, diffLines } =
    useMemo(() => {
      const d = diffVoidMaps(left, right, { eta, mergeThreshold });
      const halfSep = separation / 2;

      // Left cloud: shared (purple) + onlyInA (red), offset to -X
      const leftTombstones = [...d.shared, ...d.onlyInA];
      const lPos = new Float32Array(leftTombstones.length * 3);
      const lCol = new Float32Array(leftTombstones.length * 3);
      const sharedKeys = new Set(d.shared.map((t) => `${t.branchId}:${t.round}`));
      const sharedRgb = hexToRgb(sharedColor);
      const leftRgb = hexToRgb(leftColor);

      for (let i = 0; i < leftTombstones.length; i++) {
        const t = leftTombstones[i];
        lPos[i * 3] = t.position.x * scale - halfSep;
        lPos[i * 3 + 1] = t.position.y * scale;
        lPos[i * 3 + 2] = t.position.z * scale;

        const isShared = sharedKeys.has(`${t.branchId}:${t.round}`);
        const rgb = isShared ? sharedRgb : leftRgb;
        lCol[i * 3] = rgb.r;
        lCol[i * 3 + 1] = rgb.g;
        lCol[i * 3 + 2] = rgb.b;
      }

      // Right cloud: shared (purple) + onlyInB (blue), offset to +X
      const rightTombstones = [...d.shared, ...d.onlyInB];
      const rPos = new Float32Array(rightTombstones.length * 3);
      const rCol = new Float32Array(rightTombstones.length * 3);
      const rightRgb = hexToRgb(rightColor);

      for (let i = 0; i < rightTombstones.length; i++) {
        const t = rightTombstones[i];
        rPos[i * 3] = t.position.x * scale + halfSep;
        rPos[i * 3 + 1] = t.position.y * scale;
        rPos[i * 3 + 2] = t.position.z * scale;

        const isShared = sharedKeys.has(`${t.branchId}:${t.round}`);
        const rgb = isShared ? sharedRgb : rightRgb;
        rCol[i * 3] = rgb.r;
        rCol[i * 3 + 1] = rgb.g;
        rCol[i * 3 + 2] = rgb.b;
      }

      // Diff lines: connect shared tombstones across the split
      const lines: { from: [number, number, number]; to: [number, number, number] }[] = [];
      if (showDiffLines) {
        for (const t of d.shared) {
          lines.push({
            from: [t.position.x * scale - halfSep, t.position.y * scale, t.position.z * scale],
            to: [t.position.x * scale + halfSep, t.position.y * scale, t.position.z * scale],
          });
        }
      }

      return {
        diff: d,
        leftPositions: lPos,
        leftColors: lCol,
        rightPositions: rPos,
        rightColors: rCol,
        diffLines: lines,
      };
    }, [left, right, separation, leftColor, rightColor, sharedColor, showDiffLines, scale, eta, mergeThreshold]);

  return (
    <group>
      {/* Left universe */}
      {leftPositions.length > 0 && (
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[leftPositions, 3]} />
            <bufferAttribute attach="attributes-color" args={[leftColors, 3]} />
          </bufferGeometry>
          <pointsMaterial vertexColors size={3} sizeAttenuation={false} />
        </points>
      )}

      {/* Right universe */}
      {rightPositions.length > 0 && (
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[rightPositions, 3]} />
            <bufferAttribute attach="attributes-color" args={[rightColors, 3]} />
          </bufferGeometry>
          <pointsMaterial vertexColors size={3} sizeAttenuation={false} />
        </points>
      )}

      {/* Shared tombstone connection lines */}
      {diffLines.map((line, i) => (
        <Line
          key={`diff-${i}`}
          points={[line.from, line.to]}
          color={sharedColor}
          lineWidth={0.5}
          opacity={0.3}
          transparent
        />
      ))}

      {/* Unmergeable indicator: red boundary if JS divergence exceeds threshold */}
      {!diff.mergeable && (
        <mesh position={[0, 0.5 * scale, 0]}>
          <planeGeometry args={[0.01, scale]} />
          <meshBasicMaterial color="#ef4444" opacity={0.6} transparent />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// useTemporalVoidMap: full hook with scrubbing + forking
// ---------------------------------------------------------------------------

export interface UseTemporalVoidMapReturn {
  timeline: VoidTimeline;
  /** Current scrub position */
  viewRound: number;
  /** Set scrub position (0 to timeline.trunk.round) */
  scrubTo: (round: number) => void;
  /** Sliced view at current scrub position */
  currentView: VoidMap;
  /** Stats for the current view */
  stats: VoidBoundaryStats;
  /** Record a frame to the trunk */
  record: (input: TopologyFrameInput, result: TopologyFrameResult) => void;
  /** Record a frame to a specific fork */
  recordToFork: (forkId: string, input: TopologyFrameInput, result: TopologyFrameResult) => void;
  /** Fork the trunk at current or specified round */
  fork: (id: string, label: string, round?: number) => void;
  /** Attempt to merge a fork back into trunk */
  merge: (forkId: string) => VoidDiff;
  /** Diff a fork against the trunk */
  diff: (forkId: string) => VoidDiff;
  /** Reset everything */
  reset: () => void;
}

export function useTemporalVoidMap(
  options?: VoidMapOptions & { mergeThreshold?: number }
): UseTemporalVoidMapReturn {
  const optsRef = useRef(options);
  optsRef.current = options;

  const [timeline, setTimeline] = useState<VoidTimeline>(() => createTimeline());
  const [viewRound, setViewRound] = useState(0);

  const currentView = useMemo(
    () => sliceAtRound(timeline.trunk, viewRound),
    [timeline.trunk, viewRound]
  );

  const stats = useMemo(
    () => computeVoidBoundary(currentView, optsRef.current),
    [currentView]
  );

  const scrubTo = useCallback((round: number) => {
    setViewRound(Math.max(0, round));
  }, []);

  const record = useCallback(
    (input: TopologyFrameInput, result: TopologyFrameResult) => {
      setTimeline((prev) => {
        const newTrunk = recordFrame(prev.trunk, input, result, optsRef.current);
        setViewRound(newTrunk.round);
        return updateTrunk(prev, newTrunk);
      });
    },
    []
  );

  const recordToFork = useCallback(
    (forkId: string, input: TopologyFrameInput, result: TopologyFrameResult) => {
      setTimeline((prev) => {
        const fork = prev.forks.find((f) => f.id === forkId);
        if (!fork) return prev;
        const newMap = recordFrame(fork.map, input, result, optsRef.current);
        return updateFork(prev, forkId, newMap);
      });
    },
    []
  );

  const forkFn = useCallback(
    (id: string, label: string, round?: number) => {
      setTimeline((prev) => forkAt(prev, id, label, round));
    },
    []
  );

  const mergeFn = useCallback(
    (forkId: string): VoidDiff => {
      let resultDiff: VoidDiff = {
        onlyInA: [],
        onlyInB: [],
        shared: [],
        rejectionDelta: new Map(),
        jsDivergence: 0,
        mergeable: false,
      };
      setTimeline((prev) => {
        const result = mergeToTrunk(prev, forkId, {
          eta: optsRef.current?.eta,
          mergeThreshold: optsRef.current?.mergeThreshold,
        });
        resultDiff = result.diff;
        if (result.merged) {
          setViewRound(result.timeline.trunk.round);
        }
        return result.timeline;
      });
      return resultDiff;
    },
    []
  );

  const diffFn = useCallback(
    (forkId: string): VoidDiff => {
      const fork = timeline.forks.find((f) => f.id === forkId);
      if (!fork) {
        return {
          onlyInA: [],
          onlyInB: [],
          shared: [],
          rejectionDelta: new Map(),
          jsDivergence: 0,
          mergeable: false,
        };
      }
      return diffVoidMaps(timeline.trunk, fork.map, {
        eta: optsRef.current?.eta,
        mergeThreshold: optsRef.current?.mergeThreshold,
      });
    },
    [timeline]
  );

  const reset = useCallback(() => {
    setTimeline(createTimeline());
    setViewRound(0);
  }, []);

  return {
    timeline,
    viewRound,
    scrubTo,
    currentView,
    stats,
    record,
    recordToFork,
    fork: forkFn,
    merge: mergeFn,
    diff: diffFn,
    reset,
  };
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const clean = hex.replace('#', '').slice(0, 6);
  const num = parseInt(clean, 16);
  return {
    r: ((num >> 16) & 0xff) / 255,
    g: ((num >> 8) & 0xff) / 255,
    b: (num & 0xff) / 255,
  };
}

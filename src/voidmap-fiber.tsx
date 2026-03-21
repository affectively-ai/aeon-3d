import React, { useMemo, useCallback, useRef, useState } from 'react';
import { Line } from '@react-three/drei';
import type { ThreeEvent } from '@react-three/fiber';

import type { Vec3 } from './index.js';
import type {
  Tombstone,
  VoidMap,
  VoidBoundaryStats,
  VoidMapOptions,
} from './voidmap.js';
import type { TopologyFrameInput, TopologyFrameResult } from './index.js';
import {
  createVoidMap,
  recordFrame,
  computeVoidBoundary,
} from './voidmap.js';

// ---------------------------------------------------------------------------
// VoidMap3D component
// ---------------------------------------------------------------------------

export interface VoidMap3DProps {
  voidMap: VoidMap;
  pointSize?: number;
  pointColor?: string;
  showCentroid?: boolean;
  centroidColor?: string;
  showBoundary?: boolean;
  boundaryColor?: string;
  scale?: number;
  onTombstoneHover?: (tombstone: Tombstone | null) => void;
}

export function VoidMap3D({
  voidMap,
  pointSize = 3,
  pointColor = '#ef4444',
  showCentroid = true,
  centroidColor = '#ffffff',
  showBoundary = false,
  boundaryColor = '#ef444440',
  scale = 1,
  onTombstoneHover,
}: VoidMap3DProps) {
  const { positions, stats, boundaryLines } = useMemo(() => {
    const tombstones = voidMap.tombstones;
    const len = tombstones.length;
    const pos = new Float32Array(len * 3);

    for (let i = 0; i < len; i++) {
      const p = tombstones[i].position;
      pos[i * 3] = p.x * scale;
      pos[i * 3 + 1] = p.y * scale;
      pos[i * 3 + 2] = p.z * scale;
    }

    const s = computeVoidBoundary(voidMap);

    // Axis-aligned bounding box for boundary wireframe
    let lines: [number, number, number][][] = [];
    if (len > 0) {
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = 0; i < len; i++) {
        const p = tombstones[i].position;
        const px = p.x * scale, py = p.y * scale, pz = p.z * scale;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (pz < minZ) minZ = pz;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
        if (pz > maxZ) maxZ = pz;
      }

      // 12 edges of the AABB
      const c: [number, number, number][] = [
        [minX, minY, minZ], [maxX, minY, minZ],
        [maxX, maxY, minZ], [minX, maxY, minZ],
        [minX, minY, maxZ], [maxX, minY, maxZ],
        [maxX, maxY, maxZ], [minX, maxY, maxZ],
      ];
      lines = [
        // Bottom face
        [c[0], c[1]], [c[1], c[2]], [c[2], c[3]], [c[3], c[0]],
        // Top face
        [c[4], c[5]], [c[5], c[6]], [c[6], c[7]], [c[7], c[4]],
        // Vertical edges
        [c[0], c[4]], [c[1], c[5]], [c[2], c[6]], [c[3], c[7]],
      ];
    }

    return { positions: pos, stats: s, boundaryLines: lines };
  }, [voidMap, scale]);

  const handlePointerMove = useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      if (!onTombstoneHover || typeof event.index !== 'number') return;
      const idx = event.index;
      if (idx >= 0 && idx < voidMap.tombstones.length) {
        onTombstoneHover(voidMap.tombstones[idx]);
      }
    },
    [onTombstoneHover, voidMap.tombstones]
  );

  const handlePointerLeave = useCallback(() => {
    onTombstoneHover?.(null);
  }, [onTombstoneHover]);

  if (voidMap.tombstones.length === 0) return null;

  return (
    <group>
      {/* Tombstone point cloud */}
      <points
        onPointerMove={onTombstoneHover ? handlePointerMove : undefined}
        onPointerLeave={onTombstoneHover ? handlePointerLeave : undefined}
      >
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color={pointColor}
          size={pointSize}
          sizeAttenuation={false}
        />
      </points>

      {/* Centroid marker */}
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
            key={i}
            points={edge}
            color={boundaryColor}
            lineWidth={1}
          />
        ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// useVoidMap hook
// ---------------------------------------------------------------------------

export interface UseVoidMapReturn {
  voidMap: VoidMap;
  stats: VoidBoundaryStats;
  record: (input: TopologyFrameInput, result: TopologyFrameResult) => void;
  reset: () => void;
}

export function useVoidMap(options?: VoidMapOptions): UseVoidMapReturn {
  const optsRef = useRef(options);
  optsRef.current = options;

  const [voidMap, setVoidMap] = useState<VoidMap>(() =>
    createVoidMap(options)
  );

  const stats = useMemo(
    () => computeVoidBoundary(voidMap, optsRef.current),
    [voidMap]
  );

  const record = useCallback(
    (input: TopologyFrameInput, result: TopologyFrameResult) => {
      setVoidMap((prev) => recordFrame(prev, input, result, optsRef.current));
    },
    []
  );

  const reset = useCallback(() => {
    setVoidMap(createVoidMap(optsRef.current));
  }, []);

  return { voidMap, stats, record, reset };
}

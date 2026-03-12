# Aeon 3D

Topology-first 3D compatibility layer for replacing Three.js hot paths with deterministic `fork/race/fold/vent` execution.

## Why

Traditional frame loops force a sequential render pipeline even when work is naturally parallel. Aeon 3D models each frame as a computation topology:

- `FORK`: split geometry/material/postprocessing branches
- `RACE`: choose fastest valid branch when latency is primary
- `FOLD`: deterministically merge branch outputs into a frame bundle
- `VENT`: shed non-critical branches under pressure

## API

```ts
import { executeTopologyFrame } from '@affectively/aeon-3d';

const result = await executeTopologyFrame({
  branches: [
    { id: 'geometry', vertices: 120_000, drawCalls: 120, estimatedCostMs: 6.2 },
    { id: 'instanced-geometry', vertices: 40_000, drawCalls: 24, estimatedCostMs: 2.1 }
  ],
  strategy: 'race',
  budgetMs: 8
});
```

## Compatibility Entrypoints

The package exposes migration-safe module entrypoints:

- `@affectively/aeon-3d/three`
- `@affectively/aeon-3d/fiber`
- `@affectively/aeon-3d/drei`

These allow a global import migration now while the runtime is progressively rewritten away from Three.js internals.

## Status

Bootstrap compat surface for Gnosis and Aeon web runtimes.

# Aeon 3D

Aeon 3D is a small compatibility layer for trying topology-style execution in 3D rendering work. It is aimed at places where a normal frame loop feels too rigid and you want to model a frame as several branches that can be kept, dropped, or chosen between.

The fair brag is modest and real: this package already gives you a simple execution surface and migration-safe entrypoints for `three`, `fiber`, and `drei`.

## What It Helps You Do

- split frame work into branches,
- choose the fastest acceptable branch when latency matters,
- merge surviving branches when you want a fuller result,
- and drop non-critical work when you are over budget.

That maps cleanly onto the package vocabulary:

- `FORK`: try more than one branch
- `RACE`: keep the fastest acceptable branch
- `FOLD`: merge surviving branches
- `VENT`: shed work under pressure

## Quick Example

```ts
import { executeTopologyFrame } from '@a0n/aeon-3d';

const result = await executeTopologyFrame({
  branches: [
    { id: 'geometry', vertices: 120_000, drawCalls: 120, estimatedCostMs: 6.2 },
    { id: 'instanced-geometry', vertices: 40_000, drawCalls: 24, estimatedCostMs: 2.1 },
  ],
  strategy: 'race',
  budgetMs: 8,
});
```

The result tells you:

- which strategy actually won,
- which branch survived,
- what was dropped,
- and whether the final work still exceeded budget.

## Compatibility Entrypoints

The package exposes migration-friendly entrypoints:

- `@a0n/aeon-3d`
- `@a0n/aeon-3d/three`
- `@a0n/aeon-3d/fiber`
- `@a0n/aeon-3d/drei`

That is useful even at this early stage. It gives you a stable place to start integration work before deeper runtime changes arrive.

## Status

Aeon 3D is early. The repo is intentionally small, the API surface is still compact, and there are no standalone checks configured yet.

That said, the package already has a clear job and a clear shape, which is often the hardest part to get right early.

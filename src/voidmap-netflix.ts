/**
 * voidmap-netflix.ts
 *
 * Bridge between the ch17 Netflix void walker and aeon-3d's VoidMap.
 * Converts algorithm family races into TopologyFrameInput/Result pairs
 * that feed the VoidMap, producing a concrete demo topology where
 * 5 algorithm families compete across 8 latent taste dimensions.
 *
 * This is the demo data for the Minority Report visualization:
 * scrub through 20,000 ratings and watch the void learn which
 * algorithms fail for which users.
 */

import type {
  TopologyBranch,
  TopologyFrameInput,
  TopologyFrameResult,
} from './index.js';
import type { VoidMap, VoidMapOptions } from './voidmap.js';
import { createVoidMap, recordFrame } from './voidmap.js';

// ---------------------------------------------------------------------------
// Seeded PRNG (matches ch17 for reproducibility)
// ---------------------------------------------------------------------------

class SeededRNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed | 0 || 1;
  }

  next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    this.state = x;
    return (x >>> 0) / 0xffffffff;
  }

  gaussian(): number {
    const u1 = this.next();
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
}

// ---------------------------------------------------------------------------
// Netflix algorithm families as topology branches
// ---------------------------------------------------------------------------

interface AlgorithmFamily {
  name: string;
  shortLabel: string;
  visibility: readonly boolean[];
  noiseStd: number;
}

interface UserTaste {
  id: number;
  factors: readonly number[];
}

interface MovieProfile {
  id: number;
  factors: readonly number[];
}

interface Rating {
  userId: number;
  movieId: number;
  actual: number;
}

function createFamilies(latentDims: number): AlgorithmFamily[] {
  const families: AlgorithmFamily[] = [
    {
      name: 'Global MF (like SVD)',
      shortLabel: 'MF',
      visibility: Array.from({ length: latentDims }, (_, i) => i < Math.ceil(latentDims / 2)),
      noiseStd: 0.15,
    },
    {
      name: 'Local Neighborhood (like k-NN)',
      shortLabel: 'kNN',
      visibility: Array.from({ length: latentDims }, (_, i) => i >= Math.floor(latentDims / 2)),
      noiseStd: 0.18,
    },
    {
      name: 'Nonlinear (like RBM)',
      shortLabel: 'RBM',
      visibility: Array.from({ length: latentDims }, (_, i) => i % 2 === 0),
      noiseStd: 0.2,
    },
    {
      name: 'Temporal (like timeSVD++)',
      shortLabel: 'tSVD',
      visibility: Array.from({ length: latentDims }, (_, i) => i % 2 === 1),
      noiseStd: 0.17,
    },
  ];

  if (latentDims >= 5) {
    families.push({
      name: 'Factored (like NNMF)',
      shortLabel: 'NNMF',
      visibility: Array.from({ length: latentDims }, (_, i) => i % 3 === 0),
      noiseStd: 0.22,
    });
  }

  return families;
}

function predict(
  user: UserTaste,
  movie: MovieProfile,
  family: AlgorithmFamily,
  rng: SeededRNG
): number {
  let score = 3;
  for (let d = 0; d < user.factors.length; d++) {
    if (family.visibility[d]) {
      score += user.factors[d] * movie.factors[d] * 0.5;
    }
  }
  score += rng.gaussian() * family.noiseStd;
  return Math.max(1, Math.min(5, score));
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

export interface NetflixDemoConfig {
  latentDims?: number;
  userCount?: number;
  movieCount?: number;
  ratingsPerUser?: number;
  seed?: number;
  /** Budget in ms -- controls how aggressively branches get vented for cost */
  budgetMs?: number;
}

export interface NetflixDemoFrame {
  input: TopologyFrameInput;
  result: TopologyFrameResult;
  /** The rating this frame was generated from */
  rating: { userId: number; movieId: number; actual: number };
  /** Per-family predictions for this rating */
  predictions: readonly { family: string; prediction: number; error: number }[];
}

/**
 * Generate the full sequence of topology frames from a Netflix-like
 * synthetic dataset. Each rating becomes a frame where algorithm families
 * race to predict it. Losers get vented.
 *
 * Returns a generator so you can scrub through frames lazily.
 */
export function* generateNetflixFrames(
  config?: NetflixDemoConfig
): Generator<NetflixDemoFrame> {
  const latentDims = config?.latentDims ?? 8;
  const userCount = config?.userCount ?? 500;
  const movieCount = config?.movieCount ?? 200;
  const ratingsPerUser = config?.ratingsPerUser ?? 40;
  const seed = config?.seed ?? 42;
  const budgetMs = config?.budgetMs ?? 8;

  const rng = new SeededRNG(seed);
  const families = createFamilies(latentDims);

  // Generate users
  const users: UserTaste[] = [];
  for (let i = 0; i < userCount; i++) {
    const factors: number[] = [];
    for (let d = 0; d < latentDims; d++) factors.push(rng.gaussian());
    users.push({ id: i, factors });
  }

  // Generate movies
  const movies: MovieProfile[] = [];
  for (let i = 0; i < movieCount; i++) {
    const factors: number[] = [];
    for (let d = 0; d < latentDims; d++) factors.push(rng.gaussian());
    movies.push({ id: i, factors });
  }

  // Generate ratings and convert each to a topology frame
  for (const user of users) {
    const count = Math.max(
      1,
      Math.round(ratingsPerUser + rng.gaussian() * ratingsPerUser * 0.3)
    );

    for (let r = 0; r < count; r++) {
      const movieIdx = Math.floor(rng.next() * movieCount);
      const movie = movies[movieIdx];

      let trueScore = 3;
      for (let d = 0; d < latentDims; d++) {
        trueScore += user.factors[d] * movie.factors[d] * 0.5;
      }
      trueScore += rng.gaussian() * 0.3;
      const actual = Math.max(1, Math.min(5, Math.round(trueScore * 2) / 2));

      // Each family becomes a branch, competing to predict this rating
      const predictions: { family: string; prediction: number; error: number }[] = [];
      const branches: TopologyBranch[] = [];

      for (let f = 0; f < families.length; f++) {
        const family = families[f];
        const pred = predict(user, movie, family, rng);
        const error = Math.abs(pred - actual);
        predictions.push({ family: family.shortLabel, prediction: pred, error });

        branches.push({
          id: family.shortLabel,
          vertices: Math.round(family.visibility.filter(Boolean).length * 100),
          drawCalls: 1,
          estimatedCostMs: error * budgetMs, // Higher error = higher cost
          quality: 1 - error / 4, // Lower error = higher quality
          vent: false,
        });
      }

      // Find the winner (lowest error)
      let winnerIdx = 0;
      for (let i = 1; i < predictions.length; i++) {
        if (predictions[i].error < predictions[winnerIdx].error) winnerIdx = i;
      }

      const input: TopologyFrameInput = {
        branches,
        strategy: 'race',
        budgetMs,
      };

      // Manually construct result: winner survives, rest vented
      const ventedIds = branches
        .filter((_, i) => i !== winnerIdx)
        .map((b) => b.id);

      const winner = branches[winnerIdx];
      const result: TopologyFrameResult = {
        collapsedBy: 'race',
        winnerId: winner.id,
        survivingBranches: [winner],
        ventedBranchIds: ventedIds,
        totalVertices: winner.vertices,
        totalDrawCalls: winner.drawCalls,
        totalCostMs: winner.estimatedCostMs,
        budgetMs,
        overBudget: winner.estimatedCostMs > budgetMs,
      };

      yield {
        input,
        result,
        rating: { userId: user.id, movieId: movie.id, actual },
        predictions,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Batch runner: process N frames into a VoidMap
// ---------------------------------------------------------------------------

export interface NetflixDemoResult {
  map: VoidMap;
  framesProcessed: number;
  /** Per-family win counts */
  winCounts: ReadonlyMap<string, number>;
  /** Per-family cumulative error */
  cumulativeErrors: ReadonlyMap<string, number>;
}

/**
 * Run the Netflix demo for N frames (or all frames if N is omitted).
 * Returns the accumulated VoidMap and statistics.
 */
export function runNetflixDemo(
  config?: NetflixDemoConfig & { maxFrames?: number },
  voidMapOptions?: VoidMapOptions
): NetflixDemoResult {
  const maxFrames = config?.maxFrames;
  let map = createVoidMap(voidMapOptions);
  let processed = 0;
  const winCounts = new Map<string, number>();
  const cumulativeErrors = new Map<string, number>();

  for (const frame of generateNetflixFrames(config)) {
    map = recordFrame(map, frame.input, frame.result, voidMapOptions);
    processed++;

    // Track wins
    if (frame.result.winnerId) {
      winCounts.set(
        frame.result.winnerId,
        (winCounts.get(frame.result.winnerId) ?? 0) + 1
      );
    }

    // Track cumulative errors
    for (const pred of frame.predictions) {
      cumulativeErrors.set(
        pred.family,
        (cumulativeErrors.get(pred.family) ?? 0) + pred.error
      );
    }

    if (maxFrames !== undefined && processed >= maxFrames) break;
  }

  return { map, framesProcessed: processed, winCounts, cumulativeErrors };
}

/**
 * Run two parallel universes with different parameters on the same data.
 * Returns both VoidMaps for diff/split-screen visualization.
 *
 * Example: run one with eta=0.01 (slow learner) and one with eta=3.0
 * (aggressive learner), then diff them to see the unmergeable gap.
 */
export function runNetflixForkExperiment(
  configA: NetflixDemoConfig & { maxFrames?: number },
  configB: NetflixDemoConfig & { maxFrames?: number },
  voidMapOptionsA?: VoidMapOptions,
  voidMapOptionsB?: VoidMapOptions
): { a: NetflixDemoResult; b: NetflixDemoResult } {
  return {
    a: runNetflixDemo(configA, voidMapOptionsA),
    b: runNetflixDemo(configB, voidMapOptionsB),
  };
}

import { describe, it, expect } from 'vitest';
import { generateNetflixFrames, runNetflixDemo, runNetflixForkExperiment } from './voidmap-netflix.js';

describe('generateNetflixFrames', () => {
  it('produces frames with correct algorithm family branch IDs', () => {
    const gen = generateNetflixFrames({ latentDims: 8, userCount: 5, movieCount: 5, ratingsPerUser: 2, seed: 42 });
    const first = gen.next();
    expect(first.done).toBe(false);
    const frame = first.value!;

    const branchIds = frame.input.branches.map((b) => b.id);
    expect(branchIds).toContain('MF');
    expect(branchIds).toContain('kNN');
    expect(branchIds).toContain('RBM');
    expect(branchIds).toContain('tSVD');
    expect(branchIds).toContain('NNMF');
  });

  it('is deterministic with same seed', () => {
    const config = { latentDims: 8, userCount: 10, movieCount: 5, ratingsPerUser: 2, seed: 123 };
    const gen1 = generateNetflixFrames(config);
    const gen2 = generateNetflixFrames(config);

    for (let i = 0; i < 10; i++) {
      const f1 = gen1.next().value!;
      const f2 = gen2.next().value!;
      expect(f1.result.winnerId).toBe(f2.result.winnerId);
      expect(f1.rating.actual).toBe(f2.rating.actual);
    }
  });

  it('produces winner and vented branches each frame', () => {
    const gen = generateNetflixFrames({ latentDims: 8, userCount: 5, movieCount: 5, ratingsPerUser: 3, seed: 42 });
    const frame = gen.next().value!;
    expect(frame.result.winnerId).toBeTruthy();
    expect(frame.result.ventedBranchIds.length).toBeGreaterThan(0);
    expect(frame.result.ventedBranchIds).not.toContain(frame.result.winnerId);
  });
});

describe('runNetflixDemo', () => {
  it('accumulates a VoidMap with tombstones', () => {
    const result = runNetflixDemo({ latentDims: 8, userCount: 20, movieCount: 10, ratingsPerUser: 5, seed: 42, maxFrames: 50 });
    expect(result.framesProcessed).toBe(50);
    expect(result.map.tombstones.length).toBeGreaterThan(0);
    expect(result.map.round).toBe(50);
  });

  it('tracks win counts for all families', () => {
    const result = runNetflixDemo({ latentDims: 8, userCount: 50, movieCount: 20, ratingsPerUser: 10, seed: 42, maxFrames: 200 });
    expect(result.winCounts.size).toBeGreaterThan(0);

    // At least some families should win
    let totalWins = 0;
    for (const count of result.winCounts.values()) {
      totalWins += count;
    }
    expect(totalWins).toBe(200);
  });

  it('tracks cumulative errors', () => {
    const result = runNetflixDemo({ latentDims: 8, userCount: 10, movieCount: 5, ratingsPerUser: 5, seed: 42, maxFrames: 20 });
    expect(result.cumulativeErrors.size).toBeGreaterThan(0);
    for (const error of result.cumulativeErrors.values()) {
      expect(error).toBeGreaterThan(0);
    }
  });
});

describe('runNetflixForkExperiment', () => {
  it('produces two VoidMaps from different seeds', () => {
    const result = runNetflixForkExperiment(
      { latentDims: 8, userCount: 20, movieCount: 10, ratingsPerUser: 5, seed: 42, maxFrames: 50 },
      { latentDims: 8, userCount: 20, movieCount: 10, ratingsPerUser: 5, seed: 99, maxFrames: 50 }
    );
    expect(result.a.framesProcessed).toBe(50);
    expect(result.b.framesProcessed).toBe(50);
    // Different seeds produce different winner patterns
    const winnersA = Array.from(result.a.winCounts.entries()).sort();
    const winnersB = Array.from(result.b.winCounts.entries()).sort();
    // At least one family should have different win counts
    const anyDifferent = winnersA.some(([family, count], i) => {
      const matchB = winnersB.find(([f]) => f === family);
      return !matchB || matchB[1] !== count;
    });
    expect(anyDifferent).toBe(true);
  });
});

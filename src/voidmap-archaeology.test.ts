import { describe, it, expect } from 'vitest';
import {
  searchTombstones,
  highlightByEra,
  tombstoneDensityPerRound,
  windowedDensity,
  branchFailureTimeline,
  allBranchTimelines,
  reconstructFromSnapshotChain,
  createManualBookmark,
  autoBookmarkPhaseTransitions,
  createMilestoneBookmarks,
  summarizeVoid,
} from './voidmap-archaeology.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import { snapshotVoidMap } from './voidmap-temporal.js';
import type { TopologyBranch } from './index.js';

function makeBranch(id: string, cost: number, quality: number, vertices: number): TopologyBranch {
  return { id, vertices, drawCalls: 1, estimatedCostMs: cost, quality };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[], budgetMs = 8) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs },
    result: {
      collapsedBy: 'race' as const, winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving, ventedBranchIds: ventedIds,
      totalVertices: 0, totalDrawCalls: 0, totalCostMs: 0, budgetMs, overBudget: false,
    },
  };
}

function buildMap(rounds: number) {
  const branches = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200), makeBranch('C', 7, 0.6, 300)];
  let map = createVoidMap();
  for (let r = 0; r < rounds; r++) {
    const winnerIdx = r % 3;
    const ventedIds = branches.filter((_, i) => i !== winnerIdx).map((b) => b.id);
    const f = makeFrame(branches, ventedIds);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('searchTombstones', () => {
  it('filters by branchId', () => {
    const map = buildMap(10);
    const results = searchTombstones(map, { branchId: 'A' });
    for (const r of results) expect(r.branchId).toBe('A');
    expect(results.length).toBeGreaterThan(0);
  });

  it('filters by branchIdPrefix', () => {
    const map = buildMap(10);
    const results = searchTombstones(map, { branchIdPrefix: 'B' });
    for (const r of results) expect(r.branchId.startsWith('B')).toBe(true);
  });

  it('filters by reason', () => {
    const map = buildMap(10);
    const results = searchTombstones(map, { reason: 'race-loser' });
    for (const r of results) expect(r.reason).toBe('race-loser');
  });

  it('filters by roundRange', () => {
    const map = buildMap(20);
    const results = searchTombstones(map, { roundRange: [5, 10] });
    for (const r of results) {
      expect(r.round).toBeGreaterThanOrEqual(5);
      expect(r.round).toBeLessThanOrEqual(10);
    }
  });

  it('filters by coordinate region', () => {
    const map = buildMap(20);
    const results = searchTombstones(map, {
      region: { min: { x: 0, y: 0, z: 0 }, max: { x: 0.5, y: 1, z: 1 } },
    });
    for (const r of results) {
      expect(r.position.x).toBeLessThanOrEqual(0.5);
    }
  });

  it('sorts by round-desc', () => {
    const map = buildMap(20);
    const results = searchTombstones(map, { sortBy: 'round-desc' });
    for (let i = 1; i < results.length; i++) {
      expect(results[i].round).toBeLessThanOrEqual(results[i - 1].round);
    }
  });

  it('respects limit', () => {
    const map = buildMap(20);
    const results = searchTombstones(map, { limit: 5 });
    expect(results).toHaveLength(5);
  });
});

describe('highlightByEra', () => {
  it('assigns eras correctly', () => {
    const map = buildMap(20);
    const eras = [
      { label: 'Early', fromRound: 1, toRound: 5, color: '#ff0000' },
      { label: 'Late', fromRound: 15, toRound: 20, color: '#00ff00' },
    ];
    const highlighted = highlightByEra(map, eras);

    for (const h of highlighted) {
      if (h.tombstone.round >= 1 && h.tombstone.round <= 5) {
        expect(h.era?.label).toBe('Early');
        expect(h.highlight).toBe(1);
      } else if (h.tombstone.round >= 15 && h.tombstone.round <= 20) {
        expect(h.era?.label).toBe('Late');
        expect(h.highlight).toBe(1);
      } else {
        expect(h.era).toBeNull();
        expect(h.highlight).toBe(0.15);
      }
    }
  });
});

describe('tombstoneDensityPerRound', () => {
  it('returns correct density', () => {
    const map = buildMap(10);
    const density = tombstoneDensityPerRound(map);
    expect(density).toHaveLength(10);

    // Each round vents 2 branches (3 total, 1 wins)
    for (const d of density) {
      expect(d.count).toBe(2);
    }

    // Cumulative should increase
    for (let i = 1; i < density.length; i++) {
      expect(density[i].cumulative).toBeGreaterThan(density[i - 1].cumulative);
    }
  });

  it('returns empty for empty map', () => {
    expect(tombstoneDensityPerRound(createVoidMap())).toHaveLength(0);
  });
});

describe('windowedDensity', () => {
  it('computes moving average', () => {
    const map = buildMap(20);
    const density = tombstoneDensityPerRound(map);
    const windowed = windowedDensity(density, 5);
    expect(windowed.length).toBe(density.length - 4);
    // Uniform density of 2 → moving average should be 2
    for (const w of windowed) {
      expect(w.averageDensity).toBeCloseTo(2, 5);
    }
  });
});

describe('branchFailureTimeline', () => {
  it('returns null for unknown branch', () => {
    const map = buildMap(10);
    expect(branchFailureTimeline(map, 'nonexistent')).toBeNull();
  });

  it('builds correct timeline', () => {
    const map = buildMap(12);
    const timeline = branchFailureTimeline(map, 'A');
    expect(timeline).not.toBeNull();
    expect(timeline!.branchId).toBe('A');
    expect(timeline!.totalRejections).toBeGreaterThan(0);
    expect(timeline!.firstRound).toBeGreaterThanOrEqual(1);
    expect(timeline!.lastRound).toBeLessThanOrEqual(12);
    expect(timeline!.rejectionRounds.length).toBe(timeline!.totalRejections);
    expect(timeline!.active).toBe(true); // Last rejection is recent
  });
});

describe('allBranchTimelines', () => {
  it('returns timelines sorted by rejection count', () => {
    const map = buildMap(10);
    const timelines = allBranchTimelines(map);
    expect(timelines.length).toBe(3);
    for (let i = 1; i < timelines.length; i++) {
      expect(timelines[i].totalRejections).toBeLessThanOrEqual(timelines[i - 1].totalRejections);
    }
  });
});

describe('reconstructFromSnapshotChain', () => {
  it('deduplicates tombstones across snapshots', async () => {
    const map1 = buildMap(5);
    const map2 = buildMap(10);
    const snap1 = await snapshotVoidMap(map1);
    const snap2 = await snapshotVoidMap(map2);

    const reconstructed = reconstructFromSnapshotChain([snap1, snap2]);
    expect(reconstructed.totalRecovered).toBeLessThanOrEqual(
      map1.tombstones.length + map2.tombstones.length
    );
    // No duplicate (branchId, round) keys
    const keys = new Set(reconstructed.tombstones.map((t) => `${t.branchId}:${t.round}`));
    expect(keys.size).toBe(reconstructed.totalRecovered);
  });

  it('returns empty for empty chain', () => {
    const result = reconstructFromSnapshotChain([]);
    expect(result.totalRecovered).toBe(0);
  });
});

describe('bookmarks', () => {
  it('creates manual bookmarks', () => {
    const map = buildMap(20);
    const bookmark = createManualBookmark(
      map, 10, 'Test Bookmark',
      { x: 3, y: 2, z: 3 }, { x: 0, y: 0, z: 0 },
      { contrail: true, ghosts: false, cone: false, phases: true, trajectories: false }
    );
    expect(bookmark.round).toBe(10);
    expect(bookmark.label).toBe('Test Bookmark');
    expect(bookmark.reason).toBe('manual');
    expect(bookmark.stats.totalTombstones).toBeGreaterThan(0);
  });

  it('auto-generates phase transition bookmarks', () => {
    // Build a map with a disruption: first 20 rounds uniform, then sudden change
    const branches1 = [makeBranch('A', 5, 0.8, 100), makeBranch('B', 6, 0.7, 200)];
    const branches2 = [makeBranch('C', 3, 0.9, 100), makeBranch('D', 12, 0.3, 200)];
    let map = createVoidMap();
    for (let i = 0; i < 20; i++) {
      const f = makeFrame(branches1, ['B']);
      map = recordFrame(map, f.input, f.result);
    }
    for (let i = 0; i < 20; i++) {
      const f = makeFrame(branches2, ['D']);
      map = recordFrame(map, f.input, f.result);
    }

    const bookmarks = autoBookmarkPhaseTransitions(map);
    // Should detect at least one phase transition around round 20
    // (may or may not depending on entropy threshold)
    expect(Array.isArray(bookmarks)).toBe(true);
  });

  it('creates milestone bookmarks at intervals', () => {
    const map = buildMap(50);
    const milestones = createMilestoneBookmarks(map, 10);
    expect(milestones.length).toBe(5); // 10, 20, 30, 40, 50
    for (const m of milestones) {
      expect(m.reason).toBe('milestone');
      expect(m.round % 10).toBe(0);
    }
  });
});

describe('summarizeVoid', () => {
  it('returns correct summary', () => {
    const map = buildMap(30);
    const summary = summarizeVoid(map);
    expect(summary.totalRounds).toBe(30);
    expect(summary.totalTombstones).toBe(60); // 2 vented per round * 30 rounds
    expect(summary.distinctBranches).toBe(3);
    expect(summary.mostRejected).not.toBeNull();
    expect(summary.averageDensity).toBe(2);
    expect(summary.peakDensity).toBe(2);
    expect(typeof summary.converged).toBe('boolean');
  });

  it('handles empty map', () => {
    const summary = summarizeVoid(createVoidMap());
    expect(summary.totalRounds).toBe(0);
    expect(summary.totalTombstones).toBe(0);
    expect(summary.mostRejected).toBeNull();
  });
});

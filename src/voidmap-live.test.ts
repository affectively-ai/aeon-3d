import { describe, it, expect } from 'vitest';
import {
  inferenceEventToPosition,
  glossolaliaEventToPosition,
  inferenceEventToFrame,
  glossolaliaEventsToFrame,
  createLiveVoidMap,
  ingestInferenceEvent,
  ingestGlossolaliaStep,
  parseDashRelayInferenceEvent,
  parseDashRelayGlossolaliaEvents,
  createHeartbeatDetector,
  recordHeartbeat,
  isHeartbeatOverdue,
  type InferenceEvent,
  type GlossolaliaEvent,
  type DashRelayMessage,
} from './voidmap-live.js';

function makeInferenceEvent(overrides?: Partial<InferenceEvent>): InferenceEvent {
  return {
    requestId: 'req-1',
    serviceName: 'inference-llama70b-layer-03',
    layerIndex: 3,
    totalLayers: 80,
    hiddenStateMagnitude: 15.5,
    tokenPosition: 10,
    sequenceLength: 50,
    reason: 'layer-timeout',
    elapsedMs: 15000,
    budgetMs: 13000,
    modelName: 'llama-70b',
    timestamp: '2026-03-20T00:00:00Z',
    ...overrides,
  };
}

function makeGlossolaliaEvents(): GlossolaliaEvent[] {
  return [
    { stepId: 's1', strategy: 'temperature', tokenPosition: 5, sequenceLength: 50, logProbability: -2, won: true, timestamp: '2026-03-20T00:00:00Z' },
    { stepId: 's1', strategy: 'nucleus', tokenPosition: 5, sequenceLength: 50, logProbability: -4, won: false, winnerId: 'temperature', timestamp: '2026-03-20T00:00:00Z' },
    { stepId: 's1', strategy: 'top-k', tokenPosition: 5, sequenceLength: 50, logProbability: -5, won: false, winnerId: 'temperature', timestamp: '2026-03-20T00:00:00Z' },
  ];
}

describe('inferenceEventToPosition', () => {
  it('maps layerIndex/totalLayers to X', () => {
    const pos = inferenceEventToPosition(makeInferenceEvent({ layerIndex: 40, totalLayers: 80 }));
    expect(pos.x).toBeCloseTo(0.5, 5);
  });

  it('maps hiddenStateMagnitude to Y (log-scaled, capped)', () => {
    const pos = inferenceEventToPosition(makeInferenceEvent({ hiddenStateMagnitude: 1 }));
    expect(pos.y).toBeGreaterThanOrEqual(0);
    expect(pos.y).toBeLessThanOrEqual(2);
  });

  it('maps tokenPosition/sequenceLength to Z', () => {
    const pos = inferenceEventToPosition(makeInferenceEvent({ tokenPosition: 25, sequenceLength: 50 }));
    expect(pos.z).toBeCloseTo(0.5, 5);
  });
});

describe('glossolaliaEventToPosition', () => {
  it('maps strategy to deterministic X', () => {
    const events = makeGlossolaliaEvents();
    const pos0 = glossolaliaEventToPosition(events[0]); // temperature
    const pos1 = glossolaliaEventToPosition(events[1]); // nucleus
    expect(pos0.x).toBeLessThan(pos1.x); // temperature < nucleus in order
  });

  it('maps tokenPosition/sequenceLength to Z', () => {
    const event = makeGlossolaliaEvents()[0];
    const pos = glossolaliaEventToPosition(event);
    expect(pos.z).toBeCloseTo(5 / 50, 5);
  });
});

describe('inferenceEventToFrame', () => {
  it('produces a frame with the vented branch', () => {
    const event = makeInferenceEvent();
    const frame = inferenceEventToFrame(event);
    expect(frame.result.ventedBranchIds).toHaveLength(1);
    expect(frame.result.ventedBranchIds[0]).toContain('llama-70b');
    expect(frame.result.ventedBranchIds[0]).toContain('layer-3');
  });
});

describe('glossolaliaEventsToFrame', () => {
  it('returns null for empty events', () => {
    expect(glossolaliaEventsToFrame([])).toBeNull();
  });

  it('vents losers and keeps winner', () => {
    const events = makeGlossolaliaEvents();
    const frame = glossolaliaEventsToFrame(events)!;
    expect(frame.result.winnerId).toBe('glossolalia:temperature');
    expect(frame.result.ventedBranchIds).toContain('glossolalia:nucleus');
    expect(frame.result.ventedBranchIds).toContain('glossolalia:top-k');
    expect(frame.result.ventedBranchIds).not.toContain('glossolalia:temperature');
  });
});

describe('LiveVoidMap accumulator', () => {
  it('ingests inference events', () => {
    let live = createLiveVoidMap();
    live = ingestInferenceEvent(live, makeInferenceEvent());
    expect(live.inferenceEventCount).toBe(1);
    expect(live.map.tombstones).toHaveLength(1);
    expect(live.perModelCounts.get('llama-70b')).toBe(1);
  });

  it('ingests Glossolalia steps', () => {
    let live = createLiveVoidMap();
    live = ingestGlossolaliaStep(live, makeGlossolaliaEvents());
    expect(live.glossolaliaEventCount).toBe(3);
    expect(live.map.tombstones.length).toBeGreaterThan(0);
    expect(live.strategyStats.get('temperature')?.wins).toBe(1);
    expect(live.strategyStats.get('nucleus')?.losses).toBe(1);
  });

  it('accumulates across multiple events', () => {
    let live = createLiveVoidMap();
    for (let i = 0; i < 5; i++) {
      live = ingestInferenceEvent(live, makeInferenceEvent({ requestId: `req-${i}`, layerIndex: i }));
    }
    expect(live.inferenceEventCount).toBe(5);
    expect(live.map.round).toBe(5);
  });
});

describe('DashRelay parsing', () => {
  it('parses valid inference event', () => {
    const msg: DashRelayMessage = {
      room: 'aether-logs',
      type: 'layer-failure',
      payload: {
        requestId: 'req-1',
        serviceName: 'test-service',
        layerIndex: 5,
        totalLayers: 32,
        hiddenStateMagnitude: 10,
        tokenPosition: 3,
        sequenceLength: 20,
        reason: 'layer-timeout',
        elapsedMs: 5000,
        budgetMs: 3000,
        modelName: 'test-model',
      },
      timestamp: '2026-03-20T00:00:00Z',
    };
    const event = parseDashRelayInferenceEvent(msg);
    expect(event).not.toBeNull();
    expect(event!.layerIndex).toBe(5);
    expect(event!.reason).toBe('layer-timeout');
  });

  it('returns null for non-aether-logs room', () => {
    const msg: DashRelayMessage = {
      room: 'other-room',
      type: 'layer-failure',
      payload: { layerIndex: 5, totalLayers: 32, reason: 'layer-timeout' },
      timestamp: '2026-03-20T00:00:00Z',
    };
    expect(parseDashRelayInferenceEvent(msg)).toBeNull();
  });

  it('returns null for invalid reason', () => {
    const msg: DashRelayMessage = {
      room: 'aether-logs',
      type: 'layer-failure',
      payload: { layerIndex: 5, totalLayers: 32, reason: 'invalid-reason' },
      timestamp: '2026-03-20T00:00:00Z',
    };
    expect(parseDashRelayInferenceEvent(msg)).toBeNull();
  });

  it('parses valid Glossolalia events', () => {
    const msg: DashRelayMessage = {
      room: 'aether-logs',
      type: 'glossolalia-race',
      payload: {
        candidates: [
          { strategy: 'temperature', tokenPosition: 5, sequenceLength: 20, logProbability: -2, won: true, stepId: 's1' },
          { strategy: 'nucleus', tokenPosition: 5, sequenceLength: 20, logProbability: -4, won: false, stepId: 's1' },
        ],
      },
      timestamp: '2026-03-20T00:00:00Z',
    };
    const events = parseDashRelayGlossolaliaEvents(msg);
    expect(events).not.toBeNull();
    expect(events!).toHaveLength(2);
  });

  it('returns null for non-glossolalia type', () => {
    const msg: DashRelayMessage = {
      room: 'aether-logs',
      type: 'other-type',
      payload: { candidates: [] },
      timestamp: '2026-03-20T00:00:00Z',
    };
    expect(parseDashRelayGlossolaliaEvents(msg)).toBeNull();
  });
});

describe('Heartbeat detector', () => {
  it('starts healthy', () => {
    const state = createHeartbeatDetector(1000);
    expect(state.healthy).toBe(true);
    expect(state.missedBeats).toBe(0);
  });

  it('stays healthy with regular beats', () => {
    let state = createHeartbeatDetector(1000);
    for (let i = 0; i < 5; i++) {
      state = recordHeartbeat(state, i * 1000);
    }
    expect(state.healthy).toBe(true);
    expect(state.averageIntervalMs).toBeCloseTo(1000, -1);
  });

  it('detects missed beats', () => {
    let state = createHeartbeatDetector(1000);
    state = recordHeartbeat(state, 0);
    state = recordHeartbeat(state, 1000);
    state = recordHeartbeat(state, 5000); // 4x the expected interval
    expect(state.healthy).toBe(false);
    expect(state.missedBeats).toBeGreaterThan(0);
  });

  it('detects overdue heartbeats', () => {
    let state = createHeartbeatDetector(1000);
    state = recordHeartbeat(state, 0);
    expect(isHeartbeatOverdue(state, 2000)).toBe(true);
    expect(isHeartbeatOverdue(state, 500)).toBe(false);
  });
});

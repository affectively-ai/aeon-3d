import { describe, it, expect } from 'vitest';
import {
  createVoidCertificate,
  createAuditTrail,
  appendCertificate,
  verifyCertificateChain,
  restoreFromCertificate,
  generateSafetyProof,
  checkCompliance,
  createBequeathmentReceipt,
  buildPeriodicAuditTrail,
} from './voidmap-provenance.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import type { TopologyBranch } from './index.js';

function makeBranch(id: string, cost: number): TopologyBranch {
  return { id, vertices: 100, drawCalls: 1, estimatedCostMs: cost, quality: 0.5 };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[]) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs: 8 },
    result: {
      collapsedBy: 'race' as const, winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving, ventedBranchIds: ventedIds,
      totalVertices: 0, totalDrawCalls: 0, totalCostMs: 0, budgetMs: 8, overBudget: false,
    },
  };
}

function buildMap(rounds: number) {
  const branches = [makeBranch('safe-option', 5), makeBranch('risky-option', 10), makeBranch('default', 3)];
  let map = createVoidMap();
  for (let i = 0; i < rounds; i++) {
    const f = makeFrame(branches, ['risky-option']);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('createVoidCertificate', () => {
  it('produces a certificate with correct fields', async () => {
    const map = buildMap(10);
    const cert = await createVoidCertificate(map);
    expect(cert.cid).toBeTruthy();
    expect(cert.round).toBe(10);
    expect(cert.tombstoneCount).toBe(10);
    expect(cert.entropy).toBeGreaterThanOrEqual(0);
    expect(cert.parentCid).toBeNull();
    expect(cert.timestamp).toBeTruthy();
  });

  it('produces deterministic CID for same input', async () => {
    const map = buildMap(5);
    const cert1 = await createVoidCertificate(map);
    const cert2 = await createVoidCertificate(map);
    expect(cert1.cid).toBe(cert2.cid);
  });

  it('chains with parent CID', async () => {
    const map = buildMap(5);
    const parent = await createVoidCertificate(map);
    const child = await createVoidCertificate(map, parent.cid);
    expect(child.parentCid).toBe(parent.cid);
    expect(child.cid).not.toBe(parent.cid);
  });
});

describe('Audit trail', () => {
  it('builds a chain of certificates', async () => {
    const map5 = buildMap(5);
    const map10 = buildMap(10);

    let trail = createAuditTrail();
    const cert1 = await createVoidCertificate(map5, null);
    trail = appendCertificate(trail, cert1);
    const cert2 = await createVoidCertificate(map10, cert1.cid);
    trail = appendCertificate(trail, cert2);

    expect(trail.certificates).toHaveLength(2);
    expect(trail.headCid).toBe(cert2.cid);
    expect(trail.fromRound).toBe(5);
    expect(trail.toRound).toBe(10);
  });

  it('rejects certificate with wrong parent', async () => {
    const map = buildMap(5);
    let trail = createAuditTrail();
    const cert1 = await createVoidCertificate(map, null);
    trail = appendCertificate(trail, cert1);

    const badCert = await createVoidCertificate(map, 'wrong-parent-cid');
    expect(() => appendCertificate(trail, badCert)).toThrow('Certificate chain broken');
  });
});

describe('verifyCertificateChain', () => {
  it('validates a correct chain', async () => {
    const map5 = buildMap(5);
    const map10 = buildMap(10);

    let trail = createAuditTrail();
    const cert1 = await createVoidCertificate(map5, null);
    trail = appendCertificate(trail, cert1);
    const cert2 = await createVoidCertificate(map10, cert1.cid);
    trail = appendCertificate(trail, cert2);

    const result = verifyCertificateChain(trail);
    expect(result.valid).toBe(true);
    expect(result.brokenAt).toBeNull();
  });

  it('detects broken chain', () => {
    const trail = {
      certificates: [
        { cid: 'a', round: 1, tombstoneCount: 0, complementDistHash: '', entropy: 0, inverseBule: 0, timestamp: '', parentCid: null, branchCount: 0, serialized: '{}' },
        { cid: 'b', round: 2, tombstoneCount: 0, complementDistHash: '', entropy: 0, inverseBule: 0, timestamp: '', parentCid: 'wrong', branchCount: 0, serialized: '{}' },
      ],
      headCid: 'b',
      fromRound: 1,
      toRound: 2,
    };
    const result = verifyCertificateChain(trail);
    expect(result.valid).toBe(false);
    expect(result.brokenAt).toBe(1);
  });

  it('validates empty trail', () => {
    expect(verifyCertificateChain(createAuditTrail()).valid).toBe(true);
  });
});

describe('restoreFromCertificate', () => {
  it('round-trips correctly', async () => {
    const map = buildMap(10);
    const cert = await createVoidCertificate(map);
    const restored = restoreFromCertificate(cert);

    expect(restored.round).toBe(map.round);
    expect(restored.tombstones.length).toBe(map.tombstones.length);
    expect(restored.rejectionCounts.size).toBe(map.rejectionCounts.size);
  });
});

describe('generateSafetyProof', () => {
  it('demonstrates structured avoidance for high inverseBule', async () => {
    // Build a peaked map (one branch always loses)
    const map = buildMap(50);
    const cert = await createVoidCertificate(map);
    const proof = generateSafetyProof(cert, 0.1);

    expect(proof.consideredBranches.length).toBeGreaterThan(0);
    expect(proof.rejectedBranches).toContain('risky-option');
    expect(proof.complementDistribution.length).toBeGreaterThan(0);
    // With one branch always rejected, inverseBule should be high
    expect(proof.demonstratesStructuredAvoidance).toBe(true);
  });

  it('does not demonstrate avoidance for low inverseBule with high threshold', async () => {
    // Equal rejection → low inverseBule
    const branches = [makeBranch('A', 5), makeBranch('B', 6)];
    let map = createVoidMap();
    for (let i = 0; i < 10; i++) {
      const ventedId = i % 2 === 0 ? 'A' : 'B';
      const f = makeFrame(branches, [ventedId]);
      map = recordFrame(map, f.input, f.result);
    }
    const cert = await createVoidCertificate(map);
    const proof = generateSafetyProof(cert, 0.9);
    expect(proof.demonstratesStructuredAvoidance).toBe(false);
  });
});

describe('checkCompliance', () => {
  it('passes when all required alternatives were evaluated', async () => {
    const map = buildMap(10);
    const cert = await createVoidCertificate(map);
    const compliance = checkCompliance(cert, ['risky-option']);
    expect(compliance.compliant).toBe(true);
    expect(compliance.missingAlternatives).toHaveLength(0);
  });

  it('fails when required alternative is missing', async () => {
    const map = buildMap(10);
    const cert = await createVoidCertificate(map);
    const compliance = checkCompliance(cert, ['risky-option', 'never-considered']);
    expect(compliance.compliant).toBe(false);
    expect(compliance.missingAlternatives).toContain('never-considered');
  });
});

describe('createBequeathmentReceipt', () => {
  it('produces a receipt with correct fields', async () => {
    const map = buildMap(20);
    const receipt = await createBequeathmentReceipt(map, 'did:source', 'did:dest', 20);
    expect(receipt.sourceOwnerDid).toBe('did:source');
    expect(receipt.destinationOwnerDid).toBe('did:dest');
    expect(receipt.tombstonesTransferred).toBe(20);
    expect(receipt.statsAtTransfer.totalTombstones).toBe(20);
    expect(receipt.sourceCertificateCid).toBeTruthy();
    expect(receipt.destinationCertificateCid).toBeTruthy();
  });
});

describe('buildPeriodicAuditTrail', () => {
  it('builds a valid chain from periodic snapshots', async () => {
    const maps = [
      { round: 5, map: buildMap(5) },
      { round: 10, map: buildMap(10) },
      { round: 15, map: buildMap(15) },
    ];
    const trail = await buildPeriodicAuditTrail(maps);

    expect(trail.certificates).toHaveLength(3);
    const verification = verifyCertificateChain(trail);
    expect(verification.valid).toBe(true);
    expect(trail.fromRound).toBe(5);
    expect(trail.toRound).toBe(15);
  });
});

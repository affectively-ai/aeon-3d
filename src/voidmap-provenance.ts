/**
 * voidmap-provenance.ts
 *
 * Cryptographic void records: content-addressed certificates that prove
 * what was considered and rejected at a specific point in time.
 * The void is the audit trail of everything that was NOT chosen.
 */

import type { VoidMap, VoidBoundaryStats } from './voidmap.js';
import { complementDistribution, voidEntropy, computeVoidBoundary } from './voidmap.js';

// ---------------------------------------------------------------------------
// Certificate types
// ---------------------------------------------------------------------------

export interface VoidCertificate {
  /** SHA-256 content identifier */
  cid: string;
  /** Round at which this certificate was created */
  round: number;
  /** Number of tombstones at certification time */
  tombstoneCount: number;
  /** Hash of the complement distribution (for verification) */
  complementDistHash: string;
  /** Entropy at certification time */
  entropy: number;
  /** Inverse Bule at certification time */
  inverseBule: number;
  /** ISO timestamp */
  timestamp: string;
  /** Parent certificate CID (for chaining) */
  parentCid: string | null;
  /** Distinct branches at certification time */
  branchCount: number;
  /** Serialized VoidMap (for restoration) */
  serialized: string;
}

export interface VoidAuditTrail {
  /** Ordered chain of certificates, oldest first */
  certificates: readonly VoidCertificate[];
  /** The current (latest) certificate CID */
  headCid: string | null;
  /** Total rounds covered */
  fromRound: number;
  toRound: number;
}

export interface SafetyProof {
  /** Certificate this proof is based on */
  certificateCid: string;
  /** Round at which safety was assessed */
  round: number;
  /** Branches that were considered (all branches in the void) */
  consideredBranches: readonly string[];
  /** Branches that were rejected (have tombstones) */
  rejectedBranches: readonly string[];
  /** Per-branch rejection counts (proof of evaluation) */
  rejectionCounts: ReadonlyMap<string, number>;
  /** Complement distribution (proof of learned avoidance) */
  complementDistribution: readonly number[];
  /** Entropy (proof of structured rejection) */
  entropy: number;
  /** Inverse Bule (proof of avoidance confidence) */
  inverseBule: number;
  /** Whether the agent has demonstrated structured avoidance (inverseBule > threshold) */
  demonstratesStructuredAvoidance: boolean;
  /** Timestamp */
  timestamp: string;
}

export interface ComplianceCertificate {
  /** Certificate this compliance proof is based on */
  certificateCid: string;
  /** Required alternatives that must have been evaluated */
  requiredAlternatives: readonly string[];
  /** Which required alternatives have tombstones (were evaluated and rejected) */
  evaluatedAlternatives: readonly string[];
  /** Which required alternatives are missing from the void (never evaluated) */
  missingAlternatives: readonly string[];
  /** Is compliance met? (all required alternatives were evaluated) */
  compliant: boolean;
  /** Timestamp */
  timestamp: string;
}

export interface BequeathmentReceipt {
  /** Source ghost's void certificate CID */
  sourceCertificateCid: string;
  /** Destination ghost's void certificate CID (after transfer) */
  destinationCertificateCid: string;
  /** Source owner DID */
  sourceOwnerDid: string;
  /** Destination owner DID */
  destinationOwnerDid: string;
  /** Round at which bequeathment occurred */
  bequeathmentRound: number;
  /** Number of tombstones transferred */
  tombstonesTransferred: number;
  /** Void boundary stats at transfer time */
  statsAtTransfer: {
    entropy: number;
    inverseBule: number;
    totalTombstones: number;
    homologyRank: number;
  };
  /** Timestamp */
  timestamp: string;
}

// ---------------------------------------------------------------------------
// SHA-256 hashing
// ---------------------------------------------------------------------------

async function sha256(data: string): Promise<string> {
  if (typeof globalThis.crypto?.subtle?.digest === 'function') {
    const encoded = new TextEncoder().encode(data);
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoded);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  // Deterministic fallback: FNV-1a
  let hash = 0x811c9dc5;
  for (let i = 0; i < data.length; i++) {
    hash ^= data.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

// ---------------------------------------------------------------------------
// Certificate creation
// ---------------------------------------------------------------------------

/**
 * Create a void certificate: a content-addressed, cryptographically
 * verifiable snapshot of the void boundary at a point in time.
 */
export async function createVoidCertificate(
  map: VoidMap,
  parentCid: string | null = null,
  options?: { eta?: number }
): Promise<VoidCertificate> {
  const eta = options?.eta ?? 3.0;
  const stats = computeVoidBoundary(map, { eta });
  const compDist = complementDistribution(map, eta);

  // Serialize the VoidMap
  const serializable = {
    tombstones: map.tombstones,
    round: map.round,
    rejectionCounts: Array.from(map.rejectionCounts.entries()),
  };
  const serialized = JSON.stringify(serializable);

  // Hash the complement distribution for verification
  const complementDistHash = await sha256(JSON.stringify(compDist));

  // Hash the full certificate content
  const certContent = JSON.stringify({
    round: map.round,
    tombstoneCount: map.tombstones.length,
    complementDistHash,
    entropy: stats.entropy,
    inverseBule: stats.inverseBule,
    parentCid,
    serialized,
  });
  const cid = await sha256(certContent);

  return {
    cid,
    round: map.round,
    tombstoneCount: map.tombstones.length,
    complementDistHash,
    entropy: stats.entropy,
    inverseBule: stats.inverseBule,
    timestamp: new Date().toISOString(),
    parentCid,
    branchCount: map.rejectionCounts.size,
    serialized,
  };
}

// ---------------------------------------------------------------------------
// Audit trail
// ---------------------------------------------------------------------------

/**
 * Create an empty audit trail.
 */
export function createAuditTrail(): VoidAuditTrail {
  return {
    certificates: [],
    headCid: null,
    fromRound: 0,
    toRound: 0,
  };
}

/**
 * Append a certificate to the audit trail.
 * The certificate's parentCid must match the trail's headCid.
 */
export function appendCertificate(
  trail: VoidAuditTrail,
  certificate: VoidCertificate
): VoidAuditTrail {
  if (trail.headCid !== null && certificate.parentCid !== trail.headCid) {
    throw new Error(
      `Certificate chain broken: expected parentCid=${trail.headCid}, ` +
      `got parentCid=${certificate.parentCid}`
    );
  }

  return {
    certificates: [...trail.certificates, certificate],
    headCid: certificate.cid,
    fromRound: trail.certificates.length === 0 ? certificate.round : trail.fromRound,
    toRound: certificate.round,
  };
}

/**
 * Verify the integrity of an audit trail.
 * Checks that each certificate's parentCid matches the previous certificate's CID.
 */
export function verifyCertificateChain(
  trail: VoidAuditTrail
): {
  valid: boolean;
  brokenAt: number | null;
  reason: string | null;
} {
  if (trail.certificates.length === 0) {
    return { valid: true, brokenAt: null, reason: null };
  }

  // First certificate must have null parent
  if (trail.certificates[0].parentCid !== null) {
    return {
      valid: false,
      brokenAt: 0,
      reason: `First certificate has non-null parentCid: ${trail.certificates[0].parentCid}`,
    };
  }

  for (let i = 1; i < trail.certificates.length; i++) {
    const expected = trail.certificates[i - 1].cid;
    const actual = trail.certificates[i].parentCid;
    if (actual !== expected) {
      return {
        valid: false,
        brokenAt: i,
        reason: `Certificate ${i} parentCid=${actual}, expected=${expected}`,
      };
    }

    // Rounds must be non-decreasing
    if (trail.certificates[i].round < trail.certificates[i - 1].round) {
      return {
        valid: false,
        brokenAt: i,
        reason: `Certificate ${i} round=${trail.certificates[i].round} < previous round=${trail.certificates[i - 1].round}`,
      };
    }
  }

  return { valid: true, brokenAt: null, reason: null };
}

/**
 * Restore a VoidMap from a certificate.
 */
export function restoreFromCertificate(certificate: VoidCertificate): VoidMap {
  const parsed = JSON.parse(certificate.serialized);
  return {
    tombstones: parsed.tombstones,
    round: parsed.round,
    rejectionCounts: new Map(parsed.rejectionCounts),
  };
}

// ---------------------------------------------------------------------------
// Safety proofs
// ---------------------------------------------------------------------------

/**
 * Generate a safety proof from a void certificate.
 * The proof demonstrates that the agent considered and rejected specific
 * alternatives, providing cryptographic evidence of deliberate avoidance.
 *
 * The Inverse Bule measures how structured the avoidance is:
 *   - High Inverse Bule = strong, consistent rejection patterns = learned safety
 *   - Low Inverse Bule = uniform rejection = no structure = hasn't learned
 */
export function generateSafetyProof(
  certificate: VoidCertificate,
  avoidanceThreshold: number = 0.3
): SafetyProof {
  const map = restoreFromCertificate(certificate);
  const compDist = complementDistribution(map);
  const consideredBranches = Array.from(map.rejectionCounts.keys());
  const rejectedBranches = consideredBranches.filter(
    (b) => (map.rejectionCounts.get(b) ?? 0) > 0
  );

  return {
    certificateCid: certificate.cid,
    round: certificate.round,
    consideredBranches,
    rejectedBranches,
    rejectionCounts: map.rejectionCounts,
    complementDistribution: compDist,
    entropy: certificate.entropy,
    inverseBule: certificate.inverseBule,
    demonstratesStructuredAvoidance: certificate.inverseBule > avoidanceThreshold,
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Compliance certificates
// ---------------------------------------------------------------------------

/**
 * Check regulatory compliance: verify that all required alternatives
 * were evaluated (have tombstones in the void).
 *
 * A compliant system has tombstones for every required alternative,
 * proving it considered and rejected each one before proceeding.
 */
export function checkCompliance(
  certificate: VoidCertificate,
  requiredAlternatives: readonly string[]
): ComplianceCertificate {
  const map = restoreFromCertificate(certificate);
  const rejectedBranches = new Set(map.rejectionCounts.keys());

  const evaluatedAlternatives = requiredAlternatives.filter((alt) =>
    rejectedBranches.has(alt)
  );
  const missingAlternatives = requiredAlternatives.filter(
    (alt) => !rejectedBranches.has(alt)
  );

  return {
    certificateCid: certificate.cid,
    requiredAlternatives,
    evaluatedAlternatives,
    missingAlternatives,
    compliant: missingAlternatives.length === 0,
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Bequeathment receipts
// ---------------------------------------------------------------------------

/**
 * Create a bequeathment receipt: cryptographic proof that a void boundary
 * was transferred from one owner to another.
 */
export async function createBequeathmentReceipt(
  sourceMap: VoidMap,
  sourceOwnerDid: string,
  destinationOwnerDid: string,
  bequeathmentRound: number,
  options?: { eta?: number }
): Promise<BequeathmentReceipt> {
  const eta = options?.eta ?? 3.0;

  const sourceCert = await createVoidCertificate(sourceMap, null, { eta });
  const stats = computeVoidBoundary(sourceMap, { eta });

  // The destination certificate starts with the same void but a new owner
  const destCert = await createVoidCertificate(sourceMap, sourceCert.cid, { eta });

  return {
    sourceCertificateCid: sourceCert.cid,
    destinationCertificateCid: destCert.cid,
    sourceOwnerDid,
    destinationOwnerDid,
    bequeathmentRound,
    tombstonesTransferred: sourceMap.tombstones.length,
    statsAtTransfer: {
      entropy: stats.entropy,
      inverseBule: stats.inverseBule,
      totalTombstones: stats.totalTombstones,
      homologyRank: stats.homologyRank,
    },
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Periodic certification
// ---------------------------------------------------------------------------

/**
 * Create certificates at regular intervals, building an audit trail
 * that covers the full history of the void.
 */
export async function buildPeriodicAuditTrail(
  maps: readonly { round: number; map: VoidMap }[],
  options?: { eta?: number }
): Promise<VoidAuditTrail> {
  let trail = createAuditTrail();

  const sorted = [...maps].sort((a, b) => a.round - b.round);

  for (const { map } of sorted) {
    const parentCid = trail.headCid;
    const cert = await createVoidCertificate(map, parentCid, options);
    trail = appendCertificate(trail, cert);
  }

  return trail;
}

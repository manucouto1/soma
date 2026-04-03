# RFC-001: P2P Transport Layer with libp2p

**Status:** Draft  
**Date:** 2026-04-03  
**Author:** Manuel Couto Pintos  

## Summary

Replace WebSocket-based worker communication with a libp2p transport layer
using QUIC, Hole Punch (DCUtR), and Kademlia DHT. This enables:

- Workers behind NAT/firewalls to connect without port forwarding
- Direct P2P data transfer between workers (no relay bottleneck)
- Distributed caching via DHT (find who has a tensor without central index)

## Motivation

Current limitations:
1. Workers need exposed ports (WebSocket requires inbound connections)
2. Data between workers flows through the client (A → Client → B)
3. No way to discover which worker has a cached result
4. Coordinator is a single point of failure for routing

## Design

### Transport: QUIC

Replace WebSocket with QUIC (via `rust-libp2p`):
- Multiplexed streams over a single connection
- Built-in encryption (TLS 1.3)
- Better NAT traversal than TCP
- Zero-RTT reconnection

### NAT Traversal: Hole Punch (DCUtR)

Workers behind NAT connect outbound to a relay node (coordinator).
When two workers need to talk directly:

1. Coordinator tells both workers each other's observed address
2. Both attempt simultaneous connection (UDP hole punch)
3. If successful → direct QUIC connection, relay drops out
4. If failed → relay forwards data (fallback)

### Discovery: Kademlia DHT

Workers publish their capabilities and cached data keys to a DHT:

```
Worker A starts → joins DHT → publishes:
  - PeerId: QmWorkerA
  - Capabilities: {gpu: true, ram: 32GB, tags: ["training"]}
  - Cached keys: [CacheKey("abc123"), CacheKey("def456")]

Client needs a GPU worker → DHT lookup → finds Worker A
Client needs tensor "abc123" → DHT lookup → Worker A has it → QUIC fetch
```

No central registry needed. Coordinator is just a bootstrap node.

### Data Transfer: P2P DataRef

New `DataRef` variant for peer-to-peer data:

```rust
pub enum DataRef {
    Local { path: PathBuf },
    S3 { bucket: String, key: String, region: Option<String> },
    Cached { cache_key: CacheKey },
    Zarr { bucket: String, array_path: String, region: Option<String> },
    Stream { endpoint: String, format: StreamFormat },
    Inline { value: Value },
    // NEW
    Peer { peer_id: String, key: CacheKey },
}
```

Resolution logic:

```
resolve(DataRef::Peer { peer_id, key }):
  1. Check local cache → hit? return
  2. QUIC request to peer_id for key → receive tensor → cache locally → return
  3. Timeout? → fallback to S3 (if also stored there)
```

### Distributed Caching

When a worker computes a node output:

1. Store in local cache (as now)
2. Publish CacheKey to Kademlia DHT: "I have this key"
3. Optionally persist to S3 (for durability)

When a worker needs an input:

1. Check local cache
2. DHT lookup: "who has this CacheKey?" → get PeerId
3. QUIC fetch from peer → cache locally
4. Fallback: S3 fetch

### Pipeline Parallelism Data Flow

```
Worker A (backbone) → output tensor → ???  → Worker B (head)
```

**Current (via client):**
```
Worker A → WS → Client → WS → Worker B     ~200ms round-trip
```

**With P2P (this RFC):**
```
Worker A → QUIC → Worker B                  ~5ms direct
```

**With P2P + large tensors:**
```
Worker A → writes to S3 + publishes DataRef::Peer
Worker B → DHT lookup → QUIC fetch from A (or S3 fallback)
```

### Gossip (Future)

GossipSub could be used for:
- Broadcasting gradient updates in DataParallel (all workers need all gradients)
- Event streaming (all subscribers get execution events)
- Model parameter sync in Federated Learning

Not in scope for first iteration. AllReduce via direct QUIC is sufficient.

## Implementation Plan

### Phase 1: libp2p Integration
- Add `rust-libp2p` with QUIC + Kademlia + DCUtR
- New `SomaNode` struct wrapping libp2p Swarm
- Worker starts as a libp2p node instead of WS server
- Coordinator becomes a bootstrap node + relay

### Phase 2: P2P DataRef
- Add `DataRef::Peer { peer_id, key }`
- Implement QUIC-based tensor transfer protocol
- Request/response: "give me CacheKey X" → tensor bytes

### Phase 3: Distributed Cache
- Workers publish cache keys to DHT on compute
- `resolve(DataRef)` checks DHT before S3
- Cache eviction broadcasts "I no longer have key X"

### Phase 4: Hole Punch
- DCUtR relay integration
- Automatic upgrade: relay → direct when possible
- Fallback to relay when hole punch fails

## Dependencies

```toml
[dependencies]
libp2p = { version = "0.54", features = [
    "quic",
    "kad",
    "dcutr",
    "relay",
    "identify",
    "noise",
    "yamux",
    "tokio",
] }
```

## Migration Path

- Phase 1-2: workers support both WS (legacy) and libp2p
- Phase 3+: deprecate WS, libp2p only
- `add_worker("ws://...")` → WS mode (backward compat)
- `add_worker("/p2p/QmPeerId")` → libp2p mode

## Open Questions

1. **Authentication in P2P:** Use libp2p noise protocol (public key auth) or keep bearer tokens?
2. **Large tensor streaming:** Single QUIC stream or chunked? Compression?
3. **DHT key space:** Use CacheKey directly as DHT key, or hash to Kademlia key space?
4. **Gossip for gradients:** Worth the complexity vs direct AllReduce?

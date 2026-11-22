import os
import re
import threading
from typing import Optional, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

app = FastAPI(title="TDS Week 5 RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K_DEFAULT = 3

embedding_model = None
passage_embeddings = None
model_lock = threading.Lock()


# ============================================================
# 2. KNOWLEDGE BASE: ALL 50 PASSAGES
# Passage IDs must remain 0 to 49.
# ============================================================

PASSAGES = [
    """Raft consensus requires a quorum of 2 nodes to commit a log entry. The leader sends 150ms heartbeats to prevent follower election timeouts. Network partitions cause the minority partition to stop accepting writes. Log compaction via snapshotting reduces replay time after a crash. etcd and CockroachDB both implement Raft for replicated state machines.""",

    """Multi-Paxos consensus coordinates distributed replicated state without a single permanent master. Leader leases establish proposal rights for lease terms of 250ms before re-election. In the accept phase, a commit quorum of 4 acceptors must acknowledge the ballot number. Proposers resolve ballot conflicts through randomized backoff intervals. Google Chubby and Apache ZooKeeper incorporate Paxos-family consensus protocols.""",

    """Epidemic gossip protocols spread cluster membership updates in logarithmic rounds. The phi accrual failure detector triggers node suspicion after a 250ms suspicion timeout. If network ping latency pushes the accrual metric past 0.9, the cluster marks the node dead. Anti-entropy background exchanges repair missing data versions across healthy nodes. Apache Cassandra relies on gossip for ring state coordination.""",

    """The SWIM membership protocol provides weakly-consistent cluster membership with bounded false-positive rates. Nodes transmit ping probes periodically with a ping interval of 200ms. When an indirect ping probe fails, suspect timeout counters initiate dead-node eviction after crossing 0.7. Dissemination piggybacks membership updates onto outgoing probe packets to minimize packet overhead. HashiCorp Consul implements SWIM for decentralized cluster health monitoring.""",

    """Two-phase commit coordinates atomic distributed transactions across independent resource managers. The coordinator aborts the transaction if any participant fails to reply within a 500ms prepare timeout. In three-phase commit, non-blocking progress is guaranteed by requiring a pre-commit quorum of 2 participants. Presumed-abort optimizations eliminate logging coordinator decisions for rolled-back transactions. Distributed deadlocks during phase one are detected using cycle detection on wait-for graphs.""",

    """The Saga pattern orchestrates distributed long-running transactions across independent microservices using compensating actions. Forward execution steps timeout and trigger backward compensation if service responses exceed 500ms. Orchestration coordinators retry idempotent compensating steps across a minimum of 3 service retries before alerting. Unlike atomic two-phase commit, intermediate states remain visible to concurrent transactions across services. Temporal and Uber Cadence utilize sagas for resilient distributed workflows.""",

    """Consistent hashing maps nodes and keys onto a virtual ring to balance partition loads. When key redistribution skew exceeds a 0.5 threshold, virtual nodes are dynamically allocated to smooth variance. Vector clocks track causal event ordering across concurrent distributed replicas without synchronized clocks. Dynamic read repair discovers stale replicas during quorum read operations within 200ms intervals. Amazon Dynamo popularized this decentralized replication architecture.""",

    """Rendezvous hashing assigns object keys to highest-ranking server nodes through pseudo-random score evaluation. Hash weight multipliers scale server selection when load capacity distribution crosses a 0.7 threshold. When servers fail, displaced keys fail over to the next highest-scoring node within 250ms. Unlike consistent hash rings, rendezvous hashing eliminates virtual node maintenance overhead completely. Proxy caches like Squid and Envoy use highest random weight hashing for backend routing.""",

    """Vector clocks capture partial ordering and causal relationships among concurrent events in distributed stores. When read repairs detect version concurrency, divergence resolution triggers within 150ms of replica discovery. Version vectors prune stale ancestor entries when actor membership counts exceed 3 replicas. Conflict resolution delegates sibling merges to application-specific resolution routines when clocks are concurrent. Riak KV relies on vector clocks for causality tracking across multi-master replicas.""",

    """Hybrid logical clocks provide monotonic causality tracking by combining physical hardware timers with logical sequence counters. Clock synchronization algorithms bound maximum physical time drift within 100ms across all cluster nodes. If physical time advances past logical timestamps, the logical offset resets when skew drops below 0.7. Hybrid logical clocks produce single 64-bit timestamps without unbounded vector size inflation. CockroachDB and MongoDB use hybrid logical clocks for linearizable distributed read timestamps.""",

    """Stochastic gradient descent updates model weights iteratively using sampled mini-batches. Training mini-batches are grouped into sizes of 256 samples per forward step. The optimizer applies a base learning rate of 0.01 to scale weight gradient updates. Momentum accumulates historical gradient vectors to accelerate traversal through shallow saddle points. Weight decay provides L2 regularization to prevent over-parameterized models from overfitting.""",

    """AdamW stabilizes adaptive parameter updates by decoupling weight decay regularization from gradient moment updates. First and second exponential moment vectors track running gradients across mini-batches of 64 samples. The initial learning rate schedule initializes step updates at 0.05 prior to cosine annealing. Decoupled weight decay applies L2 penalties directly to parameters without distorting adaptive learning rates. HuggingFace transformers employ AdamW as the standard optimizer for foundation model pre-training.""",

    """Transformer encoders use multi-head self-attention mechanisms to capture long-range contextual token representations. The deep backbone stack consists of 4 sequential transformer blocks with residual connections. Warmup scheduling ramps up the step learning rate toward 0.01 before cosine decay begins. Layer normalization stabilizes hidden state activations across varying input sequence lengths. Positional embeddings inject positional sequence order into permutation-invariant attention computations.""",

    """Autoregressive transformer decoders generate token sequences iteratively using causal attention masks. The causal generative model stacks 6 transformer blocks configured with rotary position embeddings. Training sequences partition token streams into window lengths of 256 tokens per gradient step. Causal masking zeros future token attention weights to enforce unidirectional auto-regressive generation. Modern generative models like LLaMA and GPT rely on decoder-only autoregressive architectures.""",

    """Binary logistic regression maps linear logit combinations into probabilistic predictions via sigmoid activation. Classification decisions categorize positive predictions whenever estimated class probability exceeds 0.7. Evaluation validation batches process 32 instances simultaneously during offline scoring passes. Expected calibration error quantifies probability deviation between predicted confidences and true empirical accuracy. Temperature scaling adjusts logit temperatures to produce well-calibrated posterior probabilities without degrading accuracy.""",

    """Support vector machines construct maximum-margin hyperplanes to separate high-dimensional feature distributions. The soft-margin optimization penalty penalizes margin violations exceeding a slack threshold of 0.85. Training optimization solves dual quadratic programs across sub-sampled chunks of 128 support vectors. Radial basis function kernels project non-linear decision boundaries into infinite-dimensional Hilbert spaces. Platt scaling fits logistic sigmoids over raw SVM output distances to derive posterior confidence probabilities.""",

    """Multi-layer perceptrons learn hierarchical feature abstractions through stacked non-linear transformations. The feedforward classifier contains 3 hidden dense layers with GELU activations. Dropout randomly zeros unit outputs with probability exceeding 0.5 to prevent co-adaptation. Batch normalization normalizes internal activation distributions across training minibatches. Gradient clipping bounds the L2 norm of parameter gradients to avoid catastrophic exploding gradients.""",

    """Residual convolutional networks mitigate vanishing gradients across deep computer vision architectures using identity skip connections. The feature extraction backbone organizes 6 residual bottleneck blocks grouped into spatial resolution stages. Spatial feature dropout suppresses activation feature maps when unit correlation exceeds 0.7. Convolutions apply spatial kernels across input feature dimensions to learn translation-invariant spatial representations. ResNet and ConvNeXt architectures demonstrate that skip connections enable stable convergence in deep networks.""",

    """Batch normalization accelerates neural network training by normalizing layer input distributions across mini-batch dimensions. Normalization statistics calculate empirical mean and variance over batch sizes of 64 training examples. If running variance moving averages diverge beyond a 0.7 threshold, momentum updates re-calibrate batch statistics. Learnable affine scale and shift parameters restore network representational capacity after zero-mean scaling. Modern vision architectures integrate batch normalization immediately prior to non-linear activation functions.""",

    """Layer normalization computes normalization statistics independently across feature channel dimensions for each sequence sample. Deep transformer stacks apply pre-layer normalization across 6 sequential attention layers. The optimization pipeline configures learning rate warmup targets of 0.1 to prevent gradient explosion. Unlike batch normalization, layer normalization computes identical statistics during training and inference without batch dependence. Natural language processing models depend on layer normalization for recurrent and attention sequence processing.""",

    """PostgreSQL manages relational table storage using page-level multi-version concurrency control. Database server instances listen for incoming client socket connections on port 6379. Connection poolers like PgBouncer disconnect idle backend connections after a 250ms timeout. The write-ahead log ensures crash recovery by persisting transaction records before dirty buffer flushes. Vacuum processes clean dead row versions created by updates and deletes to prevent table bloat.""",

    """MySQL InnoDB implements multi-version concurrency control using undo log segments and clustered index storage. Server daemons accept incoming client connections on default network port 27017. Transaction lock wait timers abort blocked queries if row lock acquisition exceeds 1000ms. Undo logs record prior row versions to reconstruct consistent read views without table-level locking. Purge threads asynchronously deallocate committed undo log pages to reclaim tablespace.""",

    """B+ Tree data structures provide balanced logarithmic search and sequential range scan access. Internal index pages split when node fill capacity surpasses a 0.5 saturation ratio. Bulk insert workloads group leaf index modifications into 256 record chunks to minimize random I/O. Clustered indexes determine physical storage order on primary disk pages to maximize cache locality. Page latching protocols enforce concurrent thread safety during structural page split operations.""",

    """Log-structured merge trees optimize write-heavy workloads by buffering incoming modifications in memory before sequential disk writes. Compaction processes merge sorted runs into lower levels when level size saturation exceeds 0.5. Memory tables flush immutable write buffers in chunks of 128 records to sequential SSTable files. Bloom filters allow fast point lookups by determining key absence without performing disk reads. RocksDB and Apache Cassandra use LSM trees to achieve high write throughput.""",

    """MongoDB organizes schemaless documents inside scalable sharded collections across cluster replica sets. Daemon processes listen for client wire traffic over default port 6379 on each node. Write concern requires acknowledgment from a replica quorum of 4 members to confirm persistence. WiredTiger storage engine utilizes log-structured storage algorithms with Snappy compression for high throughput. Secondary nodes asynchronously replay the primary oplog to maintain read replica synchronization.""",

    """Apache Cassandra distributes partition rows across peer-to-peer storage clusters using token-based ring partitioning. Database nodes listen for client CQL requests on native transport port 6379. Consistency levels require acknowledgment from a quorum of 5 replica nodes before write confirmation. Memtables persist incoming mutations sequentially to commit logs before flushing immutable SSTables to disk. Read repairs update out-of-sync replicas asynchronously in the background when replica checksums disagree.""",

    """Redis operates as an in-memory key-value data store optimized for sub-millisecond retrieval. By default the Redis server daemon process binds to network port 6379 for TCP traffic. Client connections terminate when idle connection durations exceed the configured 100ms limit. Redis persistence snapshotting writes point-in-time RDB files and appends commands to the AOF log. The LRU eviction algorithm frees memory when maximum memory limits are reached by dropping stale keys.""",

    """Memcached provides high-throughput distributed memory object caching via a multi-threaded slab allocator architecture. Daemon worker threads bind to default TCP listening port 6379 to serve cache requests. Connection timeouts terminate stalled client sockets when inactivity intervals surpass 100ms. Slab allocation avoids memory fragmentation by pre-allocating memory into slabs of specific chunk sizes. Unlike Redis, Memcached does not support disk persistence or complex data structures like sets and hashes.""",

    """Write-ahead logging guarantees ACID durability by persisting database modification logs before changes write to table data pages. Checkpoint background processes flush dirty shared buffers when checkpoint log capacity crosses 0.85 utilization. Group commit buffers batch concurrent transaction flush requests into 32 log entries per fsync disk flush. The ARIES recovery algorithm performs analysis, redo, and undo passes to recover consistent database state following crashes. Relational databases depend on WAL to survive abrupt power outages without data corruption.""",

    """Shadow paging provides atomic transaction durability by duplicating page table directories rather than maintaining write-ahead logs. The database directory flushes new page pointer arrays when modification thresholds exceed a 0.5 update ratio. Transaction prepare phases timeout and abort dirty shadow copies when commit execution exceeds 1000ms. During commit, atomic pointer swapping updates the current page directory pointer to the shadow copy. SQLite supports shadow paging in rollback journal mode to ensure crash recovery without log replaying.""",

    """Transmission Control Protocol establishes reliable bidirectional byte streams using a three-way handshake. Web servers accept secure transport connections through standard listening port 27017. Retransmission timers detect dropped segments when unacknowledged packets exceed a 500ms RTO duration. Sliding window flow control advertises receiver buffer capacity to prevent receiver buffer exhaustion. Congestion control algorithms like BBR dynamically estimate bottleneck bandwidth and round-trip propagation time.""",

    """QUIC replaces legacy TCP transport by implementing multiplexed connection streams directly over UDP. Server listener sockets bind to transport port 9200 to receive client datagram packets. Stream loss detection initiates packet retransmission when probe timers exceed a 500ms loss timeout. By implementing stream multiplexing on UDP, QUIC prevents head-of-line blocking when individual packets drop. HTTP/3 mandates QUIC transport to achieve zero-RTT connection resumption and migration resilience.""",

    """The Domain Name System translates human-readable hostnames into routable internet protocol addresses. Recursive DNS resolvers query authoritative name servers over UDP transport port 6379. Resolver clients timeout unfulfilled recursive lookups when upstream authority responses exceed 250ms. Time-to-live attributes specify how long intermediate recursive caches retain authoritative record answers. DNSSEC signs resource record sets cryptographically using asymmetric keys to prevent cache poisoning attacks.""",

    """Multicast DNS discovers network services on local link subnets without requiring centralized DNS servers. Resolvers broadcast lookup datagrams over local multicast port 9200 to all peer hosts on the subnet. Local responder services timeout unacknowledged service discovery requests if peer responses exceed 500ms. Queries append the dot-local top-level domain to restrict discovery traffic within the local broadcast domain. Apple Bonjour and Linux Avahi use mDNS for zero-configuration printer and device discovery.""",

    """HTTP/2 introduces binary framing to interleave concurrent request and response streams over a single TCP connection. Flow control credits throttle individual stream delivery when stream buffer usage crosses a 0.9 threshold. Frame multiplexing coalesces smaller binary payload frames into groups of 64 bytes per transmission unit. HPACK header compression reduces metadata transmission overhead by maintaining shared dynamic indexing tables. Stream prioritization allows browser clients to request critical CSS and script resources ahead of decorative images.""",

    """HTTP/3 redesigns web application transport by replacing TCP with QUIC over UDP to eliminate head-of-line blocking. Stream flow control limits buffer consumption when stream receiver buffer saturation crosses 0.7. Frame encoders group QPACK header instructions into compression blocks of 64 bytes per frame. QPACK accommodates out-of-order stream delivery by decoupling dynamic header tables from individual streams. Modern web browsers negotiate HTTP/3 via Alt-Svc headers to improve loading speeds on mobile networks.""",

    """Transport Layer Security encrypts communication channels to protect confidentiality and integrity across the internet. Cipher suites negotiate ephemeral Diffie-Hellman key exchange with 192 bits of cryptographic security. The TLS 1.3 handshake reduces connection setup latency by completing in a single round trip under 100ms. Zero-RTT resumption permits returning clients to transmit encrypted application data on the initial flight. Certificate validation ensures server identity by verifying the trust chain back to installed root certificate authorities.""",

    """TLS 1.2 negotiates secure transport sessions through a two-round-trip handshake between client and server. Ephemeral Diffie-Hellman parameter generation establishes forward secrecy using discrete logarithm groups of 192 bits. Initial handshake negotiation terminates if client hello responses encounter network delays exceeding 1000ms. Static RSA key exchange in TLS 1.2 is deprecated due to vulnerability to padding oracle attacks. Session tickets allow returning clients to bypass full key exchange by re-using serialized session state.""",

    """Border Gateway Protocol manages inter-domain routing decisions across autonomous systems on the global internet. Peer router daemons establish exterior BGP peering sessions over TCP port 9200. Peering keepalive timers tear down inactive BGP neighbor sessions when keepalive arrivals exceed 250ms. Path attributes like AS-PATH prevent routing loops by rejecting path vectors containing the receiving autonomous system. BGP routers exchange reachability updates to dynamically calculate optimal autonomous system paths.""",

    """Open Shortest Path First computes optimal intra-domain routing paths within autonomous networks using link-state routing. OSPF routers send link-state advertisements over IP transport port 9200 to synchronize topology databases. Neighbor adjacency fails if hello heartbeats from adjacent routers stop arriving within a 1000ms dead interval. Dijkstra algorithm executes over synchronized link-state databases to compute shortest path trees from each router. OSPF partitions large enterprise networks into hierarchical areas to constrain flooding overhead.""",

    """Public key cryptography enables confidential communication and non-repudiation between untrusted network endpoints. Asymmetric key generators create digital certificates using RSA key lengths of 192 bits. Digital signature verification combines private key signing with a SHA-256 digest of the underlying payload. Hybrid cryptosystems combine asymmetric key exchange with symmetric stream encryption for performance efficiency. Perfect forward secrecy ensures past session keys remain secure even if long-term private keys are compromised.""",

    """Elliptic curve cryptography provides equivalent public-key security to RSA using substantially smaller key representations. Digital signature algorithms like ECDSA achieve high-grade cryptographic security using elliptic curves of 192 bits. Cryptographic message signing hashes payload inputs using a SHA-256 digest function before curve multiplication. Smaller key sizes reduce memory footprints and accelerate cryptographic handshake execution on constrained devices. Ed25519 and Secp256r1 curves represent standard elliptic curve implementations in modern security protocols.""",

    """Secure credential storage requires salting and iterative key stretching before persisting user secrets in databases. Modern cryptographic hashing pipelines compute password derivations using hardened SHA-3 algorithms with random salts. Work factor tuning balances offline attack resistance against server verification latency budgets capped at 250ms. Memory-hard functions like Argon2 protect against specialized hardware brute-force attacks via parallel ASIC chips. Slow verification deliberately imposes computation costs to render offline dictionary attacks infeasible.""",

    """PBKDF2 derives cryptographic encryption keys from human passwords by repeatedly applying pseudorandom functions. The key stretching pipeline iterates HMAC evaluations using an underlying SHA-256 cryptographic hash digest. System administrators adjust iteration counts to keep derivation execution latency bounded near 1000ms per attempt. Unlike memory-hard algorithms, PBKDF2 relies primarily on computational CPU cycles without intensive RAM consumption. Modern cryptographic standards recommend high iteration counts to mitigate GPU-accelerated dictionary attacks.""",

    """Web application firewalls inspect inbound HTTP requests to intercept SQL injection and cross-site scripting attempts. Rate limiting algorithms throttle client IP addresses when request frequencies exceed a 0.85 burst threshold. Offending clients are blocked from executing sensitive API endpoints for a penalty window of 100ms. Content Security Policy headers restrict the origins from which scripts, styles, and fonts may be executed. Cross-Site Request Forgery tokens protect state-changing POST requests by proving user origin authenticity.""",

    """API gateways protect backend microservices from resource exhaustion by enforcing token bucket rate limiting. Token leak algorithms throttle client consumers when bucket consumption surpasses a 0.5 burst capacity limit. Throttled client requests receive HTTP 429 status codes with retry-after headers specifying a 500ms wait duration. Token bucket algorithms allow temporary bursts of traffic while guaranteeing sustained request rates stay bounded. Kong and AWS API Gateway use token buckets to manage multi-tenant API quotas.""",

    """Advanced Encryption Standard delivers authenticated symmetric encryption for data at rest and in transit. Secure storage engines configure AES in GCM mode using symmetric keys of 512 bits. Message authentication codes verify payload integrity using HMAC combined with a BLAKE2 cryptographic function. Initialization vectors must be unique for every encryption operation to prevent replay and ciphertext manipulation. Key rotation policies periodically replace symmetric master keys to limit data exposure windows.""",

    """ChaCha20-Poly1305 provides high-speed authenticated stream encryption optimized for software-only execution without hardware AES acceleration. Cipher implementations initialize stream ciphers using symmetric keys of 128 bits. The Poly1305 authenticator produces 128-bit integrity tags by processing ciphertext blocks with a SHA-256 primitive. Because ChaCha20 avoids table lookups, it offers natural resistance to cache-timing side-channel attacks on general CPUs. WireGuard and mobile TLS connections default to ChaCha20-Poly1305 for efficient cryptographic processing.""",

    """OAuth 2.0 Proof Key for Code Exchange secures authorization code grants on public mobile and single-page applications. The client transforms a high-entropy secret code verifier using a BLAKE2 cryptographic digest to create the code challenge. Authorization servers reject authorization requests if code verification latencies exceed a 100ms grant window. PKCE eliminates authorization code interception attacks by requiring possession of the original plaintext code verifier. Modern identity providers require PKCE for all public client authentication flows.""",

    """JSON Web Tokens transmit cryptographically verifiable claims between distributed services in compact URL-safe strings. Digital signatures verify JWT payload tampering by generating cryptographic signatures using SHA-3 digests. Verification middleware rejects expired tokens when expiration timestamp differences exceed a 250ms clock skew threshold. Tokens comprise base64url-encoded header, payload, and cryptographic signature segments separated by periods. Stateless microservices use JWTs to transmit authenticated identity claims without querying central session stores."""
]


# ============================================================
# 3. CONCEPT ANCHORS
# Helps distinguish similar technical topics.
# ============================================================

CONCEPTS = [
    "multi-paxos", "two-phase commit", "hybrid logical clocks",
    "rendezvous hashing", "consistent hashing", "vector clocks",
    "stochastic gradient descent", "support vector machines",
    "logistic regression", "transformer encoders",
    "autoregressive transformer decoders", "multi-layer perceptrons",
    "residual convolutional networks", "batch normalization",
    "layer normalization", "mysql innodb", "postgresql",
    "log-structured merge trees", "write-ahead logging",
    "shadow paging", "multicast dns", "transmission control protocol",
    "domain name system", "border gateway protocol",
    "open shortest path first", "elliptic curve cryptography",
    "public key cryptography", "web application firewalls",
    "api gateways", "advanced encryption standard",
    "chacha20-poly1305", "json web tokens",
    "raft", "gossip", "swim", "saga", "adamw", "mongodb",
    "cassandra", "redis", "memcached", "b+ tree", "quic",
    "http/2", "http/3", "tls 1.3", "tls 1.2",
    "pbkdf2", "pkce", "oauth"
]

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or",
    "of", "to", "in", "on", "for", "by", "with", "from", "at",
    "how", "what", "which", "who", "when", "where", "why",
    "does", "do", "did", "can", "could", "would", "should",
    "explain", "tell", "me", "about", "give", "provide",
    "used", "use", "uses", "using", "its", "their", "this",
    "that", "these", "those", "it", "be", "been", "being",
    "has", "have", "had", "as", "than", "then", "into",
    "through", "between", "after", "before", "also"
}


# ============================================================
# 4. TEXT PROCESSING
# ============================================================

def tokenize(text):
    return set(
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    )


def find_anchors(question):
    q = question.lower()
    return [
        concept for concept in CONCEPTS
        if concept in q
    ]


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


# ============================================================
# 5. LOAD EMBEDDING MODEL AND KNOWLEDGE BASE
# ============================================================

@app.on_event("startup")
def load_knowledge_base():
    global embedding_model, passage_embeddings

    embedding_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir="/tmp/fastembed_cache"
    )

    vectors = list(
        embedding_model.embed(PASSAGES)
    )

    passage_embeddings = np.asarray(
        vectors, dtype=np.float32
    )

    norms = np.linalg.norm(
        passage_embeddings, axis=1, keepdims=True
    )

    passage_embeddings = passage_embeddings / np.maximum(
        norms, 1e-12
    )

    print("RAG knowledge base loaded:", len(PASSAGES))


# ============================================================
# 6. RETRIEVAL USING COSINE SIMILARITY
# ============================================================

def retrieve(question, top_k):
    with model_lock:
        q_vector = np.asarray(
            list(embedding_model.embed([question]))[0],
            dtype=np.float32
        )

    q_vector = q_vector / max(
        float(np.linalg.norm(q_vector)), 1e-12
    )

    semantic_scores = passage_embeddings @ q_vector

    q_tokens = tokenize(question)
    anchors = find_anchors(question)

    results = []

    for idx, passage in enumerate(PASSAGES):

        p_tokens = tokenize(passage)

        if q_tokens:
            overlap = len(q_tokens & p_tokens)
            lexical_score = overlap / len(q_tokens)
        else:
            lexical_score = 0.0

        # Prevent a specific named concept from being
        # confused with a different concept.
        if anchors:
            passage_lower = passage.lower()
            if not any(a in passage_lower for a in anchors):
                continue

        semantic = max(0.0, float(semantic_scores[idx]))

        # Combine semantic and lexical relevance.
        combined_score = (
            0.75 * semantic +
            0.25 * lexical_score
        )

        results.append({
            "id": idx,
            "score": float(np.clip(combined_score, 0, 1)),
            "semantic": semantic,
            "passage": passage
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 7. FACTUAL ANSWER EXTRACTION
# ============================================================

def extract_answer(question, results):

    q_tokens = tokenize(question)
    anchors = find_anchors(question)

    sentence_results = []

    for result in results:

        for sentence in sentence_split(result["passage"]):

            s_tokens = tokenize(sentence)

            if not s_tokens or not q_tokens:
                continue

            overlap = q_tokens & s_tokens

            if not overlap:
                continue

            coverage = len(overlap) / len(q_tokens)
            precision = len(overlap) / len(s_tokens)

            # Prefer sentences that directly support the question.
            relevance = (
                0.75 * coverage +
                0.25 * precision
            )

            # Require explicit concept evidence when a named
            # technical concept appears in the question.
            if anchors:
                if not any(
                    anchor in sentence.lower()
                    for anchor in anchors
                ):
                    continue

            if relevance >= 0.12:
                sentence_results.append({
                    "sentence": sentence,
                    "passage_id": result["id"],
                    "relevance": relevance,
                    "coverage": coverage
                })

    sentence_results.sort(
        key=lambda x: x["relevance"],
        reverse=True
    )

    if not sentence_results:
        return None, []

    # Keep at most three distinct evidence sentences.
    selected = []
    seen = set()

    for item in sentence_results:
        sentence = item["sentence"]

        if sentence not in seen:
            selected.append(item)
            seen.add(sentence)

        if len(selected) == 3:
            break

    # A weak lexical match is not enough to answer.
    if selected[0]["coverage"] < 0.15:
        return None, []

    answer = " ".join(
        item["sentence"] for item in selected
    )

    source_ids = list(dict.fromkeys(
        item["passage_id"] for item in selected
    ))

    return answer, source_ids


# ============================================================
# 8. API REQUEST AND RESPONSE SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: List[int]
    scores: List[float]
    confidence: float


# ============================================================
# 9. MAIN QUERY ENDPOINT
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.1
        )

    if embedding_model is None or passage_embeddings is None:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.1
        )

    # Instructions in a question cannot override the fixed
    # knowledge base or change application behavior.
    results = retrieve(question, request.top_k)

    if not results or results[0]["score"] < 0.35:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.2
        )

    answer, evidence_ids = extract_answer(
        question, results
    )

    if answer is None:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.2
        )

    # Return sources that actually support extracted sentences.
    evidence_results = [
        result for result in results
        if result["id"] in evidence_ids
    ]

    evidence_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    sources = [
        result["id"] for result in evidence_results
    ]

    scores = [
        round(float(result["score"]), 4)
        for result in evidence_results
    ]

    best_score = evidence_results[0]["score"]

    confidence = min(
        0.95,
        max(0.61, 0.61 + 0.30 * best_score)
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        scores=scores,
        confidence=round(confidence, 4)
    )


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "application": "TDS Week 5 RAG",
        "passages": len(PASSAGES),
        "endpoint": "/query"
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if passage_embeddings is not None else "loading",
        "passages_loaded": len(PASSAGES),
        "embedding_model": MODEL_NAME
    }


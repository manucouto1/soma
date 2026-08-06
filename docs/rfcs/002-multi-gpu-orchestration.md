# RFC-002: Multi-GPU Orchestration over PCIe

**Status:** Draft  
**Date:** 2026-04-09  
**Author:** Manuel Couto Pintos  

## Summary

Define how Soma orchestrates multi-GPU workloads (training and inference) across
workers that may only have PCIe interconnect (no NVLink). Leverage existing
`TrainingStrategy`, `Distribution`, and `Scheduler` primitives while delegating
low-level GPU communication to established frameworks (PyTorch, DeepSpeed,
vLLM, SGLang).

## Motivation

Soma already distributes work across workers on different servers. However,
users also need to:

1. **Select specific GPUs** for specific filters (device placement).
2. **Split a single model** across GPUs within the same node (model/tensor/pipeline parallelism).
3. **Run LLM inference** across multiple GPUs for agent graphs (Nous use case).
4. **Mix strategies**: data parallel across nodes + tensor/pipeline parallel within a node.

NVLink is not always available (consumer GPUs, cloud instances, HPC nodes with
PCIe switches). The design must work efficiently over PCIe while taking advantage
of NVLink when present.

## Context: PCIe vs NVLink

| Interconnect       | Bandwidth (bidirectional) |
|--------------------|--------------------------|
| PCIe 4.0 x16       | ~64 GB/s                 |
| PCIe 5.0 x16       | ~128 GB/s                |
| NVLink 3.0 (A100)  | 600 GB/s                 |
| NVLink 4.0 (H100)  | 900 GB/s                 |

The 5-15x gap dictates which parallelism strategies are viable over PCIe:

| Strategy                    | PCIe 4.0 penalty vs NVLink | Viable? |
|-----------------------------|----------------------------|---------|
| Data Parallel (ZeRO-1/2)   | 5-15%                      | Yes     |
| Pipeline Parallel           | 5-15%                      | Yes     |
| FSDP / ZeRO-3              | 25-50%                     | Acceptable |
| Tensor Parallel (TP=2)     | 20-40%                     | Marginal |
| Tensor Parallel (TP=4+)    | 40-60%                     | Avoid   |

**Rule of thumb**: Data Parallel > Pipeline Parallel > FSDP/ZeRO-3 > Tensor Parallel on PCIe.

## Existing Soma Primitives

These already exist and form the foundation:

- **`TrainingStrategy`** (`soma-core/src/distributed.rs`): `DataParallel`, `ModelParallel`,
  `Federated`, `PopulationBased`, `Custom` — graph-level attribute.
- **`Distribution`** (`soma-core/src/graph/filter.rs`): `Local | Remote(WorkerId | Tag) | Any`
  — per-filter placement hint.
- **`ExecutionPlan::Remote`** (`soma-compiler/src/plan.rs`): wraps a sub-plan for remote execution.
- **`ExecutionPlan::Composite`** : groups differentiable nodes into indivisible blocks
  — maps naturally to pipeline stages.
- **`Scheduler`** (`soma-compiler/src/scheduler.rs`): assigns nodes to workers, generates
  `DistributionPlan` with data transfers.
- **`DataStore` + `DataRef`** (`soma-core/src/store/`): S3, Zarr, Inline, Stream for
  inter-worker data movement.
- **`Worker` + `EnvManager`** (`soma-worker/`): isolated Python venvs per worker,
  cloudpickle serialization, `CUDA_VISIBLE_DEVICES` support.

## Design: Three Integration Layers

The design is layered so that each level can be adopted independently.

### Layer A: Soma as Orchestrator, Frameworks as Engines (no core changes)

Filters encapsulate GPU logic internally. Soma orchestrates at the graph level.

```python
class EncoderStage(Filter):
    """A filter that uses PyTorch TP internally."""
    def __init__(self, tp_size=2):
        self.tp_size = tp_size

    def fit(self, x, y=None):
        mesh = init_device_mesh("cuda", (self.tp_size,))
        parallelize_module(self.model, mesh, self.tp_plan)
        train(self.model, x, y)
        return {"state": self.model.state_dict()}

    def forward(self, x, state):
        self.model.load_state_dict(state["state"])
        return self.model(x)

class DecoderStage(Filter):
    ...

# Pipeline parallelism via Soma graph structure
g = Graph.somatize(EncoderStage(tp_size=2) >> DecoderStage(tp_size=2))
g.node("encoder", target="gpu-server-1")  # GPUs 0,1
g.node("decoder", target="gpu-server-2")  # GPUs 2,3
```

**Mapping:**

| Soma concept            | Multi-GPU role                              |
|-------------------------|---------------------------------------------|
| `Filter` with target    | Manual device placement                     |
| `Subgraph A >> B`       | Pipeline parallelism (2 stages)             |
| `TrainingStrategy::DataParallel` | FSDP/ZeRO over N workers          |
| `ExecutionPlan::Composite`       | Indivisible pipeline stage        |
| `ExecutionPlan::Parallel`        | Data parallelism branches         |
| `Partition(nodes, worker)`       | Arbitrary model parallelism       |

**Pros:** zero core changes, works today, compatible with any framework.  
**Cons:** user must know PyTorch TP/DeepSpeed internals; GPU config is imperative, not declarative.

### Layer B: Declarative Device Mesh in Graph

Extend the distribution model with explicit device placement.

#### New types

```rust
// soma-core/src/device.rs (new)

/// Identifies a device (GPU, CPU, accelerator).
pub enum DeviceId {
    Cuda(u32),          // cuda:0, cuda:1, ...
    Cpu,
    Custom(String),     // "tpu:0", "mps:0"
}

/// A group of devices that can communicate efficiently.
pub struct DeviceGroup {
    pub id: String,
    pub devices: Vec<DeviceId>,
    pub interconnect: Interconnect,
}

pub enum Interconnect {
    NVLink,
    PCIe { gen: u8 },     // 4, 5
    Network,              // cross-node
    Unknown,
}
```

#### Extended Node

```rust
// Extend existing Node
pub struct Node {
    pub id: NodeId,
    pub target: Option<String>,      // existing: worker tag
    pub devices: Option<Vec<DeviceId>>,  // NEW: explicit GPU placement
    // ...
}
```

#### User API

```python
g = Graph()
g.add_node("embed",   EmbeddingFilter(), device="cuda:0")
g.add_node("encoder", EncoderFilter(),   device=["cuda:0", "cuda:1"])  # TP=2
g.add_node("decoder", DecoderFilter(),   device=["cuda:2", "cuda:3"])  # TP=2
g.add_node("head",    ClassifierFilter(), device="cuda:3")
```

#### Compiler changes

The compiler resolves `devices` into the execution plan:

1. Detect GPU topology (`nvidia-smi topo -m` or NVML bindings).
2. Group devices by interconnect quality (NVLink groups, PCIe switch groups).
3. If `node.devices` spans a single group → allow TP (intra-group communication is fast).
4. If `node.devices` spans multiple groups → warn or reject TP, suggest PP.
5. Generate `CUDA_VISIBLE_DEVICES` env var per worker.
6. Insert `DataTransfer` nodes between pipeline stages on different device groups.

**Pros:** declarative API, compiler can validate and optimize placement.  
**Cons:** couples Soma to GPU topology detection, significant scheduler complexity,
CUDA-specific (needs abstraction for ROCm, Metal, etc.).

### Layer C: LLM Inference Backends (vLLM / SGLang)

For inference in agent graphs (Nous use case), wrap vLLM/SGLang as filters.

```python
class LLMFilter(Filter):
    """Filter backed by a vLLM or SGLang server."""
    def __init__(self, model: str, tp_size: int = 1, backend: str = "sglang"):
        self.model = model
        self.tp_size = tp_size
        self.backend = backend

    def forward(self, x, state):
        # x["prompt"] → LLM → response
        response = self.client.generate(x["prompt"], **x.get("params", {}))
        return {"text": response.text, "usage": response.usage}

# Agent graph with heterogeneous LLMs
g = Graph()
g.add_node("planner",  LLMFilter("llama-3-70b", tp_size=4))
g.add_node("coder",    LLMFilter("codestral-22b", tp_size=1))
g.add_node("reviewer", LLMFilter("llama-3-70b", tp_size=4))  # shares server
g.connect("planner", "coder")
g.connect("planner", "reviewer")
```

**Why SGLang specifically:**
- **RadixAttention** (tree-based prefix caching) maps naturally to Soma graphs
  where branches share context.
- **DP attention mode**: if model fits on 1 GPU, replicate instead of TP (0% communication
  overhead — better than TP on PCIe).
- Competitive with vLLM, 20-40% faster on multi-turn/branching workloads.

**Pros:** high value for Nous agent graphs, minimal core changes, inference engines handle TP/KV-cache.  
**Cons:** inference only, requires external server processes.

## Recommendations by Use Case

| Use case                          | PCIe strategy          | Soma layer |
|-----------------------------------|------------------------|------------|
| Train model that fits on 1 GPU    | Data Parallel (ZeRO-1/2) | A — existing `TrainingStrategy::DataParallel` |
| Train model too large for 1 GPU   | Pipeline Parallel + ZeRO-2 | A or B — filters = pipeline stages |
| LLM inference multi-GPU           | TP=2 or SGLang DP      | C — vLLM/SGLang as filters |
| Agent graphs with LLMs            | SGLang DP attention     | C — RadixAttention + Soma graphs |
| User-defined model partitioning   | Manual device placement | B — `device=["cuda:0", "cuda:1"]` |
| Federated learning                | Data stays local        | A — existing `TrainingStrategy::Federated` |
| Population-Based Training         | 1 GPU per trial         | A — existing `TrainingStrategy::PopulationBased` |

## Trade-offs and Risks

### In favor of this design

- **Separation of concerns**: topology (Graph) → compilation (Plan) → distribution (Scheduler)
  mirrors what PyTorch 2.x does with `DeviceMesh`. This is the right architecture.
- **`Filter.fit()` + `Filter.forward()`** maps cleanly to pipeline stages.
- Worker system with cloudpickle is analogous to Ray/Dask — serialization and code shipping
  are already solved.
- `DataStore` with Zarr chunks works for large activation tensors between stages.

### Risks

- **Do not reinvent NCCL**: GPU-to-GPU collectives (all-reduce, all-gather) are extremely
  optimized in NCCL. Soma must delegate to PyTorch/DeepSpeed inside filters, not attempt
  its own GPU communication.
- **GPU topology detection**: parsing `nvidia-smi topo -m`, handling PCIe switches, NVLink
  domains — complex and platform-specific. Consider using NVML bindings (`nvml-wrapper` crate).
- **Heterogeneous GPUs**: no framework supports this well. Soma could differentiate here
  (e.g., scheduler assigns fewer layers to weaker GPUs in PP) but it's hard.
- **DataStore latency**: S3 or NFS for activations between pipeline stages within the same
  node is too slow. Needs `DataRef::SharedMemory` or `DataRef::GpuDirect` for intra-node
  transfers.
- **Abstraction leak**: CUDA-specific concepts (device IDs, TP groups) may not generalize
  to ROCm, Metal, or TPUs without abstraction.

## Roadmap

### Phase 1 — Examples and Documentation (no core changes)

- Document Layer A patterns: filters wrapping FSDP, DeepSpeed, pipeline parallel.
- Create example notebooks showing multi-GPU training via Soma workers.
- Validate that `TrainingStrategy::DataParallel` + `CUDA_VISIBLE_DEVICES` works end-to-end.

### Phase 2 — LLM Inference Filters (Layer C)

- Implement `LLMFilter` wrapper for vLLM and SGLang.
- Support shared servers (multiple filters pointing to same inference endpoint).
- Integrate with Nous agent graphs.
- Benchmark RadixAttention benefit for branching agent workloads.

### Phase 3 — Declarative Device Placement (Layer B)

- Add `DeviceId`, `DeviceGroup`, `Interconnect` types to `soma-core`.
- Extend `Node` with `devices: Option<Vec<DeviceId>>`.
- GPU topology detection via NVML.
- Scheduler generates `CUDA_VISIBLE_DEVICES` and validates placement.
- Compiler warns when TP is used across slow interconnects.

### Phase 4 — Intra-Node Fast Transport

- `DataRef::SharedMemory` for activation transfer within a node (bypass S3/NFS).
- `DataRef::GpuDirect` for CUDA IPC (direct GPU memory transfer over PCIe/NVLink).
- Integrates with RFC-001 (P2P transport) for cross-node transfers.
- Benchmark against direct PyTorch `dist.send`/`dist.recv`.

## Future Considerations

- **Automatic parallelism**: compiler analyzes model size, available GPUs, and interconnect
  to automatically choose DP/PP/TP mix (similar to Alpa/Unity).
- **Elastic training**: handle GPU failures gracefully — reshard and continue without restart.
- **Heterogeneous scheduling**: scheduler accounts for GPU compute capability and VRAM
  when assigning pipeline stages.
- **Quantization-aware placement**: place quantized layers on weaker GPUs, full-precision
  on stronger ones.
- **Profiling-guided optimization**: measure actual communication/compute ratio and
  re-partition automatically.

## References

- PyTorch Tensor Parallel: `torch.distributed.tensor.parallel`
- PyTorch FSDP2: `torch.distributed.fsdp`
- DeepSpeed ZeRO: https://www.deepspeed.ai/
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- NCCL: https://developer.nvidia.com/nccl
- Alpa (automatic parallelism): https://github.com/alpa-projects/alpa

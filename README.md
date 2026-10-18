# cumux

**CUDA for GPU-less microVMs.** cumux lets a Firecracker microVM with no GPU run CUDA programs anyway. Each CUDA call is forwarded over vsock to a host daemon that owns the real GPU, so many VMs can share one card. The name is CUDA + mux (multiplexer).

> **Status: early development.** Work started October 2026. Phase 1 (below) is in progress, and nothing here is usable yet.

---

## Why

[Firecracker](https://firecracker-microvm.github.io/) microVMs boot in a fraction of a second and can be snapshotted and restored, which makes them a great fit for serverless. Firecracker deliberately supports very few devices, though, and GPUs aren't among them.

The usual way to give a VM a GPU is PCIe passthrough, and it comes with two costs:

- **No sharing.** The whole card belongs to one VM.
- **No snapshots.** GPU state lives inside the VM, so freezing and restoring the VM breaks.

So today you choose between fast, tiny VMs and GPUs. cumux tries to give you both. The VM sees a normal CUDA API but holds only opaque handles, while all real GPU state stays on the host.

## How it works

```
 microVM (no GPU)                          Host
┌──────────────────┐                ┌───────────────────────┐
│ vectorAdd app    │                │ host daemon           │
│   ↓ cudaMalloc() │     vsock      │  runs real CUDA call  │
│ guest stub (.so) │ ─────────────► │  keeps per-VM table   │
│  packs message   │ ◄───────────── │  returns result       │
└──────────────────┘                └──────────┬────────────┘
                                           real GPU
```

Think of it as a drive-thru. The microVM is a car with no kitchen. It places orders through a speaker box (the guest stub) over a wire (vsock) to a kitchen (the host daemon) with one stove (the GPU). The car only ever holds receipts (device pointers and other handles), never the food. That is why it can be parked and frozen.

### Components

| Component | What it does |
|---|---|
| **Guest stub** | A shared library loaded with `LD_PRELOAD`. It defines `cudaMalloc`, `cudaMemcpy`, kernel launch and the rest with the same signatures as the CUDA runtime, and turns each call into a message. |
| **Wire protocol** | Length-prefixed messages ("N bytes follow, here they are"), carrying a call ID, arguments and any bulk data. |
| **Transport** | `AF_UNIX` during development on a single host. `AF_VSOCK` between a guest and its host, which changes only the address struct. |
| **Host daemon** | Listens on one socket and gets one connection per VM. It runs the real CUDA calls and tracks which GPU allocations belong to which VM. |

## Roadmap

### Phase 1: vectorAdd inside a GPU-less microVM

- [ ] **Intercept.** The shim logs `intercepted cudaMalloc` and forwards to the real runtime.
- [ ] **Split across a socket.** The guest side sends the call and the host side runs it (same machine, `AF_UNIX`).
- [ ] **Bulk copy.** `cudaMemcpy` ships buffer bytes in both directions.
- [ ] **Kernel launch.** Ship the fatbin to the host and load it with the driver API (`cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel`).
- [ ] **Move into the VM.** Switch to `AF_VSOCK`, boot Firecracker, and run the sample inside it.

### Later

- Several microVMs sharing one GPU, with memory isolation, oversubscription and scheduling across tenants
- Request routing and autoscaling
- A thin invoke endpoint with scale-to-zero, which together with the items above makes a serverless GPU platform for microVMs

## Known hard problems

- **Fatbin registration happens before `main`.** CUDA programs register their device code through hidden calls (`__cudaRegisterFatBinary`, `__cudaRegisterFunction` and related functions) before user code runs. The stub has to intercept these too, or a kernel launch has nothing to resolve to.
- **Static linking.** `nvcc` links the CUDA runtime statically by default, so `LD_PRELOAD` intercepts nothing. Applications must be built with `-cudart shared`.
- **Snapshots and handle validity.** A handle is only meaningful while the daemon still holds the matching GPU state. Restoring a snapshot on another host, or after the daemon restarts, leaves dangling handles. The options are:
  1. Restore only on the same host while the daemon is running.
  2. Checkpoint the daemon's GPU state as well (for example with `cuda-checkpoint`).
  3. Snapshot only when the VM has no live GPU state.
- **Latency.** Every call is a round trip, so workloads that make many small calls pay for each one. Batching and asynchronous forwarding are possible mitigations.

## Prior art

Forwarding CUDA calls to another machine isn't a new idea. See [rCUDA](http://www.rcuda.net/), [gVirtuS](https://github.com/gvirtus/GVirtuS) and VMware Bitfusion. cumux focuses on a specific combination: **Firecracker microVMs, vsock transport, and keeping guests snapshotable** by keeping all GPU state out of the VM.

## Development environment

- Bare-metal Linux with KVM available (`/dev/kvm`)
- An NVIDIA GPU with driver and CUDA toolkit installed (development uses an RTX 2050)
- Firecracker (needed only from the "Move into the VM" step on)

Build and run instructions will be added once the first intercept step lands.

## License

TBD

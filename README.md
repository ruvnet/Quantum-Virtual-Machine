# Quantum-Virtual-Machine-
A backend-agnostic Rust crate with WASM support for quantum virtual machine scheduling. It parses OpenQASM 3, partitions device topologies into qubit tiles with buffer zones, bin-packs multiple jobs in space and time, and outputs a composite, provider-neutral QASM program with isolated classical results for each job.          Ask ChatGPT

# Quantum Virtual Machine (QVM) Scheduler and Runtime (Rust Crate)

## Introduction

Quantum hardware resources are scarce, and running multiple quantum circuits (jobs) in parallel can dramatically improve utilization and reduce user wait times. However, naively combining circuits on one chip can introduce **crosstalk** and fidelity loss – one circuit’s operations can disrupt another if qubits are too close. This Rust crate provides a **backend-agnostic Quantum Virtual Machine (QVM) scheduler** and runtime to safely execute multiple quantum programs on a single device. It ingests quantum circuits (in OpenQASM 3 or an internal IR) and schedules them onto a target hardware topology, partitioning qubits into isolated regions (“tiles”) to mitigate interference. The output is a **composite OpenQASM 3** program that can be executed on real hardware or a simulator, with all jobs **multiplexed in space and time**. The design emphasizes modularity, **WASM compatibility** for browser integration, and independence from any specific quantum vendor.

**Key Features and Objectives:**

* **OpenQASM 3 Ingestion & IR:** Accepts circuits described in OpenQASM 3 (an imperative quantum assembly language) or a high-level intermediate representation. Parses and type-checks input programs into an internal quantum circuit IR for scheduling.
* **Backend Topology & Qubit Tiling:** Represents the physical qubit coupling graph of a backend (connectivity map) and partitions it into **reusable qubit “tiles”** – contiguous subgraphs of qubits sized to fit circuits. Optionally, it can enforce buffer zones (idle qubits) around tiles to reduce cross-circuit crosstalk.
* **Multi-Job Scheduling (Space & Time):** Implements a **bin-packing scheduler** that assigns multiple circuit jobs to disjoint tiles (concurrently, for spatial multiplexing) and queues additional jobs in time slots if the device doesn’t have free qubits (temporal multiplexing). It optimizes for minimal total batches (time slices) while respecting hardware constraints.
* **Composite Circuit Composition:** Merges scheduled jobs into one composite circuit program. Each job’s quantum operations are mapped onto its assigned physical qubits, and measurement outputs are offset to distinct classical bit ranges for result demultiplexing. This ensures each job’s results can be separated after execution.
* **Provider-Neutral Output:** The final scheduled program is output in standard **OpenQASM 3** format, containing only hardware-agnostic gate instructions, control flow, and timing/barriers (no provider-specific APIs). This can be submitted to any OpenQASM-compatible quantum service or simulator.
* **WASM Compatibility & Async Design:** The crate uses portable data types and minimal `std` features (with an option to compile in `no_std` mode), making it WebAssembly-friendly. The scheduling algorithms are exposed as `async` functions (non-blocking), allowing integration with browser-based GUIs or visualization tools.
* **Examples, Tests, Benchmarks:** The crate is organized with clear modules (detailed below) and provides example usage in both CLI and WASM contexts. Included test vectors verify correctness (e.g. mapping small circuits to a known topology), and basic benchmarks measure scheduling overhead and scalability.

## Crate Architecture Overview

The crate is structured into well-defined modules, each responsible for a specific piece of the QVM scheduling pipeline. The core design follows a flow from input parsing, through scheduling, to output generation, with a clean API tying everything together. The main modules are: **parser/IR**, **topology/tiling**, **scheduler**, **composer**, and **integration (API)**. Below we describe each module’s purpose and inner workings, with code snippets and comments illustrating the design.

### Module: `circuit_ir` – Circuit Representation and Parsing

**Purpose:** Handle input circuits in OpenQASM 3 or a higher-level IR, and represent them internally for scheduling. This module defines data structures for quantum programs (qubits, gates, measurements, etc.) and parses OpenQASM text into this representation.

* *Data Structures:* The core type is a `QuantumCircuit` struct representing a single job’s circuit. It might contain fields like: number of qubits, list of operations, and classical bits used for measurements. For example:

  ```rust
  /// A quantum circuit IR representing a single job.
  pub struct QuantumCircuit {
      pub name: String,
      pub num_qubits: usize,
      pub operations: Vec<Operation>,   // sequence of quantum ops (gates, etc.)
      pub cbits: usize,                // number of classical bits used
  }

  /// An operation on the quantum circuit IR (quantum gate or measurement).
  pub enum Operation {
      Gate { name: String, targets: Vec<Qubit>, controls: Vec<Qubit> },
      Measure { qubit: Qubit, cbit_index: usize },
      Reset { qubit: Qubit },
      Barrier { qubits: Vec<Qubit> },
      // ... (others like delays or conditional gates could be included)
  }
  ```

  Each `Qubit` might be a simple index or an identifier that will later map to a physical qubit. `Measure` operations record which classical bit index they write to.

* *OpenQASM 3 Parsing:* We leverage an existing OpenQASM 3 parser or implement a lightweight parser to populate `QuantumCircuit`. OpenQASM 3 is an imperative quantum assembly language supporting quantum gates and classical control flow. For example, an OpenQASM 3 snippet:

  ```qasm
  qubit[2] q;
  bit[2] c;
  h q[0];
  cx q[0], q[1];
  measure q -> c;
  ```

  would be parsed into a `QuantumCircuit` with 2 qubits and operations `[Gate{"h", [q0]}, Gate{"cx", [q0,q1]}, Measure{q0 -> c0}, Measure{q1 -> c1}]`. The parser also handles classical logic or control if present (though for scheduling, gate sequences and qubit usage are most relevant). We ensure the parser is **WASM-compatible** (e.g., avoiding heavy file I/O; it can take a string input and produce an in-memory IR).

* *Intermediate Representation (IR):* Users can also construct circuits programmatically via an IR API instead of QASM text. For instance, the crate might expose builders like:

  ```rust
  let mut circuit = QuantumCircuit::new("job1", 3);
  circuit.h(0)?.cx(0,1)?.measure(0, 0)?.measure(1, 1)?; 
  ```

  which adds operations to the IR. This IR is designed to be **provider-neutral** and captures the necessary info (qubit count, ops sequence, etc.) for scheduling.

### Module: `topology` – Hardware Topology and Qubit Tiles

**Purpose:** Represent the physical quantum device’s qubit connectivity (topology graph) and provide algorithms to partition it into isolated tiles. This enables assigning different jobs to different regions of the chip with minimal interference.

* *Topology Representation:* We model the hardware as an undirected graph `Topology { qubits: usize, edges: Vec<(u32,u32)> }` where vertices are qubit indices and edges indicate allowed two-qubit operations (couplers). Additional data like qubit fidelities or error rates can be included to guide scheduling (e.g., prefer higher-fidelity qubits). For example, a 7-qubit ring topology might be encoded as:

  ```rust
  let topo = Topology::new(7, vec![(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0)]);
  ```

* *Tile Definition:* A **Tile** is defined as a connected subgraph of the topology of a given size (number of qubits). The scheduler will allocate one tile per concurrent job. To isolate tiles, a *buffer zone* can be enforced: any qubits adjacent to a tile’s qubits are left unused if possible, acting as a guard band to reduce crosstalk between tiles. The buffer size (in terms of graph distance) is configurable (0 = tiles can share edges, 1 = no direct neighbors shared, etc.).

* *Tile Partitioning Algorithm:* The module provides functions to find an available tile for a given circuit. One simple strategy is a greedy search: iterate over possible subgraphs of the required size and pick one with minimal overlaps and high fidelity. A more advanced approach could use graph algorithms or integer linear programming to optimally partition the graph for multiple circuits at once. For efficiency, we often use a greedy/bin-packing approach (see Scheduler below) which picks tiles for the largest jobs first (first-fit or best-fit decreasing).

  **Example:** If the hardware is IBM’s 27-qubit heavy-hex lattice, and we have circuits of 5 qubits each, the partitioner might carve out five 5-qubit tiles on the chip, leaving a qubit gap between each tile. *Figure 1* illustrates such a tiling on a heavy-hex topology: five disjoint clusters of 5 qubits (colored) with a couple of qubits left unused (white) as buffers between clusters. This tiling boosts parallel utilization 5-fold while mitigating interference by avoiding adjacent usage.

&#x20;*Figure 1: Partitioning a 27-qubit heavy-hex device into 5-qubit tiles for parallel circuits. Colored nodes are allocated to five separate circuit jobs, while white nodes remain idle as buffers to reduce crosstalk.*

* *APIs:* The topology module exposes methods like `Topology::find_tile(&self, tile_size, exclude_set) -> Option<Tile>`. The `exclude_set` parameter tracks qubits already used or reserved as buffers, ensuring new tiles don’t violate spacing rules. Each returned `Tile` contains the list of physical qubit indices. For example:

  ```rust
  struct Tile { pub qubits: Vec<u32> } 

  impl Topology {
      fn find_tile(&self, size: usize, exclude: &HashSet<u32>) -> Option<Tile> {
          // Search for `size` connected qubits not in `exclude`
          // (Could implement BFS/DFS to find connected components of required size)
      }
  }
  ```

  The partitioner will mark not only a tile’s qubits as used but also, if buffer=1, all their neighbors as excluded going forward.

### Module: `scheduler` – QVM Scheduler (Bin-Packing in Space/Time)

**Purpose:** Assign multiple input circuits to the available qubit tiles in a way that maximizes parallel execution and minimizes total runtime. It handles both spatial scheduling (which jobs run concurrently on different tiles) and temporal scheduling (sequencing batches of jobs when there aren’t enough qubits to run all at once).

* *Scheduling Algorithm:* We treat scheduling as a bin-packing/packing problem. Each job requires a bin of size equal to its qubit count (plus buffers). The hardware provides a fixed capacity (N qubits total), and we want to “pack” jobs into as few batches as possible. The scheduler proceeds as follows:

  1. **Sort Jobs:** Order incoming circuits by descending qubit count (or a heuristic like “circuit density” – e.g., CNOT count per qubit – if certain circuits are more error-sensitive). Larger circuits are placed first to ensure they get a fitting tile.
  2. **First Batch Allocation:** Starting from the largest job, request a tile from the `topology` module (`find_tile`) that fits the job’s qubit count. If found, assign that job to the tile and mark those qubits (and buffers) as occupied. Continue assigning the next jobs to remaining free regions until no more circuits fit without overlap. This forms one concurrent batch.
  3. **Subsequent Batches:** If unassigned jobs remain, free all qubits (reset the exclude set) and repeat the allocation for the next batch. This is effectively moving to the next time slice (all jobs in batch 1 will run, then batch 2, etc.). The scheduler thus produces groups of circuits that run in parallel, one group after another, such that all qubits in the device are utilized as much as possible each round.

  This approach is analogous to bin-packing where each batch is a “bin” (of size N qubits) filled with jobs. We aim to minimize the number of batches (bins), which corresponds to maximizing parallel utilization. By using known algorithms (first-fit, best-fit, etc.), the scheduler can achieve an efficient packing. For example, a scheduler might use a first-fit decreasing strategy: take each job in sorted order and put it into the first batch where it fits in an available tile, or if not, start a new batch.

* *Data Structures:* The scheduler could define a structure to hold scheduling results, e.g.:

  ```rust
  struct Schedule {
      assignments: Vec<Assignment>, // one per job
      batches: usize,               // number of sequential batches used
  }
  struct Assignment {
      job: QuantumCircuit,
      tile: Tile, 
      batch_index: usize,          // which time batch this job is scheduled in
  }
  ```

  The `assignments` list maps each job to a physical tile and a batch index. Batch 0 means it runs at time slot 0 (start of program), batch 1 means it runs after batch 0 completes, etc.

* *Temporal Scheduling & Timing:* All jobs in batch 0 will start at time t=0 concurrently, batch 1 jobs will execute after batch 0 completes, etc. In practice, the scheduler ensures no qubit is assigned to two jobs in the same batch. If a job spans batch boundaries (not typical unless a single job didn’t fit in one tile, which we do not allow), that would complicate timing – we assume each job is wholly in one batch/tile. The scheduler can compute an estimated **start time** for each batch if provided gate durations or simply treat each batch sequentially. In OpenQASM 3, we can express timing via `delay` or `barrier` instructions to separate batches, or rely on measurement + reset to mark the boundary.

* *Example:* Suppose we have 3 circuits needing 5, 5, and 3 qubits, and our hardware is 10 qubits. The scheduler might find it can place the two 5-qubit jobs on disjoint tiles (batch 0, using \~10 qubits total), and then the 3-qubit job runs in batch 1 (after one 5-qubit job finishes and frees its qubits). The result: 2 batches total. If instead the device had 5 qubits, all jobs would run sequentially (each in its own batch).

* *Crosstalk Awareness:* The scheduler works with the topology module to avoid placing simultaneous jobs too close. If buffer zones are enabled, the scheduler’s `find_tile` calls will naturally leave spacing. This aligns with research showing that scheduling multiple programs must account for hardware constraints like qubit quality and crosstalk to maintain fidelity. The scheduler could even incorporate advanced constraints (e.g., not co-locate two high-error-rate circuits in the same batch). These considerations ensure we **“do no harm”** to each circuit’s outcome while improving throughput.

### Module: `composer` – Composite Circuit Generator

**Purpose:** Build the final output circuit (OpenQASM 3 program) by merging all scheduled jobs according to the scheduler’s plan. This involves mapping each job’s logical qubits to physical qubits, inserting the appropriate timing/sequencing between batches, and adjusting classical measurement indices so results don’t conflict. The composer essentially *stitches* multiple circuits into one larger circuit.

* *Qubit Mapping:* Each `Assignment` from the scheduler gives a tile (physical qubit indices) for the job. The composer will replace the job’s qubit references (often 0..m for an m-qubit circuit) with the actual hardware qubit IDs from the `Tile`. For example, if job A’s circuit uses 3 logical qubits which the scheduler mapped to physical qubits \[5,7,8] on the chip, then every operation on “q\[0]” in A becomes an operation on physical qubit 5, “q\[1]” -> 7, etc. We maintain a mapping array for substitution.

* *Classical Bit Allocation:* To separate measurement outcomes, the composer assigns each job a distinct slice of the global classical bit register. For instance, if job A measures 3 bits and job B measures 2 bits, and both are in the composite program, A’s measurements could be allocated bits c\[0..2] and B’s c\[3..4]. The `Assignment` can carry a `cbit_offset` for each job. The composer then modifies each `Measure` operation: `measure q -> c[k]` in the original becomes `measure <phys_q> -> c[cbit_offset + k]` in the composite. This way, when the composite circuit finishes, the aggregated classical register contains each job’s results in separate regions. A mapping table of job -> (start\_bit, length) can be returned for easy demux of results.

* *Batch Separation:* Between batches, we ensure proper separation in the output QASM. After all jobs in batch 0 are listed, we insert a **barrier or timing delimiter** before batch 1’s operations begin. This guarantees that batch 0’s operations complete (and qubits are measured/reset) before batch 1 starts on those qubits. In OpenQASM, one can use `barrier all;` or specific qubits, or explicit `delay` commands with duration if known. For simplicity, a barrier is inserted on all qubits used up to that point, preventing reordering across the batch boundary. Additionally, after measuring qubits at the end of each batch, we include `reset` instructions on those qubits (if they will be reused in a later batch) to ensure they return to |0> state before the next use.

* *Output Generation:* Finally, the composite circuit is output as OpenQASM 3 text (or an AST structure that can be serialized to QASM). The program will declare a quantum register large enough for all qubits (e.g., `qubit[N] q;`) and a classical register for all bits (e.g., `bit[M] c;`). Then operations are listed in the scheduled order. Example (conceptual for two parallel jobs then one sequential):

  ```qasm
  qubit[10] q;
  bit[8] c;
  // Batch 0:
  h q[0]; cx q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];   // Job1 on tile {0,1}
  x q[5];  cx q[5], q[7]; measure q[5] -> c[2]; measure q[7] -> c[3];  // Job2 on tile {5,7}
  barrier q;  // ensure all batch0 ops done
  // Batch 1:
  reset q[0]; reset q[1]; reset q[5]; reset q[7];                     // free tile qubits from batch0
  h q[2];  z q[3]; cx q[2], q[3]; measure q[2] -> c[4]; measure q[3] -> c[5];  // Job3 on tile {2,3}
  ```

  In this example, two jobs ran in batch0 (using qubits 0-1 and 5-7) and one job in batch1 (using qubits 2-3, after they became free). Classical bits \[0..3] hold results of batch0’s jobs, and \[4..5] hold results of batch1’s job. The barrier and resets enforce the intended schedule. In OpenQASM 3, we could also use timing directives (like `delay`) if we had calibrated durations, but using barrier+reset is a simple, hardware-agnostic way to sequence batches. (OpenQASM 3’s design explicitly allows expressing timing for scheduling purposes, which future versions of the crate could leverage for more precise control.)

* *Verification:* The composer module includes checks to ensure that no classical or quantum resource conflicts exist in the final program: e.g., no qubit is operated by two jobs at the same time (ensured by scheduling), and classical bit ranges for different jobs don’t overlap. It can also optionally insert annotations or metadata (as comments) labeling which section corresponds to which original job for clarity.

### Module: `api` – High-Level API and Integration

**Purpose:** Expose a clean interface for users (both Rust and WebAssembly/JS) to utilize the scheduler. This module ties together the parser, scheduler, and composer into convenient functions, and ensures the crate can be compiled for different targets (native or WASM). It also contains integration points for CLI usage and defines feature flags (e.g., `wasm`) to toggle certain functionalities.

* *Public API:* The primary entry point might be a `Scheduler` struct in the crate’s root, configured with a target topology. For example:

  ```rust
  pub struct QvmScheduler {
      topology: Topology,
      buffer: usize,    // buffer zone setting (e.g., 0 or 1)
      tile_size: Option<usize>, // optional fixed tile size if desired
  }
  impl QvmScheduler {
      pub fn new(topology: Topology) -> Self { ... }
      pub fn buffer_zone(mut self, distance: usize) -> Self { ... }
      pub async fn schedule_circuits(&self, circuits: &[QuantumCircuit]) 
             -> Result<CompositeCircuit, ScheduleError> { ... }
  }
  ```

  The `schedule_circuits` method wraps all steps: parse (if needed), partition the graph into tiles, allocate/schedule jobs, and compose the final circuit. It returns a `CompositeCircuit` (which could be just a wrapper around the output OpenQASM text and perhaps the mapping metadata). This high-level API intentionally hides the internal details so that users can simply provide circuits and get a combined result.

* *Asynchronous Design:* The scheduling process can be compute-intensive for large numbers of circuits or qubits. To keep GUIs responsive (especially in a browser), `schedule_circuits` is defined as an `async fn` that yields to the event loop appropriately (e.g., using `tokio` or `async-std` on native, and `wasm_bindgen_futures` on WASM). Internally, long loops (like searching for tiles or sorting jobs) can be broken into chunks or use Rust async streams. By returning a `Future`, the crate allows the caller to await the result without blocking the thread. In a web context, one might `await scheduler.schedule_circuits(circuits)` from JavaScript.

* *WASM Support:* To use this crate in a browser, we ensure all data structures are serializable or easily convertible for JS. We use `#[cfg(target_arch = "wasm32")]` to conditionally include bindings. For example, we might provide:

  ```rust
  #[cfg(feature = "wasm")]
  #[wasm_bindgen]
  pub async fn schedule_qasm_files(qasm_list: Vec<JsValue>, topo: JsValue) -> Result<String, JsValue> {
      // Convert JS inputs (strings for QASM, or a JSON for topology) to Rust types
      let circuits: Vec<QuantumCircuit> = qasm_list.iter()
          .map(|jsv| parse_qasm(jsv.as_string().unwrap()))
          .collect();
      let topology: Topology = topo.into_serde().unwrap();
      let scheduler = QvmScheduler::new(topology);
      scheduler.schedule_circuits(&circuits)
          .await
          .map(|comp| comp.to_qasm_string())
          .map_err(|e| JsValue::from_str(&format!("Schedule error: {}", e)))
  }
  ```

  This function (exposed via `wasm-bindgen`) would allow a JS application to supply a list of QASM circuit strings and a topology (perhaps as a JSON) and get back the combined QASM string. All complex Rust structures (like `Topology` or `QuantumCircuit`) are either reconstructed from simple types or have `serde` support for easy JS interop. We also avoid using threads or any `std::fs` file operations so that the library runs under the `wasm32-unknown-unknown` target without issue. Memory management is handled by Rust’s allocator (which on WASM could be configured to use `wee_alloc` for smaller footprint if needed).

* *Feature Flags:* The crate can provide a `std` feature (enabled by default for normal Rust use) and a `no_std` mode (for truly embedded use or strict WASM without libstd). In `no_std`, we would rely on `alloc` and disable things like file parsing (user would pass circuits as IR or from their own parser). The `wasm` feature might turn on the `wasm-bindgen` facade shown above. We use conditional compilation to ensure these integrations don’t bloat the library for non-WASM users.

* *Error Handling:* The API returns rich errors (e.g., if a circuit can’t fit on the device at all, if parsing fails, etc.). We define a `ScheduleError` enum for common error cases (like `CircuitTooLarge{job: String}`, `TopologyDisconnected`, etc.), improving usability.

### Example Usage

#### 1. CLI Usage (Rust native)

For command-line use, a small binary can be provided (or users create their own using the library). For instance, an `examples/` directory might include a `schedule_jobs.rs` demonstrating how to use the crate:

```rust
use qvm_scheduler::{QvmScheduler, QuantumCircuit, Topology};

#[tokio::main]  // using Tokio for async main, for example
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load or define the hardware topology (example: 8-qubit ring from a JSON or so)
    let topo = Topology::from_file("topology_8qubit.json")?;  // user provides connectivity
    
    // Parse input QASM files into QuantumCircuit structs
    let circuits: Vec<QuantumCircuit> = std::env::args().skip(1)
        .map(|path| QuantumCircuit::from_qasm_file(path).expect("parse error"))
        .collect();
    if circuits.is_empty() {
        eprintln!("Please provide QASM files for scheduling.");
        return Ok(());
    }
    
    // Configure and run scheduler
    let scheduler = QvmScheduler::new(topo).buffer_zone(1);  // use buffer=1 for isolation
    let composite = scheduler.schedule_circuits(&circuits).await?;
    
    // Output the combined OpenQASM 3 program
    println!("// Composite schedule for {} jobs", circuits.len());
    println!("{}", composite.to_qasm_string());
    Ok(())
}
```

If the user runs `schedule_jobs circuit1.qasm circuit2.qasm`, the program will print an OpenQASM 3 program that implements both circuits either in parallel or sequentially as allowed. The output can be saved or sent to a quantum service. Each circuit’s results can be extracted from the appropriate classical bit indices (which the program could also print out or log).

#### 2. WASM/Web Usage

In a web application, one could compile the crate to WebAssembly (using `wasm-pack`) and call the exposed JS functions. For example, in JavaScript:

```js
import init, { schedule_qasm_files } from 'qvm_scheduler_wasm.js';

async function scheduleCircuits(qasmList, topo) {
    await init();  // initialize the WASM module
    try {
        const resultQASM = await schedule_qasm_files(qasmList, topo);
        console.log("Composite QASM:", resultQASM);
    } catch(e) {
        console.error("Scheduling failed:", e);
    }
}
```

Here, `qasmList` would be an array of QASM program strings, and `topo` could be a JSON string or object describing the hardware graph (the Rust binding will `into_serde` it into a `Topology`). The promise returned by `schedule_qasm_files` resolves to the combined QASM string. This could then be sent to a backend via AJAX, or even visualized in the browser (e.g., using a circuit diagram renderer to show the composite circuit schedule).

Because the crate is async and WASM-friendly, it can integrate into a web UI where users upload circuits and see a scheduled result, without freezing the page. The overhead for moderately sized circuits is low, and Rust/WASM provides near-native performance for the scheduling algorithms.

### Testing and Benchmarking

We include a test suite covering individual modules and end-to-end functionality:

* **Unit Tests:** e.g., `topology::tests` might create a small graph and attempt known partitions (ensuring `find_tile` returns expected tiles), `scheduler::tests` could schedule a set of toy circuits on a mock topology and verify that no qubit is double-booked and batch count is minimal, and `composer::tests` can check that two simple circuits produce the correct merged QASM (comparing against an expected string or parsing it back to an IR for structural verification).

* **Test Vectors:** We create some predefined scenarios such as: (1) **Linear Topology** – a line of qubits and two small chain circuits that should either run parallel if far apart or sequential if they would overlap; (2) **Grid Topology** – and multiple small circuits to test 2D tiling; (3) Circuits that exactly fill the device vs. circuits that leave gaps (to test buffer logic). We also test edge cases like a circuit that requires more qubits than available (expect an error), or an empty input list (expect no output or a trivial QASM program).

* **Benchmarks:** Basic performance benchmarks (using Rust’s `bench` or an external crate like Criterion) measure how the scheduler scales. For example, we measure scheduling 100 random 2-qubit circuits on a 20-qubit device, or a few larger 10-qubit circuits on a 20-qubit device, etc. The expectation is that the greedy bin-packing scheduler runs in time roughly O(J log J) for J jobs (sorting) plus the tile search time which for each job is at most O(Q) or O(Q^2) for scanning subgraphs (with Q qubits). This is quite fast for typical NISQ sizes (tens to hundreds of qubits). We also test the overhead of the crate’s async/WASM layers to ensure even in a browser the scheduling can be done on the order of milliseconds for reasonable inputs.

In summary, this Rust crate provides a **modular, efficient, and portable** solution for multi-program quantum circuit scheduling. By partitioning the device and smartly packing circuits, it boosts throughput while preserving result integrity. The design’s separation of concerns (parsing, tiling, scheduling, composing) makes it easy to maintain and extend – for example, one could plug in a more advanced partition algorithm or incorporate error rates into the scheduling heuristic without altering the other components. With WebAssembly support, the scheduler can even run in-browser for interactive visualization or education tools, demonstrating how multiple quantum programs can coexist on one quantum machine. The final output, in OpenQASM 3, remains vendor-neutral and ready to be executed on any compatible backend, truly delivering a **quantum virtual machine** abstraction above physical hardware.&#x20;

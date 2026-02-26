## 2025-02-23 - Vectorization of Sliding Window Statistics
**Learning:** Replacing per-object `deque` history buffers with a single contiguous 2D NumPy array (bands x history) allows for vectorized statistics calculation (`np.median`, `np.var`, `np.dot` for trend), resulting in massive speedups (~50x) for the 64-band RF monitor.
**Action:** When optimizing real-time signal processing with many channels, always prefer a "Structure of Arrays" (SoA) approach with pre-allocated NumPy buffers over "Array of Structures" (AoS) with dynamic lists/deques.

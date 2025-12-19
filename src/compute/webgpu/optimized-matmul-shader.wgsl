// Optimized Matrix Multiplication Shader
// Based on: https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel
// Target: >1 TFLOPS (vs naive ~1.64 GFLOPS = 600x faster)

// Workgroup tiling parameters
const TILE_SIZE: u32 = 16u;
const WORKGROUP_SIZE: u32 = 16u;

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // M, N, K

// Shared memory tiles for cache locality
var<workgroup> tileA: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tileB: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;

    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Early exit for out-of-bounds threads
    if (row >= M || col >= N) {
        return;
    }

    // Accumulator for dot product (register blocking)
    var acc: f32 = 0.0;

    // Tile over K dimension
    let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let tileK = t * TILE_SIZE;

        // Collaborative loading of tile A into shared memory
        let aRow = row;
        let aCol = tileK + local_col;
        if (aRow < M && aCol < K) {
            tileA[local_row][local_col] = matrixA[aRow * K + aCol];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        // Collaborative loading of tile B into shared memory
        let bRow = tileK + local_row;
        let bCol = col;
        if (bRow < K && bCol < N) {
            tileB[local_row][local_col] = matrixB[bRow * N + bCol];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        // Synchronize to ensure tile is loaded
        workgroupBarrier();

        // Compute partial dot product using shared memory
        // This is the hot loop - modern GPUs execute this very fast
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tileA[local_row][k] * tileB[k][local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    result[row * N + col] = acc;
}

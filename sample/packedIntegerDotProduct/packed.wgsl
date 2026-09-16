requires packed_4x8_integer_dot_product;

struct Case { lhs: u32, rhs: u32 }

@group(0) @binding(0) var<storage, read> input: array<Case>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

const kWorkgroupSize: u32 = 64; // Same as in main.ts

@compute @workgroup_size(kWorkgroupSize)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&output)) {
    return;
  }

  output[id.x] = dot4I8Packed(input[id.x].lhs, input[id.x].rhs);
}

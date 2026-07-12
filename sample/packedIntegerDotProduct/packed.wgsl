requires packed_4x8_integer_dot_product;

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&output)) {
    return;
  }
  output[id.x] = dot4I8Packed(lhs[id.x], rhs[id.x]);
}

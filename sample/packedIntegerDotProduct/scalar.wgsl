@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;

fn unpackI8(value: u32, component: u32) -> i32 {
  let byte = i32((value >> (component * 8u)) & 0xffu);
  return select(byte, byte - 256, byte >= 128);
}

fn dot4I8Scalar(a: u32, b: u32) -> i32 {
  var result = 0;
  for (var component = 0u; component < 4u; component += 1u) {
    result += unpackI8(a, component) * unpackI8(b, component);
  }
  return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&output)) {
    return;
  }
  output[id.x] = dot4I8Scalar(lhs[id.x], rhs[id.x]);
}

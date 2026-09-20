requires packed_4x8_integer_dot_product;

override packed: bool = true;
const imageWidth = 512u;
const patchWidth = 32u;
const gridWidth = 121u; // Valid 32-pixel windows on a 4-pixel grid.

@group(0) @binding(0) var<storage, read> image: array<u32>;
@group(0) @binding(1) var<storage, read> patchWords: array<u32>;
@group(0) @binding(2) var<uniform> templateEnergy: i32;
@group(0) @binding(3) var<storage, read> windowEnergies: array<i32>;
@group(0) @binding(4) var<storage, read_write> errors: array<i32>;

fn unpack(word: u32) -> vec4i {
  // Shift each byte into the sign position, then shift arithmetically back.
  return bitcast<vec4i>(vec4u(word) << vec4u(24, 16, 8, 0)) >> vec4u(24);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let candidate = id.x;
  if (candidate >= gridWidth * gridWidth) { return; }
  let origin = vec2u(candidate % gridWidth, candidate / gridWidth) * 4;
  // The fixed image's window energies are reusable across template searches.
  var error = select(0, windowEnergies[candidate] + templateEnergy, packed);
  for (var row = 0u; row < patchWidth; row++) {
    for (var col = 0u; col < patchWidth / 4; col++) {
      // All horizontal origins are aligned: no cross-word byte gathering.
      let p = image[((origin.y + row) * imageWidth + origin.x) / 4 + col];
      let t = patchWords[row * (patchWidth / 4) + col];
      if (packed) {
        // Sum (p-t)^2 = precomputed P·P + T·T - 2 sum P·T.
        error -= 2 * dot4I8Packed(p, t);
      } else {
        // Natural scalar baseline, with the same packed input and outputs.
        let d = unpack(p) - unpack(t);
        error += d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w;
      }
    }
  }
  errors[candidate] = error;
}

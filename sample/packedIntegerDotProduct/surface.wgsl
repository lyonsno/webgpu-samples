struct View {
  matrix: mat4x4f,
  selected: vec2u,
  best: vec2u,
}
@group(0) @binding(0) var<storage, read> errors: array<i32>;
@group(0) @binding(1) var<uniform> view: View;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) error: f32,
  @location(1) grid: vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertex: u32,
              @builtin(instance_index) cell: u32) -> Vertex {
  let corners = array(vec2u(0,0), vec2u(1,0), vec2u(0,1),
                      vec2u(0,1), vec2u(1,0), vec2u(1,1));
  let xy = vec2u(cell % 120, cell / 120) + corners[vertex];
  // Fixed scale: root-mean-square grayscale error / 255, always in [0,1].
  let error = sqrt(f32(errors[xy.y * 121 + xy.x]) / 1024.0) / 255.0;
  let position = vec2f(xy) / 60.0 - 1.0;
  var out: Vertex;
  out.position = view.matrix * vec4f(position.x, error * 1.5, position.y, 1);
  out.error = error;
  out.grid = vec2f(xy);
  return out;
}

struct Fragment {
  @location(0) color: vec4f,
  @location(1) candidate: u32,
}

@fragment
fn fragmentMain(in: Vertex) -> Fragment {
  let low = vec3f(0.04, 0.65, 0.75);
  let high = vec3f(1.0, 0.45, 0.12);
  var color = mix(low, high, in.error);
  // Contours every 16 grayscale levels make shallow basins visible.
  let bands = in.error * 255.0 / 16.0;
  let line = smoothstep(0.0, max(fwidth(bands), 0.01), abs(fract(bands + 0.5) - 0.5));
  color *= mix(0.65, 1.0, line);
  if (distance(in.grid, vec2f(view.selected)) < 0.8) { color = vec3f(1); }
  if (distance(in.grid, vec2f(view.best)) < 1.4) { color = vec3f(1, 0.25, 0.78); }
  // The depth-tested visible surface supplies the nearest search location.
  // Zero is reserved for background, where there is nothing to pick.
  let xy = vec2u(round(in.grid));
  return Fragment(vec4f(color, 1), 1 + xy.y * 121 + xy.x);
}

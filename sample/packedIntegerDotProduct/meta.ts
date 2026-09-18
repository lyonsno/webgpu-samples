export default {
  name: 'Packed Integer Dot Product',
  description:
    'Explore how packed integer dot products combine inputs and weights in quantized neural networks. Change a value and follow its contribution through the geometry, packed bytes, and WGSL operation.',
  filename: __DIRNAME__,
  sources: [
    { path: 'main.ts' },
    { path: 'packed.wgsl' },
    { path: 'visualization.ts' },
    { path: 'index.html' },
  ],
};

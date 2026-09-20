export default {
  name: 'Packed Integer Dot Product',
  description:
    'Explore how packed integer dot products combine inputs and weights in quantized neural networks. Change a value and follow it through signed bytes, packed words, and the WGSL operation.',
  filename: __DIRNAME__,
  sources: [
    { path: 'main.ts' },
    { path: 'packed.wgsl' },
    { path: 'visualization.ts' },
    { path: 'index.html' },
  ],
};

export default {
  name: 'Packed Integer Dot Product',
  description:
    'Explore four-component integer vectors as two coordinate pairs, inspect their packed signed bytes, and compute their dot product with dot4I8Packed.',
  filename: __DIRNAME__,
  sources: [
    { path: 'main.ts' },
    { path: 'packed.wgsl' },
    { path: 'visualization.ts' },
    { path: 'index.html' },
  ],
};

export default {
  name: 'Packed Integer Dot Product',
  description:
    'Compares signed 8-bit dot products using the packed WGSL instruction and an equivalent scalar route, with exact result validation.',
  filename: __DIRNAME__,
  sources: [
    { path: 'main.ts' },
    { path: 'reference.ts' },
    { path: 'packed.wgsl' },
    { path: 'scalar.wgsl' },
  ],
};

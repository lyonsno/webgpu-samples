export default {
  name: 'Packed Integer Dot Product',
  description:
    'Find an image patch with packed integer dot products. Explore the GPU error surface and compare packed dots with scalar squared differences.',
  filename: __DIRNAME__,
  sources: [
    { path: 'main.ts' },
    { path: 'packed.wgsl' },
    { path: 'surface.wgsl' },
    { path: 'data.ts' },
    { path: '../timestampQuery/TimestampQueryManager.ts' },
    { path: 'index.html' },
  ],
};

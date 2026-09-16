import { GUI } from 'dat.gui';
import packedWGSL from './packed.wgsl';
import { quitIfWebGPUNotAvailableOrMissingFeatures } from '../util';

const result = document.querySelector('#result') as HTMLElement;
if (
  !navigator.gpu?.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')
) {
  result.textContent =
    "This sample requires the WGSL language feature 'packed_4x8_integer_dot_product'.";
} else {
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });
  const device = await adapter?.requestDevice();
  quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);

  const lhs = [
    [1, -2, 3, -4],
    [-128, -128, -128, -128],
    [127, 127, 127, 127],
    [1, 2, 3, 4],
  ];
  const rhs = [
    [-5, 6, -7, 8],
    [-128, -128, -128, -128],
    [127, 127, 127, 127],
    [4, 3, 2, 1],
  ];

  function createInputBuffer(vectors: number[][]) {
    // Pack four signed 8-bit components into each u32, low byte first.
    // Masking preserves the two's-complement representation of negative values.
    const packed = new Uint32Array(
      vectors.map(
        ([x, y, z, w]) =>
          (x & 0xff) |
          ((y & 0xff) << 8) |
          ((z & 0xff) << 16) |
          ((w & 0xff) << 24)
      )
    );
    const buffer = device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint32Array(buffer.getMappedRange()).set(packed);
    buffer.unmap();
    return buffer;
  }

  const outputSize = lhs.length * Int32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: packedWGSL }) },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: createInputBuffer(lhs) } },
      { binding: 1, resource: { buffer: createInputBuffer(rhs) } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(lhs.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const results = new Int32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();

  const settings = { example: 0 };
  function showResult() {
    const i = settings.example;
    result.textContent = `a = [${lhs[i].join(', ')}]
b = [${rhs[i].join(', ')}]
dot4I8Packed(a, b) = ${results[i]}`;
  }
  const gui = new GUI();
  gui
    .add(settings, 'example', {
      'Mixed signs': 0,
      'Minimum signed bytes': 1,
      'Maximum signed bytes': 2,
      'Positive components': 3,
    })
    .onChange(showResult);
  showResult();
}

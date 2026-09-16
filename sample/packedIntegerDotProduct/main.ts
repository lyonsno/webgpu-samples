import packedWGSL from './packed.wgsl';
import { quitIfWebGPUNotAvailableOrMissingFeatures } from '../util';

const kSampleCases = [
  { lhs: [1, -2, 3, -4], rhs: [-5, 6, -7, 8] },
  { lhs: [-128, -128, -128, -128], rhs: [-128, -128, -128, -128] },
  { lhs: [127, 127, 127, 127], rhs: [127, 127, 127, 127] },
  { lhs: [1, 2, 3, 4], rhs: [4, 3, 2, 1] },
] as const;

const kWorkgroupSize = 64; // Same as in packed.wgsl

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

  function createInputBuffer(
    vectors: ReadonlyArray<readonly [number, number, number, number]>
  ) {
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

  const outputSize = kSampleCases.length * Int32Array.BYTES_PER_ELEMENT;
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
  const inputBuffer = createInputBuffer(
    kSampleCases.flatMap((c) => [c.lhs, c.rhs])
  );
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(kSampleCases.length / kWorkgroupSize));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const results = new Int32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();

  for (const [i, sample] of kSampleCases.entries()) {
    const expected =
      sample.lhs[0] * sample.rhs[0] +
      sample.lhs[1] * sample.rhs[1] +
      sample.lhs[2] * sample.rhs[2] +
      sample.lhs[3] * sample.rhs[3];
    const lhs = sample.lhs.map((x) => x.toString().padStart(4)).join(', ');
    const rhs = sample.rhs.map((x) => x.toString().padStart(4)).join(', ');
    const out = results[i].toString().padStart(6);
    const exp = expected.toString().padStart(6);
    result.textContent += `\ndot4I8Packed of [${lhs}] by [${rhs}] gave ${out} (expecting ${exp})`;
  }
}

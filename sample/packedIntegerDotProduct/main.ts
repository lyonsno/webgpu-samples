import { GUI } from 'dat.gui';
import packedWGSL from './packed.wgsl';
import { quitIfWebGPUNotAvailableOrMissingFeatures } from '../util';
import { createVisualization, type Side, type Vector } from './visualization';

// Pack four signed 8-bit components into a u32, low byte first.
function pack4xI8([x, y, z, w]: Vector): number {
  // `&` operator applies sign extension to i32 before operating.
  // `>>> 0` converts the final i32 to u32.
  /*prettier-ignore*/
  return ((x & 0xff) |
          ((y & 0xff) << 8) |
          ((z & 0xff) << 16) |
          ((w & 0xff) << 24)) >>> 0;
}

const outputElement = document.querySelector('#output') as HTMLElement;
if (
  !navigator.gpu?.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')
) {
  outputElement.textContent =
    "This sample requires the WGSL language feature 'packed_4x8_integer_dot_product'.";
} else {
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });
  const device = await adapter?.requestDevice();
  quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);

  const kInputSize = 2 * Uint32Array.BYTES_PER_ELEMENT;
  const inputBuffer = device.createBuffer({
    size: kInputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  const kOutputSize = Int32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({
    size: kOutputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: kOutputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: packedWGSL }) },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  async function computeDot(lhs: Vector, rhs: Vector) {
    device.queue.writeBuffer(
      inputBuffer,
      0,
      new Uint32Array([lhs, rhs].map(pack4xI8))
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, kOutputSize);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Int32Array(readbackBuffer.getMappedRange())[0];
    readbackBuffer.unmap();
    return result;
  }

  const settings = {
    lhs0: 1,
    lhs1: -2,
    lhs2: 3,
    lhs3: -4,
    rhs0: -5,
    rhs1: 6,
    rhs2: -7,
    rhs3: 8,
  };
  const gui = new GUI({ autoPlace: false, width: 246 });
  document.querySelector('#controls')!.append(gui.domElement);
  const view = createVisualization((side, component, value) => {
    const key = `${side}${component}` as keyof typeof settings;
    settings[key] = value;
    gui.updateDisplay();
    updateResult();
  });
  for (const side of ['lhs', 'rhs'] as const) {
    for (let component = 0; component < 4; component++) {
      const key = `${side}${component}` as keyof typeof settings;
      const control = gui.add(settings, key, -128, 127, 1);
      control.domElement
        .querySelector('input')!
        .setAttribute('aria-label', key);
      control.onChange(() => {
        view.select(side, component);
        updateResult();
      });
    }
  }
  const vector = (side: Side): Vector => [
    settings[`${side}0`],
    settings[`${side}1`],
    settings[`${side}2`],
    settings[`${side}3`],
  ];
  let revision = 0;
  let computing = false;
  async function updateResult() {
    revision++;
    const lhs = vector('lhs');
    const rhs = vector('rhs');
    view.update(lhs, rhs, pack4xI8(lhs), pack4xI8(rhs));
    outputElement.textContent = '…';
    if (computing) return;
    computing = true;
    try {
      // A sweep may change the inputs during readback. Compute the latest pair
      // next, and never label an older GPU result as belonging to the new view.
      let submittedRevision;
      do {
        submittedRevision = revision;
        const lhs = vector('lhs');
        const rhs = vector('rhs');
        const result = await computeDot(lhs, rhs);
        if (submittedRevision === revision) {
          outputElement.textContent = String(result);
        }
      } while (submittedRevision !== revision);
    } catch (error) {
      outputElement.textContent = `GPU calculation failed: ${error}`;
    } finally {
      computing = false;
    }
  }
  (document.querySelector('#explorer') as HTMLElement).hidden = false;
  updateResult();
}

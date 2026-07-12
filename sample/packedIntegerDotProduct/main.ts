import packedWGSL from './packed.wgsl';
import scalarWGSL from './scalar.wgsl';
import {
  makeInputVectors,
  packedDotLanguageFeature,
  RequestedRoute,
  selectRoute,
  validateRun,
} from './reference';
import { quitIfWebGPUNotAvailableOrMissingFeatures } from '../util';

const elementCount = 64 * 1024;
const workgroupSize = 64;
const previewCount = 192;
const { lhsPacked, rhsPacked, expected } = makeInputVectors(elementCount);

const adapter = await navigator.gpu?.requestAdapter({
  featureLevel: 'compatibility',
});
const device = await adapter?.requestDevice();
quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);

const packedLanguageFeatureSupported = navigator.gpu.wgslLanguageFeatures.has(
  packedDotLanguageFeature
);
const languageFeatures = new Set<string>(
  packedLanguageFeatureSupported ? [packedDotLanguageFeature] : []
);

function createInputBuffer(data: Uint32Array) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength
  );
  return buffer;
}

const lhsBuffer = createInputBuffer(lhsPacked);
const rhsBuffer = createInputBuffer(rhsPacked);
const outputBuffer = device.createBuffer({
  size: expected.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const readbackBuffer = device.createBuffer({
  size: expected.byteLength,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

async function createPipeline(code: string) {
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code }),
    },
  });
}

const scalarPipeline = await createPipeline(scalarWGSL);
const packedPipeline = packedLanguageFeatureSupported
  ? await createPipeline(packedWGSL)
  : null;

function createBindGroup(pipeline: GPUComputePipeline) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: lhsBuffer } },
      { binding: 1, resource: { buffer: rhsBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });
}

const scalarBindGroup = createBindGroup(scalarPipeline);
const packedBindGroup = packedPipeline ? createBindGroup(packedPipeline) : null;

const routeValue = document.querySelector('[data-route-value]') as HTMLElement;
const supportValue = document.querySelector(
  '[data-support-value]'
) as HTMLElement;
const durationValue = document.querySelector(
  '[data-duration-value]'
) as HTMLElement;
const validationValue = document.querySelector(
  '[data-validation-value]'
) as HTMLElement;
const fallbackValue = document.querySelector(
  '[data-fallback-value]'
) as HTMLElement;
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = canvas.getContext('2d') as CanvasRenderingContext2D;
const buttons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('[data-route]')
);

supportValue.textContent = packedLanguageFeatureSupported
  ? 'Available'
  : 'Unavailable';

function drawResults(results: Int32Array) {
  const scale = window.devicePixelRatio;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, width, height);

  const baseline = height / 2;
  const maxMagnitude = Math.max(
    1,
    ...results.slice(0, previewCount).map((value) => Math.abs(value))
  );
  const barWidth = width / previewCount;

  context.fillStyle = '#d8dde5';
  context.fillRect(0, baseline, width, 1);
  for (let index = 0; index < previewCount; ++index) {
    const value = results[index];
    const barHeight = (Math.abs(value) / maxMagnitude) * (baseline - 12);
    context.fillStyle = value >= 0 ? '#00a87a' : '#e34b7a';
    context.fillRect(
      index * barWidth,
      value >= 0 ? baseline - barHeight : baseline + 1,
      Math.max(1, barWidth - 1),
      barHeight
    );
  }
}

let runSerial = 0;

async function run(requestedRoute: RequestedRoute) {
  const serial = ++runSerial;
  buttons.forEach((button) => {
    button.disabled = true;
    button.dataset.selected = String(button.dataset.route === requestedRoute);
  });
  validationValue.textContent = 'Running';
  fallbackValue.textContent = '';

  const selection = selectRoute(requestedRoute, languageFeatures);
  const pipeline =
    selection.effectiveRoute === 'packed' ? packedPipeline : scalarPipeline;
  const bindGroup =
    selection.effectiveRoute === 'packed' ? packedBindGroup : scalarBindGroup;
  if (!pipeline || !bindGroup) {
    throw new Error(
      `effective route '${selection.effectiveRoute}' is unavailable`
    );
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(elementCount / workgroupSize));
  pass.end();
  encoder.copyBufferToBuffer(
    outputBuffer,
    0,
    readbackBuffer,
    0,
    expected.byteLength
  );

  const start = performance.now();
  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const duration = performance.now() - start;
  const actual = new Int32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();

  validateRun({ ...selection, expected, actual });
  if (serial !== runSerial) {
    return;
  }

  routeValue.textContent = selection.effectiveRoute;
  durationValue.textContent = `${duration.toFixed(2)} ms`;
  validationValue.textContent = `${actual.length.toLocaleString()} exact`;
  fallbackValue.textContent = selection.fallbackReason ?? '';
  drawResults(actual);
  buttons.forEach((button) => {
    button.disabled = false;
  });
}

buttons.forEach((button) => {
  button.addEventListener('click', () => {
    run(button.dataset.route as RequestedRoute);
  });
});

window.addEventListener('resize', () => drawResults(expected));
await run('auto');

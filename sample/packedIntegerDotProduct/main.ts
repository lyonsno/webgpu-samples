import { GUI } from 'dat.gui';
import { mat4 } from 'wgpu-matrix';
import packedWGSL from './packed.wgsl';
import surfaceWGSL from './surface.wgsl';
import {
  packPixels,
  unpack,
  makeTemplate,
  makeWindowEnergies,
  imageSize,
  patchSize,
  stride,
  gridSize,
} from './data';
import TimestampQueryManager from '../timestampQuery/TimestampQueryManager';
import { quitIfWebGPUNotAvailableOrMissingFeatures } from '../util';

const result = document.querySelector('#result')!;
if (
  !navigator.gpu?.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')
) {
  throw new Error("This sample requires 'packed_4x8_integer_dot_product'.");
}
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter?.requestDevice({
  requiredFeatures: adapter.features.has('timestamp-query')
    ? ['timestamp-query']
    : [],
});
quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);

// Prepare the source once. Both kernels consume the same packed grayscale data.
const source = document.querySelector<HTMLCanvasElement>('#source')!;
const ctx = source.getContext('2d')!;
const response = await fetch('../../assets/img/Di-3d.png');
const bitmap = await createImageBitmap(await response.blob());
ctx.drawImage(bitmap, 0, 0, imageSize, imageSize);
bitmap.close();
const rgba = ctx.getImageData(0, 0, imageSize, imageSize);
const gray = new Uint8Array(imageSize ** 2);
for (let i = 0; i < gray.length; i++) {
  gray[i] = Math.round(
    0.299 * rgba.data[i * 4] +
      0.587 * rgba.data[i * 4 + 1] +
      0.114 * rgba.data[i * 4 + 2]
  );
  rgba.data.fill(gray[i], i * 4, i * 4 + 3);
}
const image = packPixels(gray);
const buffer = (size: number, usage: GPUBufferUsageFlags) =>
  device.createBuffer({ size, usage });
const imageBuffer = buffer(
  image.byteLength,
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
);
const templateBuffer = buffer(
  patchSize ** 2,
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
);
const energyBuffer = buffer(
  16,
  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
);
const errors = buffer(
  gridSize ** 2 * 4,
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
);
const windowEnergy = makeWindowEnergies(image);
const windowEnergyBuffer = buffer(
  windowEnergy.byteLength,
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
);
const readback = buffer(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
device.queue.writeBuffer(imageBuffer, 0, image);
device.queue.writeBuffer(
  windowEnergyBuffer,
  0,
  windowEnergy.buffer as ArrayBuffer
);

const module = device.createShaderModule({ code: packedWGSL });
const pipelines = await Promise.all(
  [true, false].map((packed) =>
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, constants: { packed: Number(packed) } },
    })
  )
);
const groups = pipelines.map((pipeline) =>
  device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      imageBuffer,
      templateBuffer,
      energyBuffer,
      windowEnergyBuffer,
      errors,
    ].map((buffer, binding) => ({ binding, resource: { buffer } })),
  })
);

const canvas = document.querySelector<HTMLCanvasElement>('#surface')!;
const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });
const viewBuffer = buffer(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
const surfaceModule = device.createShaderModule({ code: surfaceWGSL });
const renderPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module: surfaceModule, entryPoint: 'vertexMain' },
  fragment: {
    module: surfaceModule,
    entryPoint: 'fragmentMain',
    targets: [{ format }],
  },
  primitive: { topology: 'triangle-list' },
  depthStencil: {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less',
  },
});
const renderGroup = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [errors, viewBuffer].map((buffer, binding) => ({
    binding,
    resource: { buffer },
  })),
});
let depth: GPUTexture;
const resize = () => {
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
  canvas.height = Math.max(
    1,
    Math.round(canvas.clientHeight * devicePixelRatio)
  );
  depth?.destroy();
  depth = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const aspect = canvas.width / canvas.height;
  const matrix = mat4.multiply(
    mat4.ortho(-1.65 * aspect, 1.65 * aspect, -1.65, 1.65, 0.1, 10),
    mat4.lookAt([2.5, 2.8, 3.2], [0, 0.3, 0], [0, 1, 0])
  );
  device.queue.writeBuffer(viewBuffer, 0, matrix.buffer as ArrayBuffer);
};
resize();

const settings = {
  packed: true,
  x: 320,
  y: 256,
  useAsTemplate: () => {
    templateX = settings.x;
    templateY = settings.y;
    update();
  },
};
let templateX = 256,
  templateY = 256;
const gui = new GUI({ autoPlace: false });
document.querySelector('#controls')!.append(gui.domElement);
gui.add(settings, 'packed').name('Use packed dots').onChange(update);
for (const axis of ['x', 'y'] as const) {
  const control = gui
    .add(settings, axis, 0, imageSize - patchSize, stride)
    .name(`Candidate ${axis}`)
    .onChange(update);
  control.domElement
    .querySelector('input')!
    .setAttribute('aria-label', `Candidate ${axis}`);
}
gui.add(settings, 'useAsTemplate').name('Use candidate as template');
source.onclick = (event) => {
  const rect = source.getBoundingClientRect();
  const snap = (v: number) =>
    Math.max(
      0,
      Math.min(imageSize - patchSize, Math.round(v / stride) * stride)
    );
  settings.x = snap(
    ((event.clientX - rect.left) / rect.width) * imageSize - patchSize / 2
  );
  settings.y = snap(
    ((event.clientY - rect.top) / rect.height) * imageSize - patchSize / 2
  );
  gui.updateDisplay();
  update();
};

const times: (number | undefined)[] = [];
const timers = pipelines.map(
  (_, mode) =>
    new TimestampQueryManager(device, (ns) => {
      times[mode] = ns / 1e6;
      document.querySelector(
        '#timing'
      )!.textContent = `Last GPU compute pass — packed: ${
        times[0]?.toFixed(3) ?? '—'
      } ms; scalar: ${
        times[1]?.toFixed(3) ?? '—'
      } ms. Excludes image preparation, rendering, and readback; toggle to compare (timings vary).`;
    })
);
if (!timers[0].timestampSupported)
  document.querySelector('#timing')!.textContent =
    'GPU timestamps unavailable on this device.';

// Serialize readback; inputs changed in flight are computed next.
let revision = 0,
  running = false;
async function update() {
  revision++;
  if (running) return;
  running = true;
  try {
    let submitted;
    do {
      submitted = revision;
      const { x, y, packed } = settings;
      const mode = packed ? 0 : 1;
      const template = makeTemplate(image, templateX, templateY);
      device.queue.writeBuffer(templateBuffer, 0, template.words);
      device.queue.writeBuffer(
        energyBuffer,
        0,
        new Int32Array([template.energy])
      );
      device.queue.writeBuffer(
        viewBuffer,
        64,
        new Uint32Array([x / stride, y / stride])
      );
      ctx.putImageData(rgba, 0, 0);
      const preview = document.querySelector<HTMLCanvasElement>('#template')!;
      preview
        .getContext('2d')!
        .drawImage(
          source,
          templateX,
          templateY,
          patchSize,
          patchSize,
          0,
          0,
          patchSize,
          patchSize
        );
      ctx.strokeStyle = '#ff40c8';
      ctx.lineWidth = 3;
      ctx.strokeRect(
        templateX - 1,
        templateY - 1,
        patchSize + 2,
        patchSize + 2
      );
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 2, y + 2, patchSize - 4, patchSize - 4);
      document.querySelector(
        '#selection'
      )!.textContent = `at (${templateX}, ${templateY}); candidate at (${x}, ${y}).`;
      const p = image[(y * imageSize + x) / 4],
        t = template.words[0];
      const lanes = (word: number) => [0, 1, 2, 3].map((i) => unpack(word, i));
      const a = lanes(p),
        b = lanes(t);
      const dot = (a: number[], b: number[]) =>
        a.reduce((sum, v, i) => sum + v * b[i], 0);
      const hex = (v: number) =>
        `0x${v.toString(16).padStart(8, '0').toUpperCase()}`;
      document.querySelector(
        '#group'
      )!.textContent = `First four pixels (signed = gray − 128):\nCandidate [${a.join(
        ', '
      )}] → ${hex(p)}\nTemplate  [${b.join(', ')}] → ${hex(
        t
      )}\ndot4I8Packed(P,T) = ${dot(a, b)}\nGroup error: ${dot(a, a)} + ${dot(
        b,
        b
      )} − 2 × (${dot(a, b)}) = ${
        dot(a, a) + dot(b, b) - 2 * dot(a, b)
      }\nSum all 256 groups for the patch error below.`;
      result.textContent = 'Computing patch errors…';
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(timers[mode].addTimestampWrite({}));
      pass.setPipeline(pipelines[mode]);
      pass.setBindGroup(0, groups[mode]);
      pass.dispatchWorkgroups(Math.ceil(gridSize ** 2 / 64));
      pass.end();
      timers[mode].resolve(encoder);
      const render = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: [0.035, 0.045, 0.065, 1],
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: depth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      render.setPipeline(renderPipeline);
      render.setBindGroup(0, renderGroup);
      render.draw(6, (gridSize - 1) ** 2);
      render.end();
      encoder.copyBufferToBuffer(
        errors,
        ((y / stride) * gridSize + x / stride) * 4,
        readback,
        0,
        4
      );
      device.queue.submit([encoder.finish()]);
      timers[mode].tryInitiateTimestampDownload();
      await readback.mapAsync(GPUMapMode.READ);
      const error = new Int32Array(readback.getMappedRange())[0];
      readback.unmap();
      if (submitted === revision)
        result.textContent = `GPU ${
          packed ? 'packed' : 'scalar'
        } patch error: ${error.toLocaleString()} (RMS ${Math.sqrt(
          error / patchSize ** 2
        ).toFixed(2)} grayscale levels). Zero is an exact match.`;
    } while (submitted !== revision);
  } finally {
    running = false;
  }
}
window.addEventListener('resize', () => {
  resize();
  update();
});
update();

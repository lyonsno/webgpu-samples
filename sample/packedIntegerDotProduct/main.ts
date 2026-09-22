import { mat4, vec3 } from 'wgpu-matrix';
import packedWGSL from './packed.wgsl';
import surfaceWGSL from './surface.wgsl';
import {
  packPixels,
  unpack,
  makeTemplate,
  makeWindowEnergies,
  findBestMatch,
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
const readback = buffer(
  errors.size,
  GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
);
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
let matrix = mat4.identity();
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
  matrix = mat4.multiply(
    mat4.ortho(-1.65 * aspect, 1.65 * aspect, -1.65, 1.65, 0.1, 10),
    mat4.lookAt([1.2, 2.8, 3.2], [0, 0.3, 0], [0, 1, 0])
  );
  device.queue.writeBuffer(viewBuffer, 0, matrix.buffer as ArrayBuffer);
};
resize();

const settings = {
  packed: true,
  x: 320,
  y: 248,
};
// Recognizable cutouts give the first interaction a useful starting point.
const cutouts = [
  [264, 248],
  [340, 252],
  [308, 312],
  [184, 144],
];
let cutout = 0;
let [templateX, templateY] = cutouts[cutout];
const snap = (value: number) =>
  Math.max(
    0,
    Math.min(imageSize - patchSize, Math.round(value / stride) * stride)
  );
document.querySelector<HTMLInputElement>('#packed')!.onchange = (event) => {
  settings.packed = (event.target as HTMLInputElement).checked;
  update();
};
for (const axis of ['x', 'y'] as const) {
  const input = document.querySelector<HTMLInputElement>(`#${axis}`)!;
  input.onchange = () => {
    if (Number.isFinite(input.valueAsNumber))
      settings[axis] = snap(input.valueAsNumber);
    input.value = String(settings[axis]);
    update();
  };
}
document.querySelector<HTMLButtonElement>('#next')!.onclick = () => {
  [templateX, templateY] = cutouts[++cutout % cutouts.length];
  update();
};
document.querySelector<HTMLButtonElement>('#use-template')!.onclick = () => {
  templateX = settings.x;
  templateY = settings.y;
  update();
};
source.onclick = (event) => {
  const rect = source.getBoundingClientRect();
  settings.x = snap(
    ((event.clientX - rect.left) / rect.width) * imageSize - patchSize / 2
  );
  settings.y = snap(
    ((event.clientY - rect.top) / rect.height) * imageSize - patchSize / 2
  );
  for (const axis of ['x', 'y'] as const)
    document.querySelector<HTMLInputElement>(`#${axis}`)!.value = String(
      settings[axis]
    );
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

// Serialize readback; publish the answer, image and landscape for one input revision.
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
      document.querySelector('#answer')!.textContent = 'Searching the picture…';
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(timers[mode].addTimestampWrite({}));
      pass.setPipeline(pipelines[mode]);
      pass.setBindGroup(0, groups[mode]);
      pass.dispatchWorkgroups(Math.ceil(gridSize ** 2 / 64));
      pass.end();
      timers[mode].resolve(encoder);
      encoder.copyBufferToBuffer(errors, 0, readback, 0, errors.size);
      device.queue.submit([encoder.finish()]);
      timers[mode].tryInitiateTimestampDownload();
      await readback.mapAsync(GPUMapMode.READ);
      const scores = new Int32Array(readback.getMappedRange()).slice();
      readback.unmap();
      if (submitted !== revision) continue;

      // Select the answer from computed scores, without using the cutout's origin.
      const best = findBestMatch(scores);
      const bestX = (best.index % gridSize) * stride;
      const bestY = Math.floor(best.index / gridSize) * stride;
      const error = scores[(y / stride) * gridSize + x / stride];
      const rms = (value: number) =>
        Math.sqrt(value / patchSize ** 2).toFixed(2);
      document.querySelector('#answer')!.textContent =
        `Found at (${bestX}, ${bestY}) — ${
          best.error === 0 ? 'an exact match' : 'the closest match'
        }.` +
        (best.count > 1
          ? ` ${best.count} locations tie; one is highlighted.`
          : '');
      document.querySelector('#best-error')!.textContent =
        best.error === 0
          ? 'Identical pixels · zero difference'
          : `Difference: ${rms(best.error)} / 255`;
      document.querySelector('#comparison-error')!.textContent =
        error === best.error
          ? 'Also a best match'
          : `Difference: ${rms(error)} / 255`;
      result.textContent = `Compared spot at (${x}, ${y}): ${
        error === best.error
          ? 'matches just as well as the highlighted answer.'
          : `a worse match, with ${rms(
              error
            )} grayscale levels of difference (RMS).`
      }`;

      ctx.putImageData(rgba, 0, 0);
      for (const [id, px, py] of [
        ['template', templateX, templateY],
        ['match', bestX, bestY],
        ['comparison', x, y],
      ] as const) {
        document
          .querySelector<HTMLCanvasElement>(`#${id}`)!
          .getContext('2d')!
          .drawImage(
            source,
            px,
            py,
            patchSize,
            patchSize,
            0,
            0,
            patchSize,
            patchSize
          );
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ff40c8';
      ctx.lineWidth = 3;
      ctx.strokeRect(bestX - 1, bestY - 1, patchSize + 2, patchSize + 2);
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 4;
      ctx.strokeRect(x + 2, y + 2, patchSize - 4, patchSize - 4);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 2, y + 2, patchSize - 4, patchSize - 4);

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
      )!.textContent = `First four pixels (signed = gray − 128):\nCompared [${a.join(
        ', '
      )}] → ${hex(p)}\nCutout   [${b.join(', ')}] → ${hex(
        t
      )}\ndot4I8Packed(P,T) = ${dot(a, b)}\nGroup error: ${dot(a, a)} + ${dot(
        b,
        b
      )} − 2 × (${dot(a, b)}) = ${
        dot(a, a) + dot(b, b) - 2 * dot(a, b)
      }\nFull patch error (GPU ${packed ? 'packed' : 'scalar'}): ${error}`;

      device.queue.writeBuffer(
        viewBuffer,
        64,
        new Uint32Array([
          x / stride,
          y / stride,
          bestX / stride,
          bestY / stride,
        ])
      );
      // Overlay the markers so a foreground ridge cannot hide either location.
      for (const [id, px, py, score, offset] of [
        ['best', bestX, bestY, best.error, 16],
        ['comparison', x, y, error, 0],
      ] as const) {
        const point = vec3.transformMat4(
          [
            px / stride / 60 - 1,
            (Math.sqrt(score / patchSize ** 2) / 255) * 1.5,
            py / stride / 60 - 1,
          ],
          matrix
        );
        const marker = document.querySelector<HTMLElement>(`#${id}-marker`)!;
        marker.hidden = false;
        marker.style.left = `${(point[0] + 1) * 50}%`;
        marker.style.top = `calc(${(1 - point[1]) * 50}% - ${offset}px)`;
      }
      const renderEncoder = device.createCommandEncoder();
      const render = renderEncoder.beginRenderPass({
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
      device.queue.submit([renderEncoder.finish()]);
    } while (submitted !== revision);
  } catch (error) {
    document.querySelector('#answer')!.textContent =
      'The search could not complete.';
    throw error;
  } finally {
    running = false;
  }
}
window.addEventListener('resize', () => {
  resize();
  update();
});
update();

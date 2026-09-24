import packedWGSL from './packed.wgsl';
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

async function main() {
  function unavailable(message: string) {
    document.querySelector('#answer')!.textContent = message;
    document.querySelector('#selected-score')!.textContent = 'Unavailable';
    document.querySelectorAll('button, input').forEach((control) => {
      control.setAttribute('disabled', '');
    });
  }
  if (
    !navigator.gpu?.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')
  ) {
    return unavailable(
      'Packed integer dot products are unavailable in this browser.'
    );
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter)
    return unavailable('No WebGPU adapter is available for this search.');
  const device = await adapter?.requestDevice({
    requiredFeatures: adapter.features.has('timestamp-query')
      ? ['timestamp-query']
      : [],
  });
  quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);

  // Prepare the source once. Both kernels consume the same packed grayscale data.
  const source = document.querySelector<HTMLCanvasElement>('#source')!;
  const ctx = source.getContext('2d')!;
  const heatmap = document.querySelector<HTMLCanvasElement>('#heatmap')!;
  const heat = heatmap.getContext('2d')!;
  const field = document.createElement('canvas');
  field.width = field.height = gridSize;
  const fieldContext = field.getContext('2d')!;
  const colors = fieldContext.createImageData(gridSize, gridSize);
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
      present();
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
  document.querySelector<HTMLButtonElement>('#inspect-best')!.onclick = () => {
    if (!scores.length) return;
    const best = findBestMatch(scores);
    settings.x = (best.index % gridSize) * stride;
    settings.y = Math.floor(best.index / gridSize) * stride;
    present();
  };
  for (const view of [source, heatmap])
    view.onclick = (event) => {
      const rect = view.getBoundingClientRect();
      settings.x = snap(
        ((event.clientX - rect.left) / rect.width) * imageSize - patchSize / 2
      );
      settings.y = snap(
        ((event.clientY - rect.top) / rect.height) * imageSize - patchSize / 2
      );
      present();
    };

  document.querySelector<HTMLInputElement>('#group-index')!.oninput = () =>
    explain();

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
        } ms. Excludes image preparation, rendering, and readback.`;
      })
  );
  if (!timers[0].timestampSupported)
    document.querySelector('#timing')!.textContent =
      'GPU timestamps unavailable on this device.';

  // Serialize readback; publish the answer, picture and score map for one input revision.
  let revision = 0,
    running = false;
  let scores = new Int32Array(0);
  let template = makeTemplate(image, templateX, templateY);
  async function update() {
    revision++;
    if (running) return;
    running = true;
    try {
      let submitted;
      do {
        submitted = revision;
        const { packed } = settings;
        const mode = packed ? 0 : 1;
        const nextTemplate = makeTemplate(image, templateX, templateY);
        device.queue.writeBuffer(templateBuffer, 0, nextTemplate.words);
        device.queue.writeBuffer(
          energyBuffer,
          0,
          new Int32Array([nextTemplate.energy])
        );
        document.querySelector('#answer')!.textContent =
          'Searching the picture…';
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass(
          timers[mode].addTimestampWrite({})
        );
        pass.setPipeline(pipelines[mode]);
        pass.setBindGroup(0, groups[mode]);
        pass.dispatchWorkgroups(Math.ceil(gridSize ** 2 / 64));
        pass.end();
        timers[mode].resolve(encoder);
        encoder.copyBufferToBuffer(errors, 0, readback, 0, errors.size);
        device.queue.submit([encoder.finish()]);
        timers[mode].tryInitiateTimestampDownload();
        await readback.mapAsync(GPUMapMode.READ);
        const nextScores = new Int32Array(readback.getMappedRange()).slice();
        readback.unmap();
        if (submitted !== revision) continue;
        scores = nextScores;
        template = nextTemplate;
      } while (submitted !== revision);
    } catch (error) {
      document.querySelector('#answer')!.textContent =
        'The search could not complete.';
      throw error;
    } finally {
      running = false;
    }
    present();
  }

  const rms = (value: number) => Math.sqrt(value / patchSize ** 2).toFixed(2);
  function present() {
    if (running || !scores.length) return;
    const { x, y } = settings;
    for (const axis of ['x', 'y'] as const)
      document.querySelector<HTMLInputElement>(`#${axis}`)!.value = String(
        settings[axis]
      );
    document.querySelector('#position')!.textContent = `${x}, ${y}`;
    // Select the answer from computed scores, without using the cutout's origin.
    const best = findBestMatch(scores);
    const bestX = (best.index % gridSize) * stride;
    const bestY = Math.floor(best.index / gridSize) * stride;
    const error = scores[(y / stride) * gridSize + x / stride];
    document.querySelector('#answer')!.textContent =
      `Found at (${bestX}, ${bestY}) — ${
        best.error === 0 ? 'an exact match' : 'the closest match'
      }; difference score ${rms(best.error)}.` +
      (best.count > 1
        ? ` ${best.count} locations tie; one is highlighted.`
        : '');
    document.querySelector('#selected-score')!.textContent = rms(error);
    const samePlace = bestX === x && bestY === y;
    for (const [id, px, py, label] of [
      ['map-best-label', bestX, bestY, `Best · ${rms(best.error)}`],
      [
        'map-comparison-label',
        x,
        y,
        samePlace
          ? `Best + inspecting · ${rms(error)}`
          : `Inspecting · ${rms(error)}`,
      ],
    ] as const) {
      const marker = document.querySelector<HTMLElement>(`#${id}`)!;
      marker.hidden = samePlace && id === 'map-best-label';
      marker.textContent = label;
      marker.style.setProperty(
        '--marker-x',
        `${((px + patchSize / 2) / imageSize) * 100}%`
      );
      marker.style.setProperty(
        '--marker-y',
        `${((py + patchSize / 2) / imageSize) * 100}%`
      );
    }

    ctx.putImageData(rgba, 0, 0);
    for (const [id, px, py] of [
      ['template', templateX, templateY],
      ['comparison', x, y],
      ['picture-detail', x, y],
      ['cutout-detail', templateX, templateY],
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
    // Fixed RMS/255 color scale for the score map.
    for (let i = 0; i < scores.length; i++) {
      const error = Math.sqrt(scores[i] / patchSize ** 2) / 255;
      colors.data.set(
        [
          (0.04 + 0.96 * error) * 255,
          (0.65 - 0.2 * error) * 255,
          (0.75 - 0.63 * error) * 255,
          255,
        ],
        i * 4
      );
    }
    fieldContext.putImageData(colors, 0, 0);
    heat.clearRect(0, 0, imageSize, imageSize);
    heat.imageSmoothingEnabled = false;
    // A grid sample is centered on its patch, not stretched to the image edge.
    const inset = (patchSize - stride) / 2;
    heat.drawImage(field, inset, inset, gridSize * stride, gridSize * stride);
    for (const overlay of [ctx, heat]) {
      overlay.setLineDash([]);
      overlay.strokeStyle = '#ff40c8';
      overlay.lineWidth = 3;
      overlay.strokeRect(bestX - 1, bestY - 1, patchSize + 2, patchSize + 2);
      overlay.strokeStyle = '#111';
      overlay.lineWidth = 4;
      overlay.strokeRect(x + 2, y + 2, patchSize - 4, patchSize - 4);
      overlay.setLineDash([4, 4]);
      overlay.strokeStyle = '#fff';
      overlay.lineWidth = 2;
      overlay.strokeRect(x + 2, y + 2, patchSize - 4, patchSize - 4);
    }

    explain();
  }

  function explain() {
    if (running || !scores.length) return;
    const { x, y } = settings;
    const error = scores[(y / stride) * gridSize + x / stride];
    document.querySelector('#lesson-score')!.textContent = rms(error);
    const groupIndex =
      document.querySelector<HTMLInputElement>('#group-index')!.valueAsNumber;
    const gx = (groupIndex % (patchSize / 4)) * 4,
      gy = Math.floor(groupIndex / (patchSize / 4));
    document.querySelector('#group-number')!.textContent = `${
      groupIndex + 1
    } of 256`;
    for (const marker of document.querySelectorAll<HTMLElement>(
      '.group-highlight'
    )) {
      marker.style.left = `${(gx / patchSize) * 100}%`;
      marker.style.top = `${(gy / patchSize) * 100}%`;
    }
    const p = image[((y + gy) * imageSize + x + gx) / 4],
      t = template.words[groupIndex];
    const lanes = (word: number) => [0, 1, 2, 3].map((i) => unpack(word, i));
    const a = lanes(p),
      b = lanes(t);
    const differences = a.map((value, i) => value - b[i]);
    const squared = differences.map((value) => value * value);
    const pixelRow = (label: string, values: number[], grayscale = false) =>
      `<tr><th scope="row">${label}</th>${values
        .map((value) =>
          grayscale
            ? `<td><span class="gray-swatch" style="background:rgb(${value} ${value} ${value});color:${
                value < 128 ? 'white' : 'black'
              }">${value}</span></td>`
            : `<td>${value.toLocaleString('en-US')}</td>`
        )
        .join('')}</tr>`;
    document.querySelector('#pixel-math')!.innerHTML =
      pixelRow(
        'Cutout',
        b.map((value) => value + 128),
        true
      ) +
      pixelRow(
        'Inspected',
        a.map((value) => value + 128),
        true
      ) +
      pixelRow('Difference<br><small>inspected − cutout</small>', differences) +
      pixelRow('Squared', squared);
    document.querySelector(
      '#direct'
    )!.textContent = `These four pairs contribute ${squared
      .reduce((sum, value) => sum + value, 0)
      .toLocaleString('en-US')} to the total squared error.`;
    const dot = (a: number[], b: number[]) =>
      a.reduce((sum, v, i) => sum + v * b[i], 0);
    const hex = (v: number) =>
      `0x${v.toString(16).padStart(8, '0').toUpperCase()}`;
    for (const [id, values, word] of [
      ['picture', a, p],
      ['cutout', b, t],
    ] as const) {
      document.querySelector(`#${id}-pixels`)!.innerHTML = values
        .map(
          (value) =>
            `<span><b style="background:rgb(${value + 128} ${value + 128} ${
              value + 128
            });color:${value < 0 ? 'white' : 'black'}">${
              value + 128
            }</b>${value}</span>`
        )
        .join('');
      document.querySelector(`#${id}-word`)!.textContent = hex(word);
    }
    document.querySelector('#dot')!.textContent = `dot4I8Packed(${hex(
      p
    )}, ${hex(t)}) = ${dot(a, b)}\n${a
      .map((v, i) => `(${v} × ${b[i]})`)
      .join(' + ')} = ${dot(a, b)}`;
    document.querySelector(
      '#group'
    )!.textContent = `These four pairs contribute ${dot(a, a)} + ${dot(
      b,
      b
    )} − 2 × (${dot(a, b)}) = ${
      dot(a, a) + dot(b, b) - 2 * dot(a, b)
    } to the error.`;
    document.querySelector(
      '#full-score'
    )!.textContent = `Across all 1,024 pixel pairs, the total squared error is ${error.toLocaleString(
      'en-US'
    )}.\nDifference score = √(${error.toLocaleString('en-US')} ÷ 1,024) = ${rms(
      error
    )}.`;
  }

  update();
}
main();

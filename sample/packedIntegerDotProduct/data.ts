export const imageSize = 512;
export const patchSize = 32;
export const stride = 4;
export const gridSize = (imageSize - patchSize) / stride + 1;

export function packPixels(pixels: Uint8Array) {
  const words = new Uint32Array(pixels.length / 4);
  for (let i = 0; i < pixels.length; i++) {
    // Centering both operands preserves their difference. c0 is the low byte.
    words[i >> 2] |= ((pixels[i] - 128) & 255) << ((i % 4) * 8);
  }
  return words;
}

export function unpack(word: number, lane: number): number {
  return (word << (24 - lane * 8)) >> 24;
}

export function makeTemplate(image: Uint32Array, x: number, y: number) {
  const words = new Uint32Array((patchSize * patchSize) / 4);
  let energy = 0;
  for (let row = 0; row < patchSize; row++) {
    for (let col = 0; col < patchSize / 4; col++) {
      const word = image[((y + row) * imageSize + x) / 4 + col];
      words[(row * patchSize) / 4 + col] = word;
      for (let lane = 0; lane < 4; lane++) energy += unpack(word, lane) ** 2;
    }
  }
  return { words, energy };
}

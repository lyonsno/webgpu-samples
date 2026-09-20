// Presentation only: the packed dot product is still computed in packed.wgsl.
export type Side = 'lhs' | 'rhs';
export type Vector = readonly [number, number, number, number];

export function createVisualization(
  edit: (side: Side, component: number, value: number) => void
) {
  const bytes = document.querySelector('#bytes') as HTMLElement;
  const bits = document.querySelector('#bits') as HTMLElement;
  const selectedLabel = document.querySelector('#selected-byte') as HTMLElement;
  const roles = { lhs: 'Input', rhs: 'Weight' };
  let selected: [Side, number] = ['lhs', 1];
  let vectors: Record<Side, Vector>;
  let words: Record<Side, number>;

  // Conventional hex notation puts the most significant byte on the left.
  for (const side of ['lhs', 'rhs'] as const) {
    const row = document.createElement('div');
    row.className = `byte-row ${side}`;
    row.innerHTML = `<strong>${roles[side]}s</strong><span>0x</span>`;
    for (const component of [3, 2, 1, 0]) {
      const button = document.createElement('button');
      button.dataset.byte = `${side}${component}`;
      button.addEventListener('click', () => {
        select(side, component);
        (document.querySelector('#bit-details') as HTMLDetailsElement).open =
          true;
      });
      row.append(button);
    }
    bytes.append(row);
  }
  for (let bit = 7; bit >= 0; bit--) {
    const button = document.createElement('button');
    button.dataset.bit = String(bit);
    button.addEventListener('click', () => {
      const [side, component] = selected;
      const byte = (vectors[side][component] & 0xff) ^ (1 << bit);
      edit(side, component, byte >= 128 ? byte - 256 : byte);
    });
    bits.append(button);
  }

  function select(side: Side, component: number) {
    selected = [side, component];
    render();
  }

  function render() {
    if (!vectors) return;
    const { lhs, rhs } = vectors;
    const [side, component] = selected;
    for (const s of ['lhs', 'rhs'] as const) {
      document.querySelector(`#word-${s}`)!.textContent = `0x${words[s]
        .toString(16)
        .padStart(8, '0')
        .toUpperCase()}`;
    }
    for (const button of bytes.querySelectorAll<HTMLButtonElement>('button')) {
      const key = button.dataset.byte!;
      const s = key.slice(0, 3) as Side;
      const i = Number(key[3]);
      const hex = ((words[s] >>> (8 * i)) & 0xff)
        .toString(16)
        .padStart(2, '0')
        .toUpperCase();
      button.innerHTML = `<small>c${i}</small><b>${hex}</b><small>${vectors[s][i]}</small>`;
      button.setAttribute(
        'aria-label',
        `${roles[s]} ${i}: ${vectors[s][i]}, hex ${hex}`
      );
      button.setAttribute(
        'aria-pressed',
        String(s === side && i === component)
      );
    }
    selectedLabel.textContent = `${roles[side]} ${component} = ${vectors[side][component]}`;
    const step = vectors[side][component] === 127 ? -1 : 1;
    const other = vectors[side === 'lhs' ? 'rhs' : 'lhs'][component];
    document.querySelector('#prediction')!.textContent = `Try ${
      step > 0 ? '+1' : '−1'
    } to ${roles[side]} ${component}: the sum changes by ${step * other}.`;
    for (const button of bits.querySelectorAll<HTMLButtonElement>('button')) {
      const bit = Number(button.dataset.bit);
      const on = (vectors[side][component] >>> bit) & 1;
      const weight = bit === 7 ? -128 : 1 << bit;
      button.innerHTML = `<b>${on}</b><small>${weight}</small>`;
      button.setAttribute('aria-label', `Toggle bit ${bit}, weight ${weight}`);
      button.setAttribute('aria-pressed', String(Boolean(on)));
    }

    const terms = lhs.map((value, i) => `(${value}) × (${rhs[i]})`);
    const sum = lhs.reduce((total, value, i) => total + value * rhs[i], 0);
    document.querySelector('#sum')!.textContent = `${terms.join(
      ' + '
    )} = ${sum}`;
  }

  return {
    select,
    update(lhs: Vector, rhs: Vector, lhsWord: number, rhsWord: number) {
      vectors = { lhs, rhs };
      words = { lhs: lhsWord, rhs: rhsWord };
      render();
    },
  };
}

// Presentation only: the packed dot product is still computed in packed.wgsl.
export type Side = 'lhs' | 'rhs';
export type Vector = readonly [number, number, number, number];

export function createVisualization(
  edit: (side: Side, component: number, value: number) => void
) {
  const plots = document.querySelector('#planes') as HTMLElement;
  const bytes = document.querySelector('#bytes') as HTMLElement;
  const bits = document.querySelector('#bits') as HTMLElement;
  const selectedLabel = document.querySelector('#selected-byte') as HTMLElement;
  let selected: [Side, number] = ['lhs', 0];
  let vectors: Record<Side, Vector>;
  let words: Record<Side, number>;

  // Conventional hex notation puts the most significant byte on the left.
  for (const side of ['lhs', 'rhs'] as const) {
    const row = document.createElement('div');
    row.className = `byte-row ${side}`;
    row.innerHTML = `<strong>${side.toUpperCase()}</strong><span>0x</span>`;
    for (const component of [3, 2, 1, 0]) {
      const button = document.createElement('button');
      button.dataset.byte = `${side}${component}`;
      button.addEventListener('click', () => select(side, component));
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
        `${s} component ${i}: ${vectors[s][i]}, hex ${hex}`
      );
      button.setAttribute(
        'aria-pressed',
        String(s === side && i === component)
      );
    }
    selectedLabel.textContent = `${side.toUpperCase()} component ${component} = ${
      vectors[side][component]
    }`;
    for (const button of bits.querySelectorAll<HTMLButtonElement>('button')) {
      const bit = Number(button.dataset.bit);
      const on = (vectors[side][component] >>> bit) & 1;
      const weight = bit === 7 ? -128 : 1 << bit;
      button.innerHTML = `<b>${on}</b><small>${weight}</small>`;
      button.setAttribute('aria-label', `Toggle bit ${bit}, weight ${weight}`);
      button.setAttribute('aria-pressed', String(Boolean(on)));
    }

    plots.innerHTML = [0, 2]
      .map((start) => {
        const a = [lhs[start], lhs[start + 1]];
        const b = [rhs[start], rhs[start + 1]];
        const dot = a[0] * b[0] + a[1] * b[1];
        const lengthSquared = b[0] ** 2 + b[1] ** 2;
        // Project LHS onto the line through RHS. A zero RHS has no direction.
        const projection = b.map((v) =>
          lengthSquared ? (v * dot) / lengthSquared : 0
        );
        const largest = Math.max(
          ...a.map(Math.abs),
          ...b.map(Math.abs),
          ...projection.map(Math.abs)
        );
        const extent = Math.max(8, 2 ** Math.ceil(Math.log2(largest + 1)));
        const scale = 115 / extent;
        const point = (v: number[]) => [v[0] * scale, -v[1] * scale];
        const [ax, ay] = point(a);
        const [bx, by] = point(b);
        const [px, py] = point(projection);
        const arrow = (x: number, y: number, name: Side) =>
          `<line class="${name} arrow" x1="0" y1="0" x2="${x}" y2="${y}" marker-end="url(#${name}-${start})" />`;
        const selectedPlane = component >= start && component < start + 2;
        return `<figure class="${selectedPlane ? 'selected-plane' : ''}">
        <figcaption>Components ${start} & ${start + 1}</figcaption>
        <svg data-plane="${start}" viewBox="-160 -150 320 300" role="img"
          aria-label="Components ${start}, ${start + 1}: LHS ${a.join(
          ', '
        )}, RHS ${b.join(', ')}. Partial dot product ${dot}.">
          <defs>${(['lhs', 'rhs'] as const)
            .map(
              (name) =>
                `<marker id="${name}-${start}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path class="${name}" d="M 0 0 L 10 5 L 0 10 z" /></marker>`
            )
            .join('')}</defs>
          <path class="grid" d="M -115 -115 H 115 V 115 H -115 Z M -57.5 -115 V 115 M 57.5 -115 V 115 M -115 -57.5 H 115 M -115 57.5 H 115" />
          <path class="axis" d="M -128 0 H 128 M 0 -128 V 128" />
          <text x="130" y="-7">c${start}</text><text x="7" y="-132">c${
          start + 1
        }</text>
          <text x="115" y="132" text-anchor="middle">${extent}</text><text x="-115" y="132" text-anchor="middle">${-extent}</text>
          <text x="-9" y="-115" text-anchor="end">${extent}</text><text x="-9" y="115" text-anchor="end">${-extent}</text>
          ${
            lengthSquared
              ? `<line class="projection" x1="${ax}" y1="${ay}" x2="${px}" y2="${py}" /><line class="projection-foot" x1="0" y1="0" x2="${px}" y2="${py}" /><circle class="projection-point" cx="${px}" cy="${py}" r="3" />`
              : ''
          }
          ${arrow(bx, by, 'rhs')}${arrow(ax, ay, 'lhs')}
          <circle class="origin" r="2.5" />
        </svg>
        <div class="coordinates"><span class="lhs">LHS (${a.join(
          ', '
        )})</span> <span class="rhs">RHS (${b.join(', ')})</span></div>
        <div class="products">(${a[0]}) × (${b[0]}) + (${a[1]}) × (${
          b[1]
        }) = <strong>${dot}</strong></div>
        <small>${
          lengthSquared
            ? 'Dashed: LHS projected onto the RHS line.'
            : 'RHS is zero: projection has no direction.'
        }</small>
      </figure>`;
      })
      .join('');
    const partials = [0, 2].map(
      (i) => lhs[i] * rhs[i] + lhs[i + 1] * rhs[i + 1]
    );
    document.querySelector('#sum')!.textContent = `${partials[0]} + (${
      partials[1]
    }) = ${partials[0] + partials[1]}`;
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

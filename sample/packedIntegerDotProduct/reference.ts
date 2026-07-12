export const packedDotLanguageFeature = 'packed_4x8_integer_dot_product';

export type RequestedRoute = 'auto' | 'packed' | 'scalar';
export type EffectiveRoute = 'packed' | 'scalar';

export type RouteSelection = {
  requestedRoute: RequestedRoute;
  effectiveRoute: EffectiveRoute;
  packedLanguageFeatureSupported: boolean;
  fallbackReason: string | null;
};

function assertI8(value: number) {
  if (!Number.isInteger(value) || value < -128 || value > 127) {
    throw new RangeError(`expected signed 8-bit integer, received ${value}`);
  }
}

export function pack4xI8(values: readonly number[]): number {
  if (values.length !== 4) {
    throw new RangeError(`expected 4 values, received ${values.length}`);
  }

  let packed = 0;
  values.forEach((value, index) => {
    assertI8(value);
    packed |= (value & 0xff) << (index * 8);
  });
  return packed >>> 0;
}

export function dot4I8(lhs: readonly number[], rhs: readonly number[]): number {
  if (lhs.length !== 4 || rhs.length !== 4) {
    throw new RangeError('signed dot product requires two 4-component vectors');
  }

  let result = 0;
  for (let index = 0; index < 4; ++index) {
    assertI8(lhs[index]);
    assertI8(rhs[index]);
    result += lhs[index] * rhs[index];
  }
  return result;
}

export function selectRoute(
  requestedRoute: RequestedRoute,
  languageFeatures: ReadonlySet<string>
): RouteSelection {
  const packedLanguageFeatureSupported = languageFeatures.has(
    packedDotLanguageFeature
  );
  const wantsPacked = requestedRoute !== 'scalar';
  const effectiveRoute =
    wantsPacked && packedLanguageFeatureSupported ? 'packed' : 'scalar';

  return {
    requestedRoute,
    effectiveRoute,
    packedLanguageFeatureSupported,
    fallbackReason:
      requestedRoute === 'packed' && !packedLanguageFeatureSupported
        ? `WGSL language feature '${packedDotLanguageFeature}' is unavailable`
        : null,
  };
}

export function validateRun({
  requestedRoute,
  effectiveRoute,
  packedLanguageFeatureSupported,
  expected,
  actual,
}: RouteSelection & {
  expected: ArrayLike<number>;
  actual: ArrayLike<number>;
}) {
  if (effectiveRoute === 'packed' && !packedLanguageFeatureSupported) {
    throw new Error(
      `requested route '${requestedRoute}' claims packed route without language feature support`
    );
  }
  if (actual.length !== expected.length) {
    throw new Error(
      `partial output: expected ${expected.length} results, received ${actual.length}`
    );
  }
  for (let index = 0; index < expected.length; ++index) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `result ${index} mismatch: expected ${expected[index]}, received ${actual[index]}`
      );
    }
  }
}

export function makeInputVectors(count: number) {
  const lhsPacked = new Uint32Array(count);
  const rhsPacked = new Uint32Array(count);
  const expected = new Int32Array(count);
  let state = 0x13579bdf;

  const nextI8 = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state >>> 24) & 0xff) - 128;
  };

  for (let index = 0; index < count; ++index) {
    const lhs = [nextI8(), nextI8(), nextI8(), nextI8()];
    const rhs = [nextI8(), nextI8(), nextI8(), nextI8()];
    lhsPacked[index] = pack4xI8(lhs);
    rhsPacked[index] = pack4xI8(rhs);
    expected[index] = dot4I8(lhs, rhs);
  }

  return { lhsPacked, rhsPacked, expected };
}

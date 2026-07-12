import assert from 'node:assert/strict';
import test from 'node:test';

const referenceModuleUrl = new URL(
  '../sample/packedIntegerDotProduct/reference.ts',
  import.meta.url
);

async function loadReferenceModule() {
  try {
    return await import(referenceModuleUrl);
  } catch (error) {
    if (error?.code === 'ERR_MODULE_NOT_FOUND') {
      assert.fail(
        'packed integer dot-product reference and route contract is missing'
      );
    }
    throw error;
  }
}

test('signed packed dot products match scalar arithmetic', async () => {
  const { dot4I8, pack4xI8 } = await loadReferenceModule();
  const lhs = [1, -2, 3, -4];
  const rhs = [-5, 6, -7, 8];

  assert.equal(pack4xI8([1, -2, 127, -128]), 0x807ffe01);
  assert.equal(dot4I8(lhs, rhs), -70);
});

test('route selection records requested and effective identity', async () => {
  const { selectRoute } = await loadReferenceModule();
  const feature = 'packed_4x8_integer_dot_product';

  assert.deepEqual(selectRoute('auto', new Set([feature])), {
    requestedRoute: 'auto',
    effectiveRoute: 'packed',
    packedLanguageFeatureSupported: true,
    fallbackReason: null,
  });
  assert.deepEqual(selectRoute('packed', new Set()), {
    requestedRoute: 'packed',
    effectiveRoute: 'scalar',
    packedLanguageFeatureSupported: false,
    fallbackReason: `WGSL language feature '${feature}' is unavailable`,
  });
  assert.deepEqual(selectRoute('scalar', new Set([feature])), {
    requestedRoute: 'scalar',
    effectiveRoute: 'scalar',
    packedLanguageFeatureSupported: true,
    fallbackReason: null,
  });
});

test('evidence validation rejects wrong routes and partial output', async () => {
  const { validateRun } = await loadReferenceModule();

  assert.throws(
    () =>
      validateRun({
        requestedRoute: 'packed',
        effectiveRoute: 'packed',
        packedLanguageFeatureSupported: false,
        expected: [11, 22],
        actual: [11, 22],
      }),
    /claims packed route without language feature support/
  );
  assert.throws(
    () =>
      validateRun({
        requestedRoute: 'auto',
        effectiveRoute: 'scalar',
        packedLanguageFeatureSupported: false,
        expected: [11, 22],
        actual: [11],
      }),
    /partial output: expected 2 results, received 1/
  );
  assert.throws(
    () =>
      validateRun({
        requestedRoute: 'auto',
        effectiveRoute: 'scalar',
        packedLanguageFeatureSupported: false,
        expected: [11, 22],
        actual: [11, 23],
      }),
    /result 1 mismatch: expected 22, received 23/
  );
});

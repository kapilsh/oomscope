import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import pickle from 'node:child_process'

import { unpickle, UnpickleError } from '../src/lib/unpickle.js'

const here = dirname(fileURLToPath(import.meta.url))
const fixture = (name) => readFileSync(join(here, 'fixtures', name))
const expected = (name) => JSON.parse(readFileSync(join(here, 'fixtures', name), 'utf8'))

// Round-trip through JSON before comparing: the fixtures' ground truth is what
// Python's json module produced, so comparing decoded-to-JSON against it is the
// only apples-to-apples check. It also normalises tuple-vs-list, which both
// sides represent as arrays.
const asJson = (v) => JSON.parse(JSON.stringify(v))

for (const proto of [2, 3, 4, 5]) {
  test(`decodes every value shape at protocol ${proto}`, () => {
    const got = unpickle(fixture(`types_p${proto}.pickle`))
    assert.deepEqual(asJson(got), expected(`types_p${proto}.json`))
  })
}

test('decodes a snapshot-shaped payload', () => {
  const got = unpickle(fixture('snapshot_shape.pickle'))
  assert.deepEqual(asJson(got), expected('snapshot_shape.json'))
})

test('large and negative integers stay exact', () => {
  const got = unpickle(fixture('types_p4.pickle'))
  assert.equal(got.addr, 138772641480704)
  assert.equal(got.time_us, 1789995997101492)
  assert.equal(got.int_2p48, 2 ** 48)
  assert.equal(got.neg_big, -(2 ** 40))
  assert.equal(got.neg_small, -1)
  assert.equal(got.int_255, 255)
  assert.equal(got.int_256, 256)
})

test('a shared object is decoded once and aliased through the memo', () => {
  const got = unpickle(fixture('types_p4.pickle'))
  assert.equal(got.shared_a, got.shared_b, 'memo should return the same object')
})

test('refuses a pickle that would construct a Python object', () => {
  // `pickle.dumps(collections.OrderedDict())` needs GLOBAL/REDUCE, which is the
  // class of payload that makes unpickling untrusted data dangerous.
  const evil = pickle.execFileSync('python3', [
    '-c',
    'import pickle,sys,collections; sys.stdout.buffer.write(pickle.dumps(collections.OrderedDict([("a",1)])))',
  ])
  assert.throws(() => unpickle(evil), UnpickleError)
})

test('refuses a truncated file rather than returning a partial tree', () => {
  const full = fixture('snapshot_shape.pickle')
  assert.throws(() => unpickle(full.subarray(0, full.length - 20)), UnpickleError)
})

test('refuses trailing garbage', () => {
  const full = fixture('snapshot_shape.pickle')
  const garbage = Buffer.concat([full.subarray(0, full.length - 1), Buffer.from([0xff])])
  assert.throws(() => unpickle(garbage), UnpickleError)
})

import test from 'node:test'
import assert from 'node:assert/strict'

import { parseSnapshot } from '../src/lib/snapshot.js'
import { blameFrame, frameLabel, fxFrame, fxLabel, provenance } from '../src/lib/frames.js'
import { buildAllocModel, describe } from '../src/lib/allocTimeline.js'

const gen = {
  filename: '/tmp/torchinductor_me/ab/cabc.py', line: 90, name: 'call',
  fx_node_op: 'call_function', fx_node_name: 'mm_1', fx_node_target: 'aten.mm.default',
  fx_original_trace: 'File "model.py", line 12, in forward\n    y = self.fc(x)',
}
const user = { filename: '/app/train.py', line: 40, name: 'step' }

const raw = {
  segments: [{
    device: 0, address: 0, total_size: 4096, stream: 0, segment_type: 'small',
    blocks: [
      { address: 0, size: 1024, requested_size: 1000, state: 'active_allocated', frames: [gen, user], forward_frames: ['  File "model.py", line 12, in forward\n', '    y = self.fc(x)\n'] },
      { address: 1024, size: 3072, state: 'inactive', frames: [] },
    ],
  }],
  device_traces: [[
    { action: 'segment_alloc', addr: 0, size: 4096, stream: 0, frames: [user], compile_context: 'N/A', user_metadata: '' },
    { action: 'alloc', addr: 0, size: 1000, stream: 0, frames: [gen, user], compile_context: 'Torch-Compiled Region: 0/0', user_metadata: { phase: 'forward', step: 2 }, internal_metadata: 'mallocWithAddress' },
  ]],
}

test('dict user_metadata becomes its JSON, once, at parse', () => {
  const m = parseSnapshot(structuredClone(raw))
  const d = m.devices[0]
  assert.equal(d.timeline.events[1].user_metadata, '{"phase":"forward","step":2}')
  assert.equal(d.segments[0].blocks[0].metadata, '{"phase":"forward","step":2}')
})

test('a live block carries its alloc event, so compile context and tags reach the views', () => {
  const d = parseSnapshot(structuredClone(raw)).devices[0]
  const p = provenance(d.segments[0].blocks[0])
  assert.equal(p.compileContext, 'Torch-Compiled Region: 0/0')
  assert.equal(p.internal, 'mallocWithAddress')
  assert.equal(p.forward.length, 2)
  // "N/A" is torch saying "not in a compiled region": nothing to show.
  assert.equal(provenance(d.timeline.events[0]).compileContext, '')
})

test('generated code is not blamed; its FX node is shown instead', () => {
  assert.equal(frameLabel(blameFrame([gen, user])), 'train.py:40  step')
  assert.equal(fxLabel(fxFrame([user, gen, user])), 'node mm_1 · call_function · aten.mm.default')
  assert.equal(fxLabel(user), '')
})

test('the timeline describes a band with its provenance', () => {
  const d = parseSnapshot(structuredClone(raw)).devices[0]
  const m = buildAllocModel(d.timeline.events, d.segments, 'blocks')
  const slot = [...m.addAt].find((s) => s >= 0)
  const x = describe(m, slot)
  assert.equal(x.prov.compileContext, 'Torch-Compiled Region: 0/0')
  assert.equal(x.metadata, '{"phase":"forward","step":2}')
})

test('a backward allocation, C++ frames only, is blamed on its autograd node', () => {
  const cpp = (name, filename = '??') => ({ filename, line: 0, name })
  const frames = [
    cpp('torch::unwind::unwind()'),
    cpp('torch::cuda::(anonymous namespace)::gather_with_cpp()', 'memory_snapshot.cpp'),
    cpp('c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void**)'),
    cpp('at::_ops::mm::call(at::Tensor const&, at::Tensor const&)'),
    cpp('torch::autograd::generated::AddmmBackward0::apply(std::vector<at::Tensor>&&)'),
    cpp('torch::autograd::Engine::thread_main()'),
  ]
  assert.equal(frameLabel(blameFrame(frames)), 'AddmmBackward0  (autograd backward)')
  assert.equal(frameLabel(blameFrame(frames.slice(0, 4))), 'aten::mm')
  assert.equal(blameFrame(frames.slice(0, 3)).name.includes('malloc'), true)
})

// A minimal pickle reader, enough for PyTorch memory snapshots.
//
// `torch.cuda.memory._dump_snapshot()` writes `pickle.dump(snapshot, f)`, and a
// snapshot is nothing but dicts, lists, tuples, strings, ints, floats and bools.
// So we do not need a general unpickler -- no GLOBAL, no REDUCE, no object
// construction, and therefore none of the code-execution surface that makes
// unpickling untrusted files dangerous. Anything outside that value subset
// raises instead of being interpreted, which is what keeps opening a stranger's
// snapshot in a browser tab a safe thing to do.
//
// Opcodes observed in real snapshots (protocol 4) are all covered; the rest of
// protocol 2-5's value opcodes are implemented too so that snapshots from other
// torch versions, or a future default protocol, do not fall over.

const OP = {
  MARK: 0x28, // (
  EMPTY_TUPLE: 0x29, // )
  STOP: 0x2e, // .
  BINFLOAT: 0x47, // G
  BININT: 0x4a, // J
  BININT1: 0x4b, // K
  NONE: 0x4e, // N
  BININT2: 0x4d, // M
  BINUNICODE: 0x58, // X
  EMPTY_LIST: 0x5d, // ]
  APPEND: 0x61, // a
  BUILD: 0x62, // b
  GLOBAL: 0x63, // c
  REDUCE: 0x52, // R
  DICT: 0x64, // d
  APPENDS: 0x65, // e
  BINGET: 0x68, // h
  LONG_BINGET: 0x6a, // j
  LIST: 0x6c, // l
  OBJ: 0x6f, // o
  BINPUT: 0x71, // q
  LONG_BINPUT: 0x72, // r
  SETITEM: 0x73, // s
  TUPLE: 0x74, // t
  SETITEMS: 0x75, // u
  EMPTY_DICT: 0x7d, // }
  PROTO: 0x80,
  NEWOBJ: 0x81,
  BINBYTES: 0x42, // B
  SHORT_BINBYTES: 0x43, // C
  TUPLE1: 0x85,
  TUPLE2: 0x86,
  TUPLE3: 0x87,
  NEWTRUE: 0x88,
  NEWFALSE: 0x89,
  LONG1: 0x8a,
  LONG4: 0x8b,
  SHORT_BINUNICODE: 0x8c,
  BINUNICODE8: 0x8d,
  BINBYTES8: 0x8e,
  EMPTY_SET: 0x8f,
  ADDITEMS: 0x90,
  FROZENSET: 0x91,
  NEWOBJ_EX: 0x92,
  STACK_GLOBAL: 0x93,
  MEMOIZE: 0x94,
  FRAME: 0x95,
  BYTEARRAY8: 0x96,
  NEXT_BUFFER: 0x97,
  READONLY_BUFFER: 0x98,
}

// Opcodes that would build arbitrary Python objects. A memory snapshot never
// contains them, and honouring them is exactly what makes pickle unsafe, so we
// name them in the error rather than silently returning something half-read.
const UNSAFE = new Set([
  OP.GLOBAL,
  OP.STACK_GLOBAL,
  OP.REDUCE,
  OP.BUILD,
  OP.OBJ,
  OP.NEWOBJ,
  OP.NEWOBJ_EX,
])

const utf8 = new TextDecoder('utf-8')

export class UnpickleError extends Error {}

/**
 * Decode a pickled value from bytes.
 *
 * Python tuples come back as arrays and dicts as plain objects, which is lossy
 * in general but exact for snapshots: every dict key in one is a string.
 *
 * @param {ArrayBuffer|Uint8Array} input
 * @returns {unknown}
 */
export function unpickle(input) {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input)
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const stack = []
  const marks = []
  const memo = new Map()
  let i = 0

  const need = (n) => {
    if (i + n > buf.length) {
      throw new UnpickleError(`truncated pickle: wanted ${n} bytes at ${i}, have ${buf.length - i}`)
    }
  }
  const u8 = () => { need(1); return buf[i++] }
  const u16 = () => { need(2); const v = view.getUint16(i, true); i += 2; return v }
  const i32 = () => { need(4); const v = view.getInt32(i, true); i += 4; return v }
  const u32 = () => { need(4); const v = view.getUint32(i, true); i += 4; return v }
  const u64 = () => { need(8); const v = view.getBigUint64(i, true); i += 8; return Number(v) }
  const str = (n) => { need(n); const s = utf8.decode(buf.subarray(i, i + n)); i += n; return s }
  const bytes = (n) => { need(n); const b = buf.slice(i, i + n); i += n; return b }

  // Little-endian two's-complement integer of arbitrary width (LONG1/LONG4).
  // Addresses and time_us in a snapshot are ~2^48, comfortably exact as
  // doubles; BigInt keeps the arithmetic exact up to that conversion so we
  // never accumulate rounding error while shifting bytes in.
  const longN = (n) => {
    need(n)
    if (n === 0) { return 0 }
    let v = 0n
    for (let k = n - 1; k >= 0; k--) { v = (v << 8n) | BigInt(buf[i + k]) }
    if (buf[i + n - 1] & 0x80) { v -= 1n << BigInt(8 * n) } // sign-extend
    i += n
    return Number(v)
  }

  const popMark = () => {
    if (marks.length === 0) { throw new UnpickleError(`no MARK on the stack at byte ${i}`) }
    return stack.splice(marks.pop())
  }

  for (;;) {
    need(1)
    const op = buf[i++]

    if (UNSAFE.has(op)) {
      throw new UnpickleError(
        `pickle opcode 0x${op.toString(16)} at byte ${i - 1} builds a Python object. ` +
        `This file is not a plain memory snapshot, and oomscope will not execute it.`,
      )
    }

    switch (op) {
      case OP.PROTO: {
        const p = u8()
        if (p > 5) { throw new UnpickleError(`pickle protocol ${p} is newer than 5`) }
        break
      }
      case OP.FRAME: u64(); break // a size hint; we already hold the whole buffer
      case OP.STOP: {
        if (stack.length !== 1) {
          throw new UnpickleError(`STOP with ${stack.length} values on the stack, expected 1`)
        }
        return stack[0]
      }

      case OP.NONE: stack.push(null); break
      case OP.NEWTRUE: stack.push(true); break
      case OP.NEWFALSE: stack.push(false); break

      case OP.BININT: stack.push(i32()); break
      case OP.BININT1: stack.push(u8()); break
      case OP.BININT2: stack.push(u16()); break
      case OP.LONG1: stack.push(longN(u8())); break
      case OP.LONG4: stack.push(longN(u32())); break
      case OP.BINFLOAT: { need(8); stack.push(view.getFloat64(i, false)); i += 8; break } // big-endian

      case OP.SHORT_BINUNICODE: stack.push(str(u8())); break
      case OP.BINUNICODE: stack.push(str(u32())); break
      case OP.BINUNICODE8: stack.push(str(u64())); break
      case OP.SHORT_BINBYTES: stack.push(bytes(u8())); break
      case OP.BINBYTES: stack.push(bytes(u32())); break
      case OP.BINBYTES8: case OP.BYTEARRAY8: stack.push(bytes(u64())); break

      case OP.EMPTY_LIST: stack.push([]); break
      case OP.EMPTY_DICT: stack.push({}); break
      case OP.EMPTY_TUPLE: stack.push([]); break
      case OP.EMPTY_SET: stack.push(new Set()); break

      case OP.MARK: marks.push(stack.length); break

      case OP.TUPLE: stack.push(popMark()); break
      case OP.TUPLE1: stack.push([stack.pop()]); break
      case OP.TUPLE2: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break }
      case OP.TUPLE3: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break }
      case OP.LIST: stack.push(popMark()); break
      case OP.DICT: {
        const items = popMark()
        const d = {}
        for (let k = 0; k < items.length; k += 2) { d[items[k]] = items[k + 1] }
        stack.push(d)
        break
      }
      case OP.FROZENSET: stack.push(new Set(popMark())); break

      case OP.APPEND: { const v = stack.pop(); stack[stack.length - 1].push(v); break }
      case OP.APPENDS: {
        const items = popMark()
        const list = stack[stack.length - 1]
        for (const v of items) { list.push(v) }
        break
      }
      case OP.ADDITEMS: {
        const items = popMark()
        const set = stack[stack.length - 1]
        for (const v of items) { set.add(v) }
        break
      }
      case OP.SETITEM: {
        const v = stack.pop(), k = stack.pop()
        stack[stack.length - 1][k] = v
        break
      }
      case OP.SETITEMS: {
        const items = popMark()
        const d = stack[stack.length - 1]
        for (let k = 0; k < items.length; k += 2) { d[items[k]] = items[k + 1] }
        break
      }

      case OP.MEMOIZE: memo.set(memo.size, stack[stack.length - 1]); break
      case OP.BINPUT: memo.set(u8(), stack[stack.length - 1]); break
      case OP.LONG_BINPUT: memo.set(u32(), stack[stack.length - 1]); break
      case OP.BINGET: case OP.LONG_BINGET: {
        const k = op === OP.BINGET ? u8() : u32()
        if (!memo.has(k)) { throw new UnpickleError(`memo miss for key ${k} at byte ${i}`) }
        stack.push(memo.get(k))
        break
      }

      default:
        throw new UnpickleError(
          `unsupported pickle opcode 0x${op.toString(16)} at byte ${i - 1}`,
        )
    }
  }
}

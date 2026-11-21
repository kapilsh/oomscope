import { create } from 'zustand'

import { unpickle } from './lib/unpickle.js'
import { parseSnapshot } from './lib/snapshot.js'

export const TABS = ['overview', 'timeline', 'history', 'segments', 'allocations', 'structure']

export const useStore = create((set, get) => ({
  model: null,
  fileName: null,
  fileBytes: 0,
  parseMs: 0,
  error: null,
  loading: false,
  tab: 'overview',
  device: 0,
  pool: null, // pool key within the device, or null for all of it
  selectedBlock: null, // {segment, block} from the segment map
  expandedBlame: null, // frame key whose stack is open
  historyAt: null, // event the history view is at, or null for the snapshot

  setTab: (tab) => set({ tab }),
  setDevice: (device) => set({ device, pool: null, selectedBlock: null, expandedBlame: null, historyAt: null }),
  setPool: (pool) => set({ pool, selectedBlock: null, expandedBlame: null, historyAt: null }),
  setHistoryAt: (historyAt) => set({ historyAt }),
  /** Open the allocator's state as it stood after event i. */
  showHistory: (i) => set({ tab: 'history', historyAt: i }),
  selectBlock: (sel) => set({ selectedBlock: sel }),
  toggleBlame: (key) => set({ expandedBlame: get().expandedBlame === key ? null : key }),
  clear: () => set({
    model: null, fileName: null, fileBytes: 0, parseMs: 0,
    error: null, selectedBlock: null, expandedBlame: null, tab: 'overview', device: 0, pool: null, historyAt: null,
  }),

  /** @param {File|{name:string, buffer:ArrayBuffer}} file */
  load: async (file) => {
    set({ loading: true, error: null })
    try {
      const buffer = file.buffer ?? await file.arrayBuffer()
      const t0 = performance.now()
      const raw = unpickle(buffer)
      const model = parseSnapshot(raw)
      const parseMs = performance.now() - t0
      set({
        model,
        fileName: file.name,
        fileBytes: buffer.byteLength,
        parseMs,
        loading: false,
        device: model.devices[0]?.id ?? 0,
        pool: null,
        tab: 'overview',
        selectedBlock: null,
        expandedBlame: null,
        historyAt: null,
      })
    } catch (err) {
      set({ loading: false, error: err.message ?? String(err), model: null })
    }
  },
}))

/** The device the user is looking at, or the first one if that id is gone. */
export function useDevice() {
  return useStore((s) => {
    if (!s.model) { return null }
    return s.model.devices.find((d) => d.id === s.device) ?? s.model.devices[0] ?? null
  })
}

/**
 * What the views render: the whole device, or one pool of it. A pool is
 * shaped like a device, so no view needs to know which it was handed.
 */
export function useView() {
  const device = useDevice()
  const pool = useStore((s) => s.pool)
  if (!device || pool == null) { return device }
  return device.pools.find((p) => p.key === pool) ?? device
}

import { create } from 'zustand'

import { unpickle } from './lib/unpickle.js'
import { parseSnapshot } from './lib/snapshot.js'

export const TABS = ['overview', 'segments', 'allocations', 'timeline']

export const useStore = create((set, get) => ({
  model: null,
  fileName: null,
  fileBytes: 0,
  parseMs: 0,
  error: null,
  loading: false,
  tab: 'overview',
  device: 0,
  selectedBlock: null, // {segment, block} from the segment map
  expandedBlame: null, // frame key whose stack is open

  setTab: (tab) => set({ tab }),
  setDevice: (device) => set({ device, selectedBlock: null }),
  selectBlock: (sel) => set({ selectedBlock: sel }),
  toggleBlame: (key) => set({ expandedBlame: get().expandedBlame === key ? null : key }),
  clear: () => set({
    model: null, fileName: null, fileBytes: 0, parseMs: 0,
    error: null, selectedBlock: null, expandedBlame: null, tab: 'overview', device: 0,
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
        tab: 'overview',
        selectedBlock: null,
        expandedBlame: null,
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

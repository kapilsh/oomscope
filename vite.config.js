import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Deployed to GitHub Pages from docs/ on the default branch.
export default defineConfig({
  base: '/oomscope/',
  plugins: [react()],
  build: { outDir: 'docs', emptyOutDir: true },
})

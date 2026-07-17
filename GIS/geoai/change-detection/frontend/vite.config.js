import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5178,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8005', changeOrigin: true },
      '/output': { target: 'http://127.0.0.1:8005', changeOrigin: true },
    },
  },
})

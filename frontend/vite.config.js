import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/Hack_Havoc_COEP/', // <--- Add this line
  plugins: [
    react(),
    tailwindcss(),
  ],
})
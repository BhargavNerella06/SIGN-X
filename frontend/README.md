# SIGN-X Frontend

React + TypeScript + Tailwind CSS + Vite dashboard for the real-time ISL translator. See the root [`README.md`](../README.md) for full setup.

## Quick start

```powershell
npm install
npm run dev      # dev server on http://localhost:5173, proxies /api and /ws to http://localhost:8000
npm run build    # type-check (tsc -b) + production build to dist/
```

## Layout

- `src/components/` — `CameraView` (getUserMedia capture + landmark overlay), `PredictionPanel`, `ConfidenceBar`, `SentencePanel`, `SystemStats`, `ControlPanel`, `PredictionHistory`, `Navbar`.
- `src/pages/` — `Dashboard` (main real-time translator), `Dataset`, `Model`, `About`.
- `src/services/` — `websocket.ts` (`/ws/predict` client with auto-reconnect), `api.ts` (REST calls).
- `src/types.ts` — TypeScript types mirroring the backend's WebSocket/REST payloads.

Backend URL (build-time env vars): set `VITE_API_BASE_URL` to the backend origin (e.g. `https://signx-api.onrender.com`); the WebSocket URL is derived from it (`https` → `wss`). `VITE_WS_BASE_URL` overrides the WebSocket origin alone. With neither set, REST calls go to the page's own origin (Vite's dev proxy forwards `/api` to `localhost:8000`) and the WebSocket goes to port 8000 on the page's host, using `wss:` when the page is served over https. Speech is played by the browser (`src/services/speech.ts`); the server's engine is only a fallback when the browser has no speech synthesis.

Verified with Node.js 24 / npm 11: `npm install` and `npm run build` both succeed cleanly (see the project's overall build report for details).

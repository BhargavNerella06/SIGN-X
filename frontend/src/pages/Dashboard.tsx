import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import CameraView from "../components/CameraView";
import ControlPanel from "../components/ControlPanel";
import PredictionHistory from "../components/PredictionHistory";
import PredictionPanel from "../components/PredictionPanel";
import SentencePanel from "../components/SentencePanel";
import SystemStats from "../components/SystemStats";
import { initialLiveState, liveReducer } from "../services/liveState";
import { createBrowserSpeech, type BrowserSpeech } from "../services/speech";
import { SignXSocket } from "../services/websocket";
import type { ConnectionStatus, ModelInfo, ServerMessage } from "../types";

export default function Dashboard() {
  const socketRef = useRef<SignXSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [live, dispatch] = useReducer(liveReducer, initialLiveState);
  const { prediction, history } = live;
  const [paused, setPaused] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const speechRef = useRef<BrowserSpeech | null>(null);

  useEffect(() => {
    speechRef.current = createBrowserSpeech();
    return () => speechRef.current?.stop();
  }, []);

  useEffect(() => {
    const socket = new SignXSocket();
    socketRef.current = socket;

    const unsubStatus = socket.onStatusChange(setConnectionStatus);
    const unsubMessage = socket.onMessage((msg: ServerMessage) => {
      // Prediction, history, sentence and frame errors (incl. stale/duplicate
      // commit filtering) are handled by liveReducer.
      dispatch({ kind: "message", msg, time: new Date().toLocaleTimeString() });
      if (msg.type === "connection") {
        setModelInfo(msg.model_info);
        setBanner(msg.vision_available ? null : "MediaPipe model unavailable on the server — see docs/model.md.");
      } else if (msg.type === "status") {
        if (msg.message === "speaking") setIsSpeaking(true);
        if (msg.message === "speech_stopped" || msg.message === "nothing_to_speak") setIsSpeaking(false);
      }
    });

    socket.connect();

    return () => {
      unsubStatus();
      unsubMessage();
      socket.disconnect();
    };
  }, []);

  const handleFrame = useCallback((blob: Blob) => {
    socketRef.current?.sendFrame(blob);
  }, []);

  const canSendFrame = useCallback(() => socketRef.current?.canSendFrame() ?? false, []);

  // Camera switched: the server drops its temporal state (keeping the
  // sentence) and the UI shows UNKNOWN until fresh predictions arrive.
  const handleSourceChange = useCallback(() => {
    const resetSent = socketRef.current?.resetStream() ?? false;
    dispatch({ kind: "source-change", resetSent });
  }, []);

  // Speech plays in the browser (the server may have no audio device). The
  // server's own engine is only a fallback for browsers without speech
  // synthesis, and only audible when the backend runs on this machine.
  const sentence = prediction?.sentence ?? "";
  const handleSpeak = useCallback(() => {
    const speech = speechRef.current;
    setSpeechError(null);
    if (speech?.available) {
      const started = speech.speak(sentence, {
        onEnd: () => setIsSpeaking(false),
        onError: (message) => {
          setIsSpeaking(false);
          setSpeechError(message);
        },
      });
      setIsSpeaking(started);
      return;
    }
    socketRef.current?.speak();
  }, [sentence]);

  const handleStopSpeech = useCallback(() => {
    speechRef.current?.stop();
    setIsSpeaking(false);
    socketRef.current?.stopSpeech();
  }, []);

  const shownBanner = banner ?? live.frameError ?? speechError;

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">SIGN-X</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">Real-Time Indian Sign Language Translator</p>
      </div>

      {shownBanner && (
        <div className="mb-4 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
          {shownBanner}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div className="card p-4">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Live Camera
            </h2>
            <CameraView
              onFrame={handleFrame}
              onSourceChange={handleSourceChange}
              canSendFrame={canSendFrame}
              landmarks={prediction?.landmarks}
              paused={paused}
            />
            <ControlPanel connectionStatus={connectionStatus} prediction={prediction} />
          </div>

          <div className="mt-6">
            <PredictionHistory history={history} onClear={() => dispatch({ kind: "clear-history" })} />
          </div>
        </div>

        <div className="flex flex-col gap-6 lg:col-span-2">
          <PredictionPanel prediction={prediction} />

          <SentencePanel
            sentence={sentence}
            paused={paused}
            isSpeaking={isSpeaking}
            onSpeak={handleSpeak}
            onStop={handleStopSpeech}
            onClear={() => socketRef.current?.clearSentence()}
            onTogglePause={() => setPaused((p) => !p)}
          />

          <SystemStats modelInfo={modelInfo} prediction={prediction} />
        </div>
      </div>
    </div>
  );
}

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import Webcam from "react-webcam";

export default function Demo() {
  const webcamRef = useRef(null);
  const cameraRef = useRef(null);
  const handsRef = useRef(null);
  const pollingRef = useRef(null);

  const [userId, setUserId] = useState(null);
  const [raw, setRaw] = useState("...");
  const [cleaned, setCleaned] = useState("...");
  const [error, setError] = useState(null);

  const BASE_URL = "http://localhost:8000";

  //anonymous user ID
  useEffect(() => {
    let storedId = localStorage.getItem("asl_user_id");
    if (!storedId) {
      storedId = "guest_" + Math.random().toString(36).substring(2, 10);
      localStorage.setItem("asl_user_id", storedId);
    }
    setUserId(storedId);
  }, []);

  const sendKeypoints = async (keypoints) => {
    try {
      await axios.post(`${BASE_URL}/predict`, { keypoints, user_id: userId });
      setError(null);
    } catch (err) {
      setError("Prediction error");
      console.error("Predict error details:", err);
    }
  };

  const pollBuffer = async () => {
    try {
      const res = await axios.get(`${BASE_URL}/buffer`, {
        params: { user_id: userId },
      });
      setRaw(res.data.raw || "...");
      setCleaned(res.data.cleaned || "...");
      setError(null);
    } catch (err) {
      setError("Backend offline");
      console.error("Polling buffer error:", err);
    }
  };

  const resetBuffer = async () => {
    try {
      await axios.post(`${BASE_URL}/reset`, { user_id: userId });
      setRaw("...");
      setCleaned("...");
      setError(null);
    } catch (err) {
      setError("Could not reset buffer.");
      console.error("Reset error:", err);
    }
  };

  const cleanText = async () => {
    try {
      const res = await axios.post(`${BASE_URL}/clean`, { user_id: userId });
      if (res.data.cleaned !== undefined) {
        setCleaned(res.data.cleaned);
        setError(null);
      }
    } catch (err) {
      setError("Could not clean text.");
      console.error("Clean text error:", err);
    }
  };

  useEffect(() => {
    if (!userId) return;

    const onResults = async (results) => {
      if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

      const hand = results.multiHandLandmarks[0];
      const keypoints = hand.flatMap((lm) => [lm.x, lm.y, lm.z]);
      while (keypoints.length < 126) keypoints.push(0.0);
      await sendKeypoints(keypoints);
    };

    if (!window.Hands || !window.Camera) {
      console.error("❌ MediaPipe not available. Include it via CDN in index.html");
      return;
    }

    handsRef.current = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    handsRef.current.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    handsRef.current.onResults(onResults);

    const interval = setInterval(() => {
      if (webcamRef.current?.video?.readyState === 4) {
        if (!cameraRef.current) {
          cameraRef.current = new window.Camera(webcamRef.current.video, {
            onFrame: async () => {
              if (webcamRef.current?.video) {
                await handsRef.current.send({ image: webcamRef.current.video });
              }
            },
            width: 640,
            height: 480,
          });
          cameraRef.current.start();
        }
        clearInterval(interval);
      }
    }, 100);

    pollingRef.current = setInterval(pollBuffer, 1000);

    return () => {
      clearInterval(interval);
      clearInterval(pollingRef.current);
      if (cameraRef.current) cameraRef.current.stop();
      if (handsRef.current?.close) handsRef.current.close();
      cameraRef.current = null;
      handsRef.current = null;
    };
  }, [userId]);

  return (
    <div className="min-h-screen bg-gray-900 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-fuchsia-900 via-slate-900 to-black text-white flex flex-col items-center px-4 py-8">
      <h1 className="text-4xl sm:text-5xl mb-10 font-extrabold tracking-tight neonTitle">
        <span className="pr-3">🤟</span> ASL Translator 
      </h1>

      <div className="w-full max-w-6xl glass border border-fuchsia-900/40 rounded-3xl p-6 md:p-8 shadow-2xl flex flex-col lg:flex-row gap-8">
        <div className="flex-1 overflow-hidden rounded-xl border border-fuchsia-800/30 shadow-xl neonEdge">
          <Webcam
            ref={webcamRef}
            mirrored
            className="w-full h-auto object-cover"
            audio={false}
            videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
          />
        </div>

        <div className="w-full lg:w-96 flex flex-col justify-between gap-6">
          <div>
            <label className="text-xs tracking-wide text-pink-400">RAW BUFFER</label>
            <div className="mt-1 mb-5 bg-black/60 backdrop-blur-sm p-3 font-mono text-pink-200 rounded-md border border-pink-600/40">
              {raw}
            </div>

            <label className="text-xs tracking-wide text-emerald-400">CLEANED SENTENCE</label>
            <div className="mt-1 relative bg-black/80 backdrop-blur-sm p-3 font-semibold text-emerald-200 rounded-md border border-emerald-600/40 typing">
              {cleaned}
              <span className="cursorBlock" />
            </div>
          </div>

          <div className="flex gap-4 pt-2">
            <button
              onClick={resetBuffer}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-fuchsia-600 to-violet-600 hover:brightness-110 transition rounded-md shadow-lg shadow-fuchsia-800/40 font-semibold"
            >
              🔄 Reset
            </button>

            <button
              onClick={cleanText}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-emerald-600 to-green-600 hover:brightness-110 transition rounded-md shadow-lg shadow-emerald-800/40 font-semibold"
            >
              ✨ Clean
            </button>
          </div>

          {error && <p className="text-sm text-red-400 mt-2 font-mono">❌ {error}</p>}
        </div>
      </div>

      <p className="mt-10 text-xs text-white/40">
      © 2025 Group O-12 • FastAPI · PyTorch · React · Tailwind
      </p>
    </div>
  );
}

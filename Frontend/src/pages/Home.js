import React from "react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="text-white min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex flex-col items-center justify-center p-10">
      <h1 className="text-5xl font-extrabold text-center mb-6">
        🌍 Real‑Time ASL Translator
      </h1>
      <p className="text-xl text-center mb-8 max-w-xl text-gray-300">
        Convert American Sign Language into clear, corrected English sentences in real time — right in your browser.
      </p>
      <Link
        to="/Try"
        className="px-8 py-3 bg-pink-600 text-white text-lg rounded-full shadow hover:bg-pink-700 transition"
      >
        🚀 Try It Now
      </Link>
    </div>
  );
}
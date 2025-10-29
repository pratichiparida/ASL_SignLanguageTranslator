import React from "react";

export default function About() {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center py-12 px-6">
      <h2 className="text-3xl font-bold mb-4">About This Project</h2>
      <p className="max-w-2xl text-gray-300 mb-4 text-center">
        The ASL Translator converts American Sign Language into coherent English sentences in real time. It uses:
      </p>
      <ul className="list-disc list-inside text-gray-400 mb-4">
        <li>MediaPipe Hands for landmark detection</li>
        <li>PyTorch MLP for gesture classification</li>
        <li>SymSpell &amp; Gramformer for typo + grammar correction</li>
        <li>FastAPI backend &amp; React + Tailwind frontend</li>
      </ul>
      <p className="text-gray-400">
        Source(Frontend)&nbsp;→&nbsp;
        <a href="https://github.com/BhabeshKumar/asl_web" className="text-blue-400 underline">
        https://github.com/BhabeshKumar/asl_web
        </a>
        
      </p>
      <p className="text-gray-400">
        Source(backend)&nbsp;→&nbsp;
        <a href="https://github.com/BhabeshKumar/asl-backend" className="text-blue-400 underline">
        https://github.com/BhabeshKumar/asl-backend
        </a>
        
      </p>
    </div>
  );
}
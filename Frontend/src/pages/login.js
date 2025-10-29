import React, { useState } from "react";
import { auth, provider } from "../firebase";
import {
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from "firebase/auth";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      if (isRegistering) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      navigate("/try");
    } catch (err) {
      setError(err.message);
    }
  };

  const handleGoogle = async () => {
    try {
      await signInWithPopup(auth, provider);
      navigate("/try");
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 to-gray-800 text-white flex items-center justify-center px-4">
      <div className="bg-white/10 p-8 rounded-xl w-full max-w-md shadow-xl ring-1 ring-white/20 backdrop-blur">
        <h2 className="text-2xl font-bold text-center mb-6">
          {isRegistering ? "Create an Account" : "Sign In"}
        </h2>

        {error && <p className="text-red-400 text-sm mb-4">{error}</p>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            className="w-full px-4 py-2 rounded-md bg-gray-900 border border-white/20 text-white"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <input
            type="password"
            placeholder="Password"
            className="w-full px-4 py-2 rounded-md bg-gray-900 border border-white/20 text-white"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button
            type="submit"
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 rounded-md shadow"
          >
            {isRegistering ? "Register" : "Login"}
          </button>
        </form>

        <p className="text-center mt-4 text-sm">
          {isRegistering ? "Already have an account?" : "New here?"}{" "}
          <button
            onClick={() => setIsRegistering(!isRegistering)}
            className="text-blue-400 underline"
          >
            {isRegistering ? "Sign In" : "Register"}
          </button>
        </p>

        <hr className="my-6 border-white/20" />

        <button
          onClick={handleGoogle}
          className="w-full bg-white text-black font-semibold py-2 rounded-md shadow flex items-center justify-center gap-2 hover:bg-gray-200"
        >
          <img
            src="https://www.svgrepo.com/show/475656/google-color.svg"
            alt="Google"
            className="w-5 h-5"
          />
          Continue with Google
        </button>
      </div>
    </div>
  );
}

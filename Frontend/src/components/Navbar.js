import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <nav className="bg-gray-900 text-white flex items-center justify-between p-4 shadow">
      <div className="text-xl font-bold">
        <Link to="/">🤟 ASL Translator</Link>
      </div>
      <div className="space-x-4">
        <Link to="/" className="hover:text-pink-400">Home</Link>
        <Link to="/about" className="hover:text-pink-400">About</Link>
        <Link to="/try" className="hover:text-pink-400">Translator</Link>
        <Link to="/history" className="hover:text-pink-400">History</Link>
        {user ? (
          <button onClick={handleLogout} className="bg-pink-600 px-4 py-1 rounded hover:bg-pink-700">
            Logout
          </button>
        ) : (
          <Link to="/login" className="bg-blue-600 px-4 py-1 rounded hover:bg-blue-700">
            Login
          </Link>
        )}
      </div>
    </nav>
  );
}

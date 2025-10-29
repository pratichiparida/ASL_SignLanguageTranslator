import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Try from "./pages/Try";
import About from "./pages/About";
import Login from "./pages/login";
import ProtectedRoute from "./ProtectedRoute";
import { AuthProvider } from "./AuthContext";
import History from "./pages/History";


function App() {
  return (
    <AuthProvider>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/try"
            element={
              <ProtectedRoute>
                <Try />
              </ProtectedRoute>
            }
          />
          
          <Route path="/about" element={<About />} />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <History />
              </ProtectedRoute>
            }
          />
          <Route path="/login" element={<Login />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;

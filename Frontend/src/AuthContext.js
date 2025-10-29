import React, { createContext, useContext, useEffect, useState } from "react";
import { auth, provider } from "./firebase";
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from "firebase/auth";


const AuthContext = createContext();


export const useAuth = () => useContext(AuthContext);


export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, setUser);
    return () => unsubscribe();
  }, []);

  
  const loginWithGoogle = () => signInWithPopup(auth, provider);

 
  const logout = () => signOut(auth);

  return (
    <AuthContext.Provider value={{ user, loginWithGoogle, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

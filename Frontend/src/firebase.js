// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyBlwDALriYUYciBbOEKIXri4T5suCznujw",
  authDomain: "signlanguage-3f345.firebaseapp.com",
  projectId: "signlanguage-3f345",
  storageBucket: "signlanguage-3f345.firebasestorage.app",
  messagingSenderId: "501746906925",
  appId: "1:501746906925:web:45603b856b674ec8d985a4",
  measurementId: "G-PQ5PJNLKBL"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();
const db = getFirestore(app);  

export { app, auth, provider, db };

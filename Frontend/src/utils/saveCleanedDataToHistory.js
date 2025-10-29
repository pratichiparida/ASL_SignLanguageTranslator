// src/utils/saveCleanedDataToHistory.js
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { db, auth } from "../firebase";

export async function saveCleanedDataToHistory(cleanedData) {
  const user = auth.currentUser;
  if (!user) {
    console.error("User not authenticated");
    return;
  }

  try {
    await addDoc(collection(db, "users", user.uid, "history"), {
      cleanedData,
      timestamp: serverTimestamp(),
      type: "reset",
    });
  } catch (error) {
    console.error("Failed to save history:", error);
  }
}

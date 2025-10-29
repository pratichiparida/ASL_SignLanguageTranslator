import { useEffect, useState } from "react";
import {
  getFirestore,
  collection,
  getDocs,
  query,
  orderBy,
  limit,
} from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const db = getFirestore();
  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (!user) return;

      try {
        const q = query(
          collection(db, "users", user.uid, "history"),
          orderBy("timestamp", "desc"),
          limit(50)
        );

        const snapshot = await getDocs(q);
        const historyData = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setHistory(historyData);
      } catch (error) {
        console.error("Error fetching history:", error);
      } finally {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">Reset History</h1>
      {loading ? (
        <p className="text-gray-500">Loading history...</p>
      ) : history.length === 0 ? (
        <p className="text-gray-500">No history available.</p>
      ) : (
        <ul className="space-y-4">
          {history.map((item) => (
            <li key={item.id} className="border rounded p-4 bg-white shadow">
              <p className="text-sm text-gray-700 mb-1">
                <strong>Time:</strong>{" "}
                {item.timestamp?.toDate
                  ? item.timestamp.toDate().toLocaleString()
                  : "Unknown"}
              </p>
              <p className="text-sm text-gray-700 mb-2">
                <strong>Type:</strong> {item.type}
              </p>
              <pre className="bg-gray-100 p-2 rounded overflow-x-auto text-sm text-gray-800">
                {JSON.stringify(item.cleanedData, null, 2)}
              </pre>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

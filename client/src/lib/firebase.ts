import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyABw5XHxlMvNros2Bp04idQfD9ROsvc6mc",
  authDomain: "foodconnect-dc7ce.firebaseapp.com",
  projectId: "foodconnect-dc7ce",
  storageBucket: "foodconnect-dc7ce.firebasestorage.app",
  messagingSenderId: "329492075093",
  appId: "1:329492075093:web:54611482890b31991cbadc",
  measurementId: "G-0R0ENB70W6"
};

let app;
try {
  app = initializeApp(firebaseConfig);
} catch (error) {
  console.error('Firebase initialization error:', error);
  // Fallback initialization
  app = initializeApp(firebaseConfig);
}

export const auth = getAuth(app);
export const db = getFirestore(app);
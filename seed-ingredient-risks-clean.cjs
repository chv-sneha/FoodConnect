const admin = require('firebase-admin');
require('dotenv').config();

// Load Firebase credentials from environment variable
const serviceAccount = JSON.parse(process.env.FIREBASE_CREDENTIALS || '{}');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

const ingredientRisks = {
  "potato": {
    "name": "Potato",
    "synonyms": ["potato", "potato starch", "dehydrated potato"],
    "effects": "May cause rash or digestive discomfort in nightshade-sensitive individuals.",
    "severity": "Medium"
  },
  "dairy": {
    "name": "Dairy",
    "synonyms": ["dairy", "milk", "milk solids", "butter", "ghee"],
    "effects": "Causes bloating and cramps in lactose-intolerant individuals.",
    "severity": "High"
  },
  "nuts": {
    "name": "Nuts",
    "synonyms": ["nuts", "peanuts", "tree nuts", "almonds", "cashews"],
    "effects": "Can cause severe allergic reactions including anaphylaxis.",
    "severity": "Critical"
  },
  "gluten": {
    "name": "Gluten",
    "synonyms": ["gluten", "wheat", "barley", "rye", "wheat flour"],
    "effects": "Causes digestive issues in celiac disease and gluten sensitivity.",
    "severity": "High"
  }
};

async function seed() {
  const batch = db.batch();
  Object.keys(ingredientRisks).forEach((id) => {
    const ref = db.collection('ingredientRisks').doc(id);
    batch.set(ref, ingredientRisks[id]);
  });
  await batch.commit();
  console.log('Seeded ingredientRisks collection');
}

seed().catch(console.error);
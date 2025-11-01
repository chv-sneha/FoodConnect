const admin = require('firebase-admin');

// Firebase service account credentials should be loaded from environment variables
// const serviceAccount = require('./firebase-service-account.json'); // Load from secure file
const serviceAccount = {
  type: process.env.FIREBASE_TYPE,
  project_id: process.env.FIREBASE_PROJECT_ID,
  private_key_id: process.env.FIREBASE_PRIVATE_KEY_ID,
  private_key: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
  client_email: process.env.FIREBASE_CLIENT_EMAIL,
  client_id: process.env.FIREBASE_CLIENT_ID,
  auth_uri: process.env.FIREBASE_AUTH_URI,
  token_uri: process.env.FIREBASE_TOKEN_URI,
  auth_provider_x509_cert_url: process.env.FIREBASE_AUTH_PROVIDER_X509_CERT_URL,
  client_x509_cert_url: process.env.FIREBASE_CLIENT_X509_CERT_URL,
  universe_domain: process.env.FIREBASE_UNIVERSE_DOMAIN
};

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
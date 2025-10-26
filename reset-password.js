import bcrypt from 'bcrypt';

async function hashPassword(password) {
  const hash = await bcrypt.hash(password, 10);
  console.log(`Password: ${password}`);
  console.log(`Hashed: ${hash}`);
}

// Hash a simple password for testing
hashPassword('password123');
/**
 * InstantDB API wrapper for meme generator app
 * Handles authentication, CRUD operations for memes, and upvoting
 * 
 * Note: This file expects InstantDB to be loaded via:
 * <script type="importmap"> or import from CDN
 * For now, we'll use a global approach that works with CDN
 */

const APP_ID = '36cac805-1119-4ff6-ba6b-82a36a05f8e0';

// Initialize InstantDB - will be set when InstantDB loads
let db = null;
let InstantDBLib = null; // Will hold the InstantDB library
let authState = { isLoading: true, user: null, error: null };

/**
 * Sets the InstantDB library (called after library loads)
 * @param {Object} lib - InstantDB library object with init, i, id functions
 */
function setInstantDBLib(lib) {
  InstantDBLib = lib;
}

/**
 * Initializes the InstantDB connection and sets up auth listener
 */
async function initDB() {
  // If already initialized, return existing db
  if (db) {
    return db;
  }

  if (!InstantDBLib) {
    // Try to get from window if loaded via script tag
    if (window.InstantDBLib && window.InstantDBLib.init && window.InstantDBLib.i) {
      InstantDBLib = window.InstantDBLib;
    } else {
      throw new Error('InstantDB not loaded. Make sure to include @instantdb/core.');
    }
  }

  const { init, i, id } = InstantDBLib;

  // Define schema
  const schema = i.schema({
    entities: {
      $users: i.entity({
        email: i.string().unique().indexed(),
        username: i.string().optional(),
      }),
      memes: i.entity({
        userId: i.string().indexed(),
        username: i.string(),
        title: i.string(), // title extracted from text boxes
        imageData: i.string(), // base64 encoded image
        textBoxes: i.any(), // array of text box objects
        upvotes: i.number(),
        upvotedBy: i.any(), // array of user IDs
        createdAt: i.date(),
        updatedAt: i.date(),
      }),
    },
    links: {
      memeOwner: {
        forward: { on: 'memes', has: 'one', label: 'owner' },
        reverse: { on: '$users', has: 'many', label: 'memes' },
      },
    },
  });

  // Initialize database
  db = init({ appId: APP_ID, schema });

  // Set up auth listener
  db.subscribeAuth((auth) => {
    console.log('Auth state changed:', auth);
    authState = {
      isLoading: auth.isLoading,
      user: auth.user,
      error: auth.error,
    };
    
    // Dispatch custom event for auth state changes
    window.dispatchEvent(new CustomEvent('authStateChanged', { detail: authState }));
  });

  return db;
}

/**
 * Gets the current user
 * @returns {Promise<Object|null>} Current user or null if not authenticated
 */
async function getCurrentUser() {
  if (!db) {
    await initDB();
  }
  try {
    const auth = await db.getAuth();
    console.log('getAuth() returned:', auth);
    return auth.user;
  } catch (err) {
    console.error('Error getting auth:', err);
    // Fallback to authState
    return authState.user;
  }
}

/**
 * Sends a magic code to the user's email
 * @param {string} email - User's email address
 * @returns {Promise<void>}
 */
async function sendMagicCode(email) {
  if (!db) {
    await initDB();
  }
  if (!db || !db.auth) {
    throw new Error('Database not initialized');
  }
  return db.auth.sendMagicCode({ email });
}

/**
 * Signs in with a magic code
 * @param {string} email - User's email address
 * @param {string} code - Magic code from email
 * @returns {Promise<void>}
 */
async function signInWithMagicCode(email, code) {
  if (!db) {
    await initDB();
  }
  if (!db || !db.auth) {
    throw new Error('Database not initialized');
  }
  console.log('Signing in with magic code for:', email, 'code length:', code.length);
  try {
    const result = await db.auth.signInWithMagicCode({ email, code });
    console.log('Sign in request completed, result:', result);
    // Don't wait here - let the caller wait for auth state change
  } catch (err) {
    console.error('Error in signInWithMagicCode:', err);
    // Check if it's an error object with body
    if (err.body && err.body.message) {
      const error = new Error(err.body.message);
      error.body = err.body;
      throw error;
    }
    throw err;
  }
}

/**
 * Signs out the current user
 * @returns {Promise<void>}
 */
async function signOut() {
  if (!db) {
    return;
  }
  return db.auth.signOut();
}

/**
 * Updates the current user's username
 * @param {string} username - New username
 * @returns {Promise<void>}
 */
async function updateUsername(username) {
  if (!db) {
    throw new Error('Database not initialized');
  }
  const user = await getCurrentUser();
  if (!user) {
    throw new Error('User not authenticated');
  }
  return db.transact(db.tx.$users[user.id].update({ username }));
}

/**
 * Extracts a title from text boxes
 * @param {Array} textBoxes - Array of text box objects
 * @returns {string} Title string
 */
function extractTitleFromTextBoxes(textBoxes) {
  if (!textBoxes || textBoxes.length === 0) {
    return 'Untitled Meme';
  }
  
  // Combine all text box texts, filtering out empty ones
  const texts = textBoxes
    .map(box => box.text && box.text.trim())
    .filter(text => text && text.length > 0);
  
  if (texts.length === 0) {
    return 'Untitled Meme';
  }
  
  // Join with " | " separator, or just use the first one if it's long
  if (texts.length === 1) {
    return texts[0];
  }
  
  // Limit total length to 100 characters
  const combined = texts.join(' | ');
  return combined.length > 100 ? combined.substring(0, 97) + '...' : combined;
}

/**
 * Creates a new meme
 * @param {string} imageData - Base64 encoded image data
 * @param {Array} textBoxes - Array of text box objects
 * @returns {Promise<string>} ID of the created meme
 */
async function createMeme(imageData, textBoxes) {
  if (!db) {
    await initDB();
  }
  if (!InstantDBLib) {
    throw new Error('InstantDB library not loaded');
  }
  const { id } = InstantDBLib;
  
  // Try multiple times to get the user (auth state might be propagating)
  let user = null;
  for (let i = 0; i < 5; i++) {
    user = await getCurrentUser();
    if (user) break;
    
    // Also check auth state directly
    const authState = getAuthState();
    if (authState && authState.user && !authState.isLoading) {
      user = authState.user;
      break;
    }
    
    // Wait a bit before retrying
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  
  if (!user) {
    console.error('No user found after retries. Auth state:', getAuthState());
    throw new Error('User not authenticated. Please sign in again.');
  }

  const memeId = id();
  const username = user.username || user.email.split('@')[0];
  const now = Date.now();
  const title = extractTitleFromTextBoxes(textBoxes);

  await db.transact(
    db.tx.memes[memeId].update({
      userId: user.id,
      username: username,
      title: title,
      imageData: imageData,
      textBoxes: textBoxes,
      upvotes: 0,
      upvotedBy: [],
      createdAt: now,
      updatedAt: now,
    }).link({ owner: user.id })
  );

  return memeId;
}

/**
 * Gets all memes
 * @returns {Promise<Array>} Array of meme objects
 */
async function getMemes() {
  if (!db) {
    await initDB();
  }
  
  return new Promise((resolve, reject) => {
    let resolved = false;
    const unsubscribe = db.subscribeQuery({ memes: {} }, (resp) => {
      if (resp.error) {
        if (!resolved) {
          resolved = true;
          unsubscribe();
          reject(new Error(resp.error.message));
        }
        return;
      }
      if (resp.data && !resolved) {
        resolved = true;
        unsubscribe();
        const memes = resp.data.memes || [];
        // Sort by createdAt descending (newest first)
        memes.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
        resolve(memes);
      }
    });
  });
}

/**
 * Subscribes to memes with a callback
 * @param {Function} callback - Callback function that receives memes array
 * @returns {Function} Unsubscribe function
 */
function subscribeMemes(callback) {
  if (!db) {
    initDB();
  }
  
  return db.subscribeQuery({ memes: {} }, (resp) => {
    if (resp.error) {
      callback(null, new Error(resp.error.message));
      return;
    }
    if (resp.data) {
      const memes = resp.data.memes || [];
      // Sort by createdAt descending (newest first)
      memes.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
      callback(memes, null);
    }
  });
}

/**
 * Updates an existing meme
 * @param {string} memeId - ID of the meme to update
 * @param {string} imageData - Base64 encoded image data
 * @param {Array} textBoxes - Array of text box objects
 * @returns {Promise<void>}
 */
async function updateMeme(memeId, imageData, textBoxes) {
  if (!db) {
    await initDB();
  }
  
  // Try multiple times to get the user (auth state might be propagating)
  let user = null;
  for (let i = 0; i < 5; i++) {
    user = await getCurrentUser();
    if (user) break;
    
    // Also check auth state directly
    const authState = getAuthState();
    if (authState && authState.user && !authState.isLoading) {
      user = authState.user;
      break;
    }
    
    // Wait a bit before retrying
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  
  if (!user) {
    throw new Error('User not authenticated. Please sign in again.');
  }

  const title = extractTitleFromTextBoxes(textBoxes);
  
  await db.transact(
    db.tx.memes[memeId].update({
      title: title,
      imageData: imageData,
      textBoxes: textBoxes,
      updatedAt: Date.now(),
    })
  );
}

/**
 * Deletes a meme
 * @param {string} memeId - ID of the meme to delete
 * @returns {Promise<void>}
 */
async function deleteMeme(memeId) {
  if (!db) {
    await initDB();
  }
  
  // Try multiple times to get the user (auth state might be propagating)
  let user = null;
  for (let i = 0; i < 5; i++) {
    user = await getCurrentUser();
    if (user) break;
    
    // Also check auth state directly
    const authState = getAuthState();
    if (authState && authState.user && !authState.isLoading) {
      user = authState.user;
      break;
    }
    
    // Wait a bit before retrying
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  
  if (!user) {
    throw new Error('User not authenticated. Please sign in again.');
  }

  await db.transact(db.tx.memes[memeId].delete());
}

/**
 * Upvotes a meme (prevents duplicate upvotes)
 * @param {string} memeId - ID of the meme to upvote
 * @returns {Promise<void>}
 */
async function upvoteMeme(memeId) {
  if (!db) {
    await initDB();
  }
  
  // Try multiple times to get the user (auth state might be propagating)
  let user = null;
  for (let i = 0; i < 5; i++) {
    user = await getCurrentUser();
    if (user) break;
    
    // Also check auth state directly
    const authState = getAuthState();
    if (authState && authState.user && !authState.isLoading) {
      user = authState.user;
      break;
    }
    
    // Wait a bit before retrying
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  
  if (!user) {
    throw new Error('User not authenticated. Please sign in again.');
  }

  // Get current meme data
  return new Promise((resolve, reject) => {
    let resolved = false;
    const unsubscribe = db.subscribeQuery({ 
      memes: { 
        $: { 
          where: { id: memeId } 
        } 
      } 
    }, async (resp) => {
      if (resp.error) {
        if (!resolved) {
          resolved = true;
          unsubscribe();
          reject(new Error(resp.error.message));
        }
        return;
      }
      if (resp.data && !resolved) {
        if (resp.data.memes && resp.data.memes.length > 0) {
          resolved = true;
          unsubscribe();
          const meme = resp.data.memes[0];
          const upvotedBy = meme.upvotedBy || [];
          
          // Check if user already upvoted
          if (upvotedBy.includes(user.id)) {
            // Remove upvote
            const newUpvotedBy = upvotedBy.filter(id => id !== user.id);
            const newUpvotes = Math.max(0, meme.upvotes - 1);
            await db.transact(
              db.tx.memes[memeId].update({
                upvotes: newUpvotes,
                upvotedBy: newUpvotedBy,
              })
            );
          } else {
            // Add upvote
            const newUpvotedBy = [...upvotedBy, user.id];
            const newUpvotes = (meme.upvotes || 0) + 1;
            await db.transact(
              db.tx.memes[memeId].update({
                upvotes: newUpvotes,
                upvotedBy: newUpvotedBy,
              })
            );
          }
          resolve();
        } else if (resp.data) {
          // Data loaded but no meme found
          resolved = true;
          unsubscribe();
          reject(new Error('Meme not found'));
        }
      }
    });
  });
}

/**
 * Gets a single meme by ID
 * @param {string} memeId - ID of the meme
 * @returns {Promise<Object>} Meme object
 */
async function getMemeById(memeId) {
  if (!db) {
    await initDB();
  }
  
  return new Promise((resolve, reject) => {
    let resolved = false;
    const unsubscribe = db.subscribeQuery({ 
      memes: { 
        $: { 
          where: { id: memeId } 
        } 
      } 
    }, (resp) => {
      if (resp.error) {
        if (!resolved) {
          resolved = true;
          unsubscribe();
          reject(new Error(resp.error.message));
        }
        return;
      }
      if (resp.data) {
        if (!resolved) {
          resolved = true;
          unsubscribe();
          if (resp.data.memes && resp.data.memes.length > 0) {
            resolve(resp.data.memes[0]);
          } else {
            reject(new Error('Meme not found'));
          }
        }
      }
    });
  });
}

/**
 * Gets the current auth state synchronously
 * @returns {Object} Current auth state
 */
function getAuthState() {
  return authState;
}

// Export functions
window.InstantDB = {
  init: initDB,
  setInstantDBLib,
  getCurrentUser,
  getAuthState,
  sendMagicCode,
  signInWithMagicCode,
  signOut,
  updateUsername,
  createMeme,
  getMemes,
  subscribeMemes,
  updateMeme,
  deleteMeme,
  upvoteMeme,
  getMemeById,
};

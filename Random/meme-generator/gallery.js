/**
 * Gallery logic for browsing and interacting with memes
 * Handles displaying memes, upvoting, editing, and deleting
 */

const memeGrid = document.getElementById('memeGrid');
const loading = document.getElementById('loading');
const emptyState = document.getElementById('emptyState');
const sortSelect = document.getElementById('sortSelect');
const usernameDisplay = document.getElementById('usernameDisplay');
const logoutBtn = document.getElementById('logoutBtn');

let currentUser = null;
let memes = [];
let unsubscribeMemes = null;
let currentSort = 'newest';

/**
 * Checks if user is authenticated and redirects to auth page if not
 */
async function checkAuth() {
  try {
    await window.InstantDB.init();
    
    // Wait a moment for auth state to settle after redirect
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Try multiple times to get the user (auth state might be propagating)
    let user = null;
    for (let i = 0; i < 5; i++) {
      user = await window.InstantDB.getCurrentUser();
      if (user) break;
      
      // Also check auth state directly
      const authState = window.InstantDB.getAuthState();
      if (authState && authState.user && !authState.isLoading) {
        user = authState.user;
        break;
      }
      
      // Wait a bit before retrying
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    if (!user) {
      console.log('No user found after retries, redirecting to auth');
      window.location.href = 'auth.html';
      return;
    }
    
    currentUser = user;
    updateUserDisplay();
    console.log('User authenticated:', user.email);
  } catch (err) {
    console.error('Auth check failed:', err);
    // Don't redirect immediately on error - might be a temporary issue
    // Check auth state one more time
    try {
      const authState = window.InstantDB.getAuthState();
      if (authState && authState.user) {
        currentUser = authState.user;
        updateUserDisplay();
        return;
      }
    } catch (e) {
      console.error('Could not check auth state:', e);
    }
    window.location.href = 'auth.html';
  }
}

/**
 * Updates the username display in the header
 */
function updateUserDisplay() {
  if (currentUser) {
    const username = currentUser.username || currentUser.email.split('@')[0];
    usernameDisplay.textContent = username;
  }
}

/**
 * Handles logout
 */
async function onLogout() {
  try {
    await window.InstantDB.signOut();
    window.location.href = 'auth.html';
  } catch (err) {
    console.error('Logout failed:', err);
    alert('Failed to logout. Please try again.');
  }
}

/**
 * Formats a timestamp into a readable date string
 * @param {number} timestamp - Unix timestamp in milliseconds
 * @returns {string} Formatted date string
 */
function formatDate(timestamp) {
  if (!timestamp) return 'Unknown';
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  
  return date.toLocaleDateString();
}

/**
 * Sorts memes based on current sort option
 * @param {Array} memesArray - Array of memes to sort
 * @returns {Array} Sorted array of memes
 */
function sortMemes(memesArray) {
  const sorted = [...memesArray];
  
  switch (currentSort) {
    case 'newest':
      sorted.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
      break;
    case 'oldest':
      sorted.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
      break;
    case 'upvotes':
      sorted.sort((a, b) => (b.upvotes || 0) - (a.upvotes || 0));
      break;
  }
  
  return sorted;
}

/**
 * Renders a single meme card
 * @param {Object} meme - Meme object
 * @returns {HTMLElement} Meme card element
 */
function renderMemeCard(meme) {
  const card = document.createElement('div');
  card.className = 'meme-card';
  card.dataset.memeId = meme.id;
  
  const isOwner = meme.userId === currentUser.id;
  const hasUpvoted = (meme.upvotedBy || []).includes(currentUser.id);
  
  const title = meme.title || 'Untitled Meme';
  
  card.innerHTML = `
    <div class="meme-title">${escapeHtml(title)}</div>
    <div class="meme-image-container">
      <img src="${meme.imageData}" alt="${escapeHtml(title)}" class="meme-image">
    </div>
    <div class="meme-info">
      <span class="meme-username">${escapeHtml(meme.username)}</span>
      <span class="meme-date">${formatDate(meme.createdAt)}</span>
    </div>
    <div class="meme-actions">
      <div class="upvote-section">
        <button class="upvote-btn ${hasUpvoted ? 'upvoted' : ''}" data-meme-id="${meme.id}">
          ${hasUpvoted ? '▲' : '△'} Upvote
        </button>
        <span class="upvote-count">${meme.upvotes || 0}</span>
      </div>
      ${isOwner ? `
        <div class="meme-owner-actions">
          <button class="btn-edit" onclick="editMeme('${meme.id}')">Edit</button>
          <button class="btn-delete" onclick="deleteMeme('${meme.id}')">Delete</button>
        </div>
      ` : ''}
    </div>
  `;
  
  // Add upvote event listener
  const upvoteBtn = card.querySelector('.upvote-btn');
  upvoteBtn.addEventListener('click', () => handleUpvote(meme.id));
  
  return card;
}

/**
 * Escapes HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Renders all memes in the grid
 */
function renderMemes() {
  if (memes.length === 0) {
    memeGrid.style.display = 'none';
    emptyState.style.display = 'block';
    loading.style.display = 'none';
    return;
  }
  
  emptyState.style.display = 'none';
  loading.style.display = 'none';
  memeGrid.style.display = 'grid';
  
  // Clear existing memes
  memeGrid.innerHTML = '';
  
  // Sort and render
  const sortedMemes = sortMemes(memes);
  sortedMemes.forEach(meme => {
    const card = renderMemeCard(meme);
    memeGrid.appendChild(card);
  });
}

/**
 * Handles upvote button click
 * @param {string} memeId - ID of the meme to upvote
 */
async function handleUpvote(memeId) {
  try {
    await window.InstantDB.upvoteMeme(memeId);
    // The subscription will automatically update the UI
  } catch (err) {
    console.error('Error upvoting meme:', err);
    alert('Failed to upvote meme: ' + (err.message || 'Unknown error'));
  }
}

/**
 * Handles edit button click
 * @param {string} memeId - ID of the meme to edit
 */
function editMeme(memeId) {
  window.location.href = `index.html?edit=${memeId}`;
}

/**
 * Handles delete button click
 * @param {string} memeId - ID of the meme to delete
 */
async function deleteMeme(memeId) {
  if (!confirm('Are you sure you want to delete this meme? This action cannot be undone.')) {
    return;
  }
  
  try {
    await window.InstantDB.deleteMeme(memeId);
    // The subscription will automatically update the UI
  } catch (err) {
    console.error('Error deleting meme:', err);
    alert('Failed to delete meme: ' + (err.message || 'Unknown error'));
  }
}

/**
 * Handles sort select change
 */
function handleSortChange() {
  currentSort = sortSelect.value;
  renderMemes();
}

/**
 * Subscribes to memes updates
 */
function subscribeToMemes() {
  unsubscribeMemes = window.InstantDB.subscribeMemes((memesArray, error) => {
    if (error) {
      console.error('Error loading memes:', error);
      loading.textContent = 'Error loading memes: ' + error.message;
      return;
    }
    
    memes = memesArray || [];
    renderMemes();
  });
}

/**
 * Initializes the gallery
 */
async function init() {
  await checkAuth();
  
  // Show loading state
  loading.style.display = 'block';
  memeGrid.style.display = 'none';
  emptyState.style.display = 'none';
  
  // Subscribe to memes
  subscribeToMemes();
  
  // Listen for auth state changes
  window.addEventListener('authStateChanged', (e) => {
    const authState = e.detail;
    console.log('Auth state changed in gallery.js:', authState);
    
    // Only redirect if auth is fully loaded and there's definitely no user
    // Don't redirect if we're still loading or if we already have a user
    if (!authState.isLoading && !authState.user && !currentUser) {
      console.log('No user in auth state, redirecting to auth page');
      window.location.href = 'auth.html';
    } else if (authState.user) {
      currentUser = authState.user;
      updateUserDisplay();
    }
  });
}

// Event listeners
sortSelect.addEventListener('change', handleSortChange);
logoutBtn.addEventListener('click', onLogout);

// Make edit and delete functions globally available for onclick handlers
window.editMeme = editMeme;
window.deleteMeme = deleteMeme;

// Initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

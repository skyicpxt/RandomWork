/**
 * Authentication logic for meme generator app
 * Handles magic code authentication flow
 */

const emailForm = document.getElementById('emailForm');
const codeForm = document.getElementById('codeForm');
const emailStep = document.getElementById('emailStep');
const codeStep = document.getElementById('codeStep');
const emailInput = document.getElementById('emailInput');
const codeInput = document.getElementById('codeInput');
const sentEmailDisplay = document.getElementById('sentEmailDisplay');
const authError = document.getElementById('authError');
const backToEmailBtn = document.getElementById('backToEmailBtn');

let sentEmail = '';

/**
 * Shows an error message to the user
 * @param {string} message - Error message to display
 */
function showError(message) {
  authError.textContent = message;
  authError.style.display = 'block';
  setTimeout(() => {
    authError.style.display = 'none';
  }, 5000);
}

/**
 * Hides the error message
 */
function hideError() {
  authError.style.display = 'none';
}

/**
 * Handles email form submission
 * @param {Event} e - Form submit event
 */
async function handleEmailSubmit(e) {
  e.preventDefault();
  hideError();
  
  const email = emailInput.value.trim();
  if (!email) {
    showError('Please enter your email address');
    return;
  }

  // Disable form while processing
  const submitBtn = emailForm.querySelector('button[type="submit"]');
  const originalText = submitBtn.textContent;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Sending...';

  try {
    // Ensure InstantDB is initialized
    if (!window.InstantDB) {
      showError('Database not loaded. Please refresh the page.');
      return;
    }
    
    await window.InstantDB.init();
    await window.InstantDB.sendMagicCode(email);
    sentEmail = email;
    sentEmailDisplay.textContent = email;
    emailStep.style.display = 'none';
    codeStep.style.display = 'block';
    codeInput.value = '';
    codeInput.focus();
  } catch (err) {
    console.error('Error sending magic code:', err);
    const errorMsg = err.body?.message || err.message || err.toString() || 'Failed to send verification code. Please try again.';
    showError(errorMsg);
    console.error('Full error:', err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalText;
  }
}

/**
 * Handles code form submission
 * @param {Event} e - Form submit event
 */
async function handleCodeSubmit(e) {
  e.preventDefault();
  hideError();
  
  const code = codeInput.value.trim();
  if (!code) {
    showError('Please enter the verification code');
    return;
  }

  // Disable form while processing
  const submitBtn = codeForm.querySelector('button[type="submit"]');
  const originalText = submitBtn.textContent;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Verifying...';

  try {
    // Ensure InstantDB is initialized
    if (!window.InstantDB) {
      showError('Database not loaded. Please refresh the page.');
      return;
    }
    
    await window.InstantDB.init();
    
    // Set up listener BEFORE signing in to catch the state change
    let authResolved = false;
    let authenticatedUser = null;
    let pollInterval = null;
    
    const authPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (!authResolved) {
          authResolved = true;
          if (pollInterval) clearInterval(pollInterval);
          window.removeEventListener('authStateChanged', checkAuthState);
          // Try one more time to get user
          window.InstantDB.getCurrentUser().then(user => {
            if (user) {
              authenticatedUser = user;
              resolve();
            } else {
              reject(new Error('Timeout waiting for authentication. The code may be invalid or expired.'));
            }
          }).catch(() => {
            reject(new Error('Timeout waiting for authentication. The code may be invalid or expired.'));
          });
        }
      }, 15000); // Increased timeout to 15 seconds
      
      const checkAuthState = (e) => {
        const authState = e.detail;
        console.log('Auth state in listener:', authState);
        
        // Only process if not already resolved
        if (authResolved) return;
        
        // If we have a user and auth is not loading, we're done
        if (authState.user && !authState.isLoading) {
          authResolved = true;
          clearTimeout(timeout);
          if (pollInterval) clearInterval(pollInterval);
          window.removeEventListener('authStateChanged', checkAuthState);
          authenticatedUser = authState.user;
          console.log('User authenticated via event:', authState.user);
          resolve();
        } else if (authState.error && !authState.isLoading) {
          authResolved = true;
          clearTimeout(timeout);
          if (pollInterval) clearInterval(pollInterval);
          window.removeEventListener('authStateChanged', checkAuthState);
          reject(new Error(authState.error.message || 'Authentication failed'));
        } else if (!authState.isLoading && !authState.user) {
          // Auth finished loading but no user - check directly
          console.log('Auth loaded but no user yet, checking directly...');
          // This will be handled by the polling interval
        }
      };
      
      // Also check the current auth state immediately
      const currentAuthState = window.InstantDB.getAuthState();
      console.log('Current auth state before sign in:', currentAuthState);
      if (currentAuthState && currentAuthState.user && !currentAuthState.isLoading) {
        authResolved = true;
        clearTimeout(timeout);
        authenticatedUser = currentAuthState.user;
        console.log('User already authenticated:', currentAuthState.user);
        resolve();
        return; // Don't set up listener if already authenticated
      }
      
      window.addEventListener('authStateChanged', checkAuthState);
      
      // Also poll getCurrentUser and getAuthState as a backup
      let pollCount = 0;
      pollInterval = setInterval(() => {
        if (authResolved) {
          clearInterval(pollInterval);
          return;
        }
        pollCount++;
        if (pollCount > 30) { // Stop after 15 seconds (30 * 500ms)
          clearInterval(pollInterval);
          return;
        }
        
        // Check auth state directly
        const state = window.InstantDB.getAuthState();
        if (state && state.user && !state.isLoading && !authResolved) {
          authResolved = true;
          clearTimeout(timeout);
          clearInterval(pollInterval);
          window.removeEventListener('authStateChanged', checkAuthState);
          authenticatedUser = state.user;
          console.log('User found via state polling:', state.user);
          resolve();
          return;
        }
        
        // Also try getCurrentUser
        window.InstantDB.getCurrentUser().then(user => {
          if (user && !authResolved) {
            authResolved = true;
            clearTimeout(timeout);
            clearInterval(pollInterval);
            window.removeEventListener('authStateChanged', checkAuthState);
            authenticatedUser = user;
            console.log('User found via getCurrentUser polling:', user);
            resolve();
          }
        }).catch(() => {});
      }, 500);
    });
    
    // Now sign in
    console.log('Calling signInWithMagicCode...');
    await window.InstantDB.signInWithMagicCode(sentEmail, code);
    console.log('signInWithMagicCode completed, waiting for auth state...');
    
    // Wait for auth state to update
    await authPromise;
    
    // Verify we have a user
    if (!authenticatedUser) {
      // Final check
      authenticatedUser = await window.InstantDB.getCurrentUser();
    }
    
    if (!authenticatedUser) {
      throw new Error('Sign in appeared successful but user not found. Please try again.');
    }
    
    console.log('Authentication successful, redirecting...', authenticatedUser);
    // Wait a moment to ensure auth state is fully propagated before redirecting
    await new Promise(resolve => setTimeout(resolve, 1000));
    // Redirect to meme creation page on success
    window.location.href = 'index.html';
  } catch (err) {
    console.error('Error signing in:', err);
    const errorMsg = err.body?.message || err.message || err.toString() || 'Invalid verification code. Please try again.';
    showError(errorMsg);
    console.error('Full error:', err);
    codeInput.value = '';
    codeInput.focus();
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalText;
  }
}

/**
 * Handles back button click to return to email step
 */
function handleBackToEmail() {
  codeStep.style.display = 'none';
  emailStep.style.display = 'block';
  sentEmail = '';
  codeInput.value = '';
  emailInput.focus();
}

/**
 * Checks if user is already authenticated and redirects if so
 */
async function checkAuth() {
  try {
    // Wait for InstantDB to be available
    let retries = 0;
    while (!window.InstantDB && retries < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries++;
    }
    
    if (!window.InstantDB) {
      console.error('InstantDB not loaded after waiting');
      return;
    }

    // Initialize InstantDB
    await window.InstantDB.init();
    
    // Wait a bit for auth to settle
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const user = await window.InstantDB.getCurrentUser();
    if (user) {
      // User is already logged in, redirect to meme creation
      window.location.href = 'index.html';
    }
  } catch (err) {
    // User not authenticated, stay on auth page
    console.log('User not authenticated:', err);
  }
}

// Event listeners
emailForm.addEventListener('submit', handleEmailSubmit);
codeForm.addEventListener('submit', handleCodeSubmit);
backToEmailBtn.addEventListener('click', handleBackToEmail);

// Check auth on page load
checkAuth();

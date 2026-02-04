/**
 * Meme Generator: load image, create text boxes anywhere on the image with drag/edit/delete.
 */

const MAX_CANVAS_SIZE = 600;
const canvas = document.getElementById('memeCanvas');
const ctx = canvas.getContext('2d');
const imageInput = document.getElementById('imageInput');
const addTextBtn = document.getElementById('addTextBtn');
const downloadBtn = document.getElementById('downloadBtn');
const postBtn = document.getElementById('postBtn');
const logoutBtn = document.getElementById('logoutBtn');
const usernameDisplay = document.getElementById('usernameDisplay');
const hint = document.getElementById('hint');
const canvasWrap = document.querySelector('.canvas-wrap');
const textBoxEditor = document.getElementById('textBoxEditor');
const textBoxText = document.getElementById('textBoxText');
const textBoxFontSize = document.getElementById('textBoxFontSize');
const textBoxFontSizeValue = document.getElementById('textBoxFontSizeValue');
const textBoxColor = document.getElementById('textBoxColor');
const deleteTextBtn = document.getElementById('deleteTextBtn');

let currentImage = null;
let imageObjectURL = null;
let textBoxes = [];
let selectedTextBoxId = null;
let isDragging = false;
let dragOffset = { x: 0, y: 0 };
let canvasScale = { x: 1, y: 1 };
let currentUser = null;
let editingMemeId = null;

/**
 * Creates a new text box object.
 * @param {number} x - X position on canvas
 * @param {number} y - Y position on canvas
 * @returns {Object} Text box object
 */
function createTextBox(x, y) {
  return {
    id: Date.now() + Math.random(),
    x: x,
    y: y,
    text: 'New text',
    fontSize: 48,
    color: '#ffffff'
  };
}

/**
 * Computes canvas dimensions from image dimensions, capping at MAX_CANVAS_SIZE while keeping aspect ratio.
 * @param {number} imgWidth - Image width
 * @param {number} imgHeight - Image height
 * @returns {{ width: number, height: number }}
 */
function getCanvasSize(imgWidth, imgHeight) {
  if (imgWidth <= MAX_CANVAS_SIZE && imgHeight <= MAX_CANVAS_SIZE) {
    return { width: imgWidth, height: imgHeight };
  }
  const scale = Math.min(MAX_CANVAS_SIZE / imgWidth, MAX_CANVAS_SIZE / imgHeight);
  return {
    width: Math.round(imgWidth * scale),
    height: Math.round(imgHeight * scale),
  };
}

/**
 * Converts screen coordinates to canvas coordinates.
 * @param {number} screenX - Screen X coordinate
 * @param {number} screenY - Screen Y coordinate
 * @returns {{ x: number, y: number }} Canvas coordinates
 */
function screenToCanvas(screenX, screenY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (screenX - rect.left) * scaleX,
    y: (screenY - rect.top) * scaleY
  };
}

/**
 * Finds a text box at the given canvas coordinates.
 * @param {number} x - Canvas X coordinate
 * @param {number} y - Canvas Y coordinate
 * @returns {Object|null} Text box or null
 */
function findTextBoxAt(x, y) {
  for (let i = textBoxes.length - 1; i >= 0; i--) {
    const box = textBoxes[i];
    ctx.font = `bold ${box.fontSize}px Impact, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const metrics = ctx.measureText(box.text);
    const textWidth = metrics.width;
    const textHeight = box.fontSize;
    const halfWidth = textWidth / 2;
    const halfHeight = textHeight / 2;
    
    if (x >= box.x - halfWidth && x <= box.x + halfWidth &&
        y >= box.y - halfHeight && y <= box.y + halfHeight) {
      return box;
    }
  }
  return null;
}

/**
 * Draws the current image and all text boxes onto the canvas. No-op if no image is loaded.
 */
function draw() {
  if (!currentImage) return;

  const img = currentImage;
  const { width, height } = getCanvasSize(img.naturalWidth, img.naturalHeight);

  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(img, 0, 0, width, height);

  ctx.textAlign = 'center';
  ctx.strokeStyle = '#000';

  // Draw all text boxes
  textBoxes.forEach(box => {
    const isSelected = box.id === selectedTextBoxId;
    
    ctx.font = `bold ${box.fontSize}px Impact, sans-serif`;
    ctx.lineWidth = Math.max(2, box.fontSize / 8);
    ctx.fillStyle = box.color;
    ctx.textBaseline = 'middle';
    
    ctx.strokeText(box.text, box.x, box.y);
    ctx.fillText(box.text, box.x, box.y);
    
    // Draw selection indicator only when selected
    if (isSelected) {
      ctx.strokeStyle = '#00aaff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      const metrics = ctx.measureText(box.text);
      const textWidth = metrics.width;
      const textHeight = box.fontSize;
      const padding = 8;
      ctx.strokeRect(
        box.x - textWidth / 2 - padding,
        box.y - textHeight / 2 - padding,
        textWidth + padding * 2,
        textHeight + padding * 2
      );
      ctx.setLineDash([]);
      ctx.strokeStyle = '#000';
    }
  });
}

/**
 * Updates the editor panel with the selected text box's properties.
 */
function updateEditor() {
  if (selectedTextBoxId === null) {
    textBoxEditor.style.display = 'none';
    return;
  }
  
  const box = textBoxes.find(b => b.id === selectedTextBoxId);
  if (!box) {
    textBoxEditor.style.display = 'none';
    return;
  }
  
  textBoxEditor.style.display = 'block';
  textBoxText.value = box.text;
  textBoxFontSize.value = box.fontSize;
  textBoxFontSizeValue.textContent = box.fontSize;
  textBoxColor.value = box.color;
}

/**
 * Handles canvas click events to add or select text boxes.
 * @param {MouseEvent} e - Mouse event
 */
function onCanvasClick(e) {
  if (!currentImage) return;
  
  const { x, y } = screenToCanvas(e.clientX, e.clientY);
  const clickedBox = findTextBoxAt(x, y);
  
  if (clickedBox) {
    selectedTextBoxId = clickedBox.id;
    updateEditor();
    draw();
  } else {
    // Deselect when clicking outside
    selectedTextBoxId = null;
    updateEditor();
    draw();
  }
}

/**
 * Handles canvas mouse down for dragging.
 * @param {MouseEvent} e - Mouse event
 */
function onCanvasMouseDown(e) {
  if (!currentImage) return;
  
  const { x, y } = screenToCanvas(e.clientX, e.clientY);
  const clickedBox = findTextBoxAt(x, y);
  
  if (clickedBox) {
    isDragging = true;
    selectedTextBoxId = clickedBox.id;
    dragOffset.x = x - clickedBox.x;
    dragOffset.y = y - clickedBox.y;
    updateEditor();
    canvas.style.cursor = 'grabbing';
  }
}

/**
 * Handles canvas mouse move for dragging.
 * @param {MouseEvent} e - Mouse event
 */
function onCanvasMouseMove(e) {
  if (!currentImage) return;
  
  const { x, y } = screenToCanvas(e.clientX, e.clientY);
  
  if (isDragging && selectedTextBoxId !== null) {
    const box = textBoxes.find(b => b.id === selectedTextBoxId);
    if (box) {
      box.x = x - dragOffset.x;
      box.y = y - dragOffset.y;
      draw();
    }
  } else {
    // Update cursor
    const hoveredBox = findTextBoxAt(x, y);
    canvas.style.cursor = hoveredBox ? 'grab' : 'crosshair';
  }
}

/**
 * Handles canvas mouse up to stop dragging.
 */
function onCanvasMouseUp() {
  isDragging = false;
  canvas.style.cursor = 'crosshair';
}

/**
 * Handles canvas mouse leave to stop dragging.
 */
function onCanvasMouseLeave() {
  isDragging = false;
  canvas.style.cursor = 'crosshair';
}

/**
 * Loads the selected image file, revokes any previous object URL, and redraws.
 * @param {Event} e - Input change event
 */
function onImageChange(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  if (imageObjectURL) {
    URL.revokeObjectURL(imageObjectURL);
    imageObjectURL = null;
  }

  imageObjectURL = URL.createObjectURL(file);
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    currentImage = img;
    canvasWrap.classList.add('has-image');
    addTextBtn.disabled = false;
    downloadBtn.disabled = false;
    if (currentUser) {
      postBtn.disabled = false;
    }
    textBoxes = [];
    selectedTextBoxId = null;
    editingMemeId = null; // Reset edit mode when loading new image
    postBtn.textContent = 'Post Meme';
    updateEditor();
    draw();
  };
  img.onerror = () => {
    currentImage = null;
    addTextBtn.disabled = true;
    downloadBtn.disabled = true;
    if (imageObjectURL) {
      URL.revokeObjectURL(imageObjectURL);
      imageObjectURL = null;
    }
  };
  img.src = imageObjectURL;
}

/**
 * Adds a new text box at the center of the canvas.
 */
function onAddText() {
  if (!currentImage) return;
  const { width, height } = getCanvasSize(currentImage.naturalWidth, currentImage.naturalHeight);
  const newBox = createTextBox(width / 2, height / 2);
  textBoxes.push(newBox);
  selectedTextBoxId = newBox.id;
  updateEditor();
  draw();
}

/**
 * Updates the selected text box's text and redraws.
 */
function onTextBoxTextChange() {
  if (selectedTextBoxId === null) return;
  const box = textBoxes.find(b => b.id === selectedTextBoxId);
  if (box) {
    box.text = textBoxText.value;
    draw();
  }
}

/**
 * Updates the selected text box's font size and redraws.
 */
function onTextBoxFontSizeChange() {
  if (selectedTextBoxId === null) return;
  const box = textBoxes.find(b => b.id === selectedTextBoxId);
  if (box) {
    box.fontSize = Number(textBoxFontSize.value);
    textBoxFontSizeValue.textContent = box.fontSize;
    draw();
  }
}

/**
 * Updates the selected text box's color and redraws.
 */
function onTextBoxColorChange() {
  if (selectedTextBoxId === null) return;
  const box = textBoxes.find(b => b.id === selectedTextBoxId);
  if (box) {
    box.color = textBoxColor.value;
    draw();
  }
}

/**
 * Deletes the selected text box.
 */
function onDeleteTextBox() {
  if (selectedTextBoxId === null) return;
  textBoxes = textBoxes.filter(b => b.id !== selectedTextBoxId);
  selectedTextBoxId = null;
  updateEditor();
  draw();
}

/**
 * Triggers a download of the canvas as meme.png.
 */
function onDownload() {
  if (!currentImage) return;
  const link = document.createElement('a');
  link.download = 'meme.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

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
 * Posts the current meme to the gallery
 */
async function onPostMeme() {
  if (!currentImage) {
    alert('Please choose an image first');
    return;
  }

  // Verify we have a user before proceeding
  if (!currentUser) {
    // Try to get user one more time
    try {
      currentUser = await window.InstantDB.getCurrentUser();
      if (!currentUser) {
        const authState = window.InstantDB.getAuthState();
        if (authState && authState.user) {
          currentUser = authState.user;
        } else {
          alert('You must be signed in to post memes. Redirecting to login...');
          window.location.href = 'auth.html';
          return;
        }
      }
    } catch (err) {
      console.error('Error getting user:', err);
      alert('You must be signed in to post memes. Redirecting to login...');
      window.location.href = 'auth.html';
      return;
    }
  }

  try {
    postBtn.disabled = true;
    postBtn.textContent = 'Posting...';

    // Convert canvas to base64
    const imageData = canvas.toDataURL('image/png');

    if (editingMemeId) {
      // Update existing meme
      await window.InstantDB.updateMeme(editingMemeId, imageData, textBoxes);
      alert('Meme updated successfully!');
      window.location.href = 'gallery.html';
    } else {
      // Create new meme
      await window.InstantDB.createMeme(imageData, textBoxes);
      alert('Meme posted successfully!');
      // Optionally redirect to gallery
      const goToGallery = confirm('Meme posted! Would you like to go to the gallery?');
      if (goToGallery) {
        window.location.href = 'gallery.html';
      }
    }
  } catch (err) {
    console.error('Error posting meme:', err);
    alert('Failed to post meme: ' + (err.message || 'Unknown error'));
  } finally {
    postBtn.disabled = false;
    postBtn.textContent = editingMemeId ? 'Update Meme' : 'Post Meme';
  }
}

/**
 * Loads a meme for editing
 * @param {string} memeId - ID of the meme to edit
 */
async function loadMemeForEdit(memeId) {
  try {
    const meme = await window.InstantDB.getMemeById(memeId);
    
    // Check if user owns this meme
    if (meme.userId !== currentUser.id) {
      alert('You can only edit your own memes');
      window.location.href = 'gallery.html';
      return;
    }

    // Load image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      currentImage = img;
      canvasWrap.classList.add('has-image');
      addTextBtn.disabled = false;
      downloadBtn.disabled = false;
      postBtn.disabled = false;
      
      // Load text boxes
      textBoxes = meme.textBoxes || [];
      selectedTextBoxId = null;
      updateEditor();
      draw();
    };
    img.onerror = () => {
      alert('Failed to load meme image');
    };
    img.src = meme.imageData;
    
    editingMemeId = memeId;
    postBtn.textContent = 'Update Meme';
  } catch (err) {
    console.error('Error loading meme:', err);
    alert('Failed to load meme: ' + (err.message || 'Unknown error'));
    window.location.href = 'gallery.html';
  }
}

/**
 * Checks URL parameters for edit mode
 */
function checkEditMode() {
  const urlParams = new URLSearchParams(window.location.search);
  const editId = urlParams.get('edit');
  if (editId) {
    loadMemeForEdit(editId);
  }
}

// Initialize app
async function init() {
  await checkAuth();
  checkEditMode();
  
  // Listen for auth state changes
  window.addEventListener('authStateChanged', (e) => {
    const authState = e.detail;
    console.log('Auth state changed in app.js:', authState);
    
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
imageInput.addEventListener('change', onImageChange);
addTextBtn.addEventListener('click', onAddText);
canvas.addEventListener('click', onCanvasClick);
canvas.addEventListener('mousedown', onCanvasMouseDown);
canvas.addEventListener('mousemove', onCanvasMouseMove);
canvas.addEventListener('mouseup', onCanvasMouseUp);
canvas.addEventListener('mouseleave', onCanvasMouseLeave);
textBoxText.addEventListener('input', onTextBoxTextChange);
textBoxFontSize.addEventListener('input', onTextBoxFontSizeChange);
textBoxColor.addEventListener('input', onTextBoxColorChange);
deleteTextBtn.addEventListener('click', onDeleteTextBox);
downloadBtn.addEventListener('click', onDownload);
postBtn.addEventListener('click', onPostMeme);
logoutBtn.addEventListener('click', onLogout);

// Initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

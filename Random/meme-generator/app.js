/**
 * Meme Generator: load image, create text boxes anywhere on the image with drag/edit/delete.
 */

const MAX_CANVAS_SIZE = 600;
const canvas = document.getElementById('memeCanvas');
const ctx = canvas.getContext('2d');
const imageInput = document.getElementById('imageInput');
const addTextBtn = document.getElementById('addTextBtn');
const downloadBtn = document.getElementById('downloadBtn');
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
    textBoxes = [];
    selectedTextBoxId = null;
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

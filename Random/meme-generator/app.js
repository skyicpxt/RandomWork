/**
 * Meme Generator: load image, draw top/bottom text (white with black outline), download as PNG.
 */

const MAX_CANVAS_SIZE = 600;
const canvas = document.getElementById('memeCanvas');
const ctx = canvas.getContext('2d');
const imageInput = document.getElementById('imageInput');
const topTextInput = document.getElementById('topText');
const bottomTextInput = document.getElementById('bottomText');
const topFontSizeInput = document.getElementById('topFontSize');
const topFontSizeValue = document.getElementById('topFontSizeValue');
const topFontColorInput = document.getElementById('topFontColor');
const bottomFontSizeInput = document.getElementById('bottomFontSize');
const bottomFontSizeValue = document.getElementById('bottomFontSizeValue');
const bottomFontColorInput = document.getElementById('bottomFontColor');
const downloadBtn = document.getElementById('downloadBtn');
const hint = document.getElementById('hint');
const canvasWrap = document.querySelector('.canvas-wrap');

let currentImage = null;
let imageObjectURL = null;

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
 * Draws the current image and text onto the canvas. No-op if no image is loaded.
 */
function draw() {
  if (!currentImage) return;

  const img = currentImage;
  const { width, height } = getCanvasSize(img.naturalWidth, img.naturalHeight);

  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(img, 0, 0, width, height);

  const topText = topTextInput.value.trim();
  const bottomText = bottomTextInput.value.trim();
  const x = width / 2;

  ctx.textAlign = 'center';
  ctx.strokeStyle = '#000';

  if (topText) {
    const topFontSize = Number(topFontSizeInput.value);
    const topPadding = topFontSize * 0.5;
    ctx.font = `bold ${topFontSize}px Impact, sans-serif`;
    ctx.lineWidth = Math.max(2, topFontSize / 8);
    ctx.fillStyle = topFontColorInput.value;
    ctx.textBaseline = 'top';
    ctx.strokeText(topText, x, topPadding);
    ctx.fillText(topText, x, topPadding);
  }

  if (bottomText) {
    const bottomFontSize = Number(bottomFontSizeInput.value);
    const bottomPadding = bottomFontSize * 0.5;
    ctx.font = `bold ${bottomFontSize}px Impact, sans-serif`;
    ctx.lineWidth = Math.max(2, bottomFontSize / 8);
    ctx.fillStyle = bottomFontColorInput.value;
    ctx.textBaseline = 'bottom';
    ctx.strokeText(bottomText, x, height - bottomPadding);
    ctx.fillText(bottomText, x, height - bottomPadding);
  }
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
    downloadBtn.disabled = false;
    draw();
  };
  img.onerror = () => {
    currentImage = null;
    downloadBtn.disabled = true;
    if (imageObjectURL) {
      URL.revokeObjectURL(imageObjectURL);
      imageObjectURL = null;
    }
  };
  img.src = imageObjectURL;
}

/**
 * Updates the top font size display label and redraws.
 */
function onTopFontSizeChange() {
  topFontSizeValue.textContent = topFontSizeInput.value;
  draw();
}

/**
 * Updates the bottom font size display label and redraws.
 */
function onBottomFontSizeChange() {
  bottomFontSizeValue.textContent = bottomFontSizeInput.value;
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

imageInput.addEventListener('change', onImageChange);
topTextInput.addEventListener('input', draw);
bottomTextInput.addEventListener('input', draw);
topFontSizeInput.addEventListener('input', onTopFontSizeChange);
topFontColorInput.addEventListener('input', draw);
bottomFontSizeInput.addEventListener('input', onBottomFontSizeChange);
bottomFontColorInput.addEventListener('input', draw);
downloadBtn.addEventListener('click', onDownload);

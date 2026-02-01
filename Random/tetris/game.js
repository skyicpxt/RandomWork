const COLS = 10;
const ROWS = 20;
const BLOCK_SIZE = 30;

const SHAPES = [
  [[1, 1, 1, 1]], // I
  [[1, 1], [1, 1]], // O
  [[0, 1, 0], [1, 1, 1]], // T
  [[0, 1, 1], [1, 1, 0]], // S
  [[1, 1, 0], [0, 1, 1]], // Z
  [[1, 0, 0], [1, 1, 1]], // J
  [[0, 0, 1], [1, 1, 1]], // L
];

const COLORS = [
  '#00f0f0', // I - cyan
  '#f0f000', // O - yellow
  '#a000f0', // T - purple
  '#00f000', // S - green
  '#f00000', // Z - red
  '#0000f0', // J - blue
  '#f0a000', // L - orange
];

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');

let board;
let currentPiece;
let currentColor;
let currentX;
let currentY;
let score;
let gameOver;
let dropInterval;
const DROP_MS = 800;

function initBoard() {
  board = Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
}

function drawBlock(ctx, x, y, color) {
  const padding = 1;
  ctx.fillStyle = color;
  ctx.fillRect(x * BLOCK_SIZE + padding, y * BLOCK_SIZE + padding, BLOCK_SIZE - padding, BLOCK_SIZE - padding);
}

function drawBoard() {
  ctx.fillStyle = '#0f0f1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      if (board[row][col]) {
        drawBlock(ctx, col, row, board[row][col]);
      }
    }
  }

  if (currentPiece && !gameOver) {
    for (let row = 0; row < currentPiece.length; row++) {
      for (let col = 0; col < currentPiece[row].length; col++) {
        if (currentPiece[row][col]) {
          drawBlock(ctx, currentX + col, currentY + row, currentColor);
        }
      }
    }
  }
}

function randomPiece() {
  const i = Math.floor(Math.random() * SHAPES.length);
  return { shape: SHAPES[i].map(row => [...row]), color: COLORS[i] };
}

function collision(piece, x, y) {
  for (let row = 0; row < piece.length; row++) {
    for (let col = 0; col < piece[row].length; col++) {
      if (!piece[row][col]) continue;
      const newX = x + col;
      const newY = y + row;
      if (newX < 0 || newX >= COLS || newY >= ROWS) return true;
      if (newY >= 0 && board[newY][newX]) return true;
    }
  }
  return false;
}

function spawnPiece() {
  const { shape, color } = randomPiece();
  currentPiece = shape;
  currentColor = color;
  currentX = Math.floor((COLS - shape[0].length) / 2);
  currentY = 0;

  if (collision(currentPiece, currentX, currentY)) {
    gameOver = true;
    clearInterval(dropInterval);
    scoreEl.textContent = 'Game Over! Score: ' + score;
  }
}

function mergePiece() {
  for (let row = 0; row < currentPiece.length; row++) {
    for (let col = 0; col < currentPiece[row].length; col++) {
      if (currentPiece[row][col]) {
        const y = currentY + row;
        const x = currentX + col;
        if (y >= 0) board[y][x] = currentColor;
      }
    }
  }
  clearLines();
  spawnPiece();
}

function clearLines() {
  let linesCleared = 0;
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row].every(cell => cell !== 0)) {
      board.splice(row, 1);
      board.unshift(Array(COLS).fill(0));
      linesCleared++;
      row++;
    }
  }
  if (linesCleared > 0) {
    score += linesCleared * 100;
    scoreEl.textContent = 'Score: ' + score;
  }
}

function rotatePiece() {
  if (!currentPiece || gameOver) return;
  const rows = currentPiece.length;
  const cols = currentPiece[0].length;
  const rotated = Array(cols).fill(null).map(() => Array(rows).fill(0));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      rotated[c][rows - 1 - r] = currentPiece[r][c];
    }
  }
  if (!collision(rotated, currentX, currentY)) {
    currentPiece = rotated;
  } else {
    if (!collision(rotated, currentX - 1, currentY)) {
      currentX--;
      currentPiece = rotated;
    } else if (!collision(rotated, currentX + 1, currentY)) {
      currentX++;
      currentPiece = rotated;
    }
  }
}

function moveLeft() {
  if (!currentPiece || gameOver) return;
  if (!collision(currentPiece, currentX - 1, currentY)) currentX--;
}

function moveRight() {
  if (!currentPiece || gameOver) return;
  if (!collision(currentPiece, currentX + 1, currentY)) currentX++;
}

function moveDown() {
  if (!currentPiece || gameOver) return;
  if (collision(currentPiece, currentX, currentY + 1)) {
    mergePiece();
  } else {
    currentY++;
  }
}

function gameLoop() {
  if (!gameOver) {
    moveDown();
    drawBoard();
  }
}

function handleKey(e) {
  if (gameOver) return;
  switch (e.code) {
    case 'ArrowLeft':
    case 'KeyA':
      e.preventDefault();
      moveLeft();
      break;
    case 'ArrowRight':
    case 'KeyD':
      e.preventDefault();
      moveRight();
      break;
    case 'ArrowDown':
    case 'KeyS':
      e.preventDefault();
      moveDown();
      break;
    case 'ArrowUp':
    case 'KeyW':
      e.preventDefault();
      rotatePiece();
      break;
    default:
      return;
  }
  drawBoard();
}

function start() {
  initBoard();
  score = 0;
  gameOver = false;
  scoreEl.textContent = 'Score: 0';
  spawnPiece();
  drawBoard();
  dropInterval = setInterval(gameLoop, DROP_MS);
  canvas.focus();
}

document.addEventListener('keydown', handleKey);
start();

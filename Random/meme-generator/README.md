# Meme Generator - Full Stack App

A full-stack meme generator application where users can create memes with custom text overlays, post them to a gallery, and upvote their favorites.

## Features

- **User Authentication**: Email-based magic code authentication via InstantDB
- **Meme Creation**: Upload images and add customizable text boxes with:
  - Drag-and-drop positioning
  - Customizable font size (16-120px)
  - Color picker for text color
  - Multiple text boxes per meme
- **Gallery**: Browse all posted memes with:
  - Sort by newest, oldest, or most upvoted
  - Upvote/downvote memes (one vote per user)
  - View meme details (creator, timestamp, upvote count)
- **User Management**: 
  - Edit your own memes
  - Delete your own memes
  - View username/email in header

## Technology Stack

- **Frontend**: Vanilla HTML, CSS, JavaScript
- **Backend**: InstantDB (Backend-as-a-Service)
- **Database**: InstantDB (App ID: `36cac805-1119-4ff6-ba6b-82a36a05f8e0`)

## Setup Instructions

### Prerequisites

- A web server (or use a local development server)
- An InstantDB account (already configured with App ID)

### Installation

1. **Clone or download this repository**

2. **Set up InstantDB Schema**

   You need to configure the database schema in your InstantDB dashboard. The schema includes:

   **Entities:**
   - `$users`: 
     - `email` (string, unique, indexed)
     - `username` (string, optional)
   - `memes`:
     - `userId` (string, indexed)
     - `username` (string)
     - `imageData` (string) - base64 encoded image
     - `textBoxes` (any) - array of text box objects
     - `upvotes` (number)
     - `upvotedBy` (any) - array of user IDs
     - `createdAt` (date)
     - `updatedAt` (date)

   **Links:**
   - `memeOwner`: Links memes to users (many-to-one)

3. **Set up Permissions**

   Configure InstantDB permissions to allow:
   - Users to create their own memes
   - Users to update/delete their own memes
   - All users to read all memes
   - Users to upvote memes

4. **Run the Application**

   Open `index.html` in a web browser, or use a local server:

   ```bash
   # Using Python 3
   python -m http.server 8000

   # Using Node.js (http-server)
   npx http-server

   # Using PHP
   php -S localhost:8000
   ```

   Then navigate to `http://localhost:8000` in your browser.

## File Structure

```
meme-generator/
├── index.html          # Main meme creation page
├── app.js             # Meme creation logic and canvas manipulation
├── styles.css         # Styles for meme creation page
├── auth.html          # Login/signup page
├── auth.js            # Authentication logic (magic code flow)
├── gallery.html       # Gallery page for browsing memes
├── gallery.js         # Gallery logic (display, upvote, edit, delete)
├── api.js             # InstantDB API wrapper
├── shared.css         # Shared styles across all pages
└── README.md          # This file
```

## How to Use

### Creating a Meme

1. **Sign In**: 
   - Go to `auth.html` (or you'll be redirected if not signed in)
   - Enter your email address
   - Check your email for the verification code
   - Enter the code to sign in

2. **Create Your Meme**:
   - Click "Choose image" to upload an image
   - Click "Add text box" to add text overlays
   - Click on the canvas to position text boxes
   - Drag text boxes to reposition them
   - Select a text box to edit its properties:
     - Change the text content
     - Adjust font size (16-120px)
     - Change text color
     - Delete the text box
   - Click "Download meme" to save locally
   - Click "Post Meme" to share it to the gallery

### Browsing the Gallery

1. Navigate to the Gallery page
2. Browse memes in the grid layout
3. Use the sort dropdown to filter by:
   - Newest First
   - Oldest First
   - Most Upvoted
4. Click the upvote button to upvote/downvote memes
5. View meme details (creator, timestamp, upvote count)

### Editing Your Memes

1. Go to the Gallery
2. Find one of your memes (you'll see Edit/Delete buttons)
3. Click "Edit" to modify the meme
4. Make your changes
5. Click "Update Meme" to save changes

### Deleting Your Memes

1. Go to the Gallery
2. Find one of your memes
3. Click "Delete"
4. Confirm the deletion

## API Reference

The `api.js` file provides the following functions:

### Authentication
- `init()` - Initialize InstantDB connection
- `getCurrentUser()` - Get currently authenticated user
- `sendMagicCode(email)` - Send magic code to email
- `signInWithMagicCode(email, code)` - Sign in with magic code
- `signOut()` - Sign out current user
- `getAuthState()` - Get current auth state synchronously

### Memes
- `createMeme(imageData, textBoxes)` - Create a new meme
- `getMemes()` - Get all memes (one-time query)
- `subscribeMemes(callback)` - Subscribe to meme updates
- `getMemeById(memeId)` - Get a single meme by ID
- `updateMeme(memeId, imageData, textBoxes)` - Update an existing meme
- `deleteMeme(memeId)` - Delete a meme
- `upvoteMeme(memeId)` - Toggle upvote on a meme

## Troubleshooting

### Authentication Issues

- **"Database not loaded"**: Make sure InstantDB library loads from CDN. Check browser console for errors.
- **Sign-in loop**: The app includes retry logic. If you still experience issues, check:
  - Browser console for error messages
  - That InstantDB schema is properly configured
  - That permissions allow user authentication

### Meme Display Issues

- **Images not showing**: Check that `imageData` is properly base64 encoded
- **Text boxes not rendering**: Ensure `textBoxes` array contains valid objects with `x`, `y`, `text`, `fontSize`, and `color` properties

### Performance

- Large images are automatically resized to max 600px (maintaining aspect ratio)
- Base64 image data can be large - consider implementing image compression or using InstantDB Storage for production

## Development Notes

- The app uses ES6 modules for loading InstantDB from CDN
- All authentication state is managed through InstantDB's built-in auth system
- Real-time updates are handled via InstantDB subscriptions
- The app includes retry logic for authentication to handle timing issues

## Future Enhancements

Potential improvements:
- Image compression before upload
- User profiles and avatars
- Comments on memes
- Meme categories/tags
- Search functionality
- Pagination for large galleries
- Image filters/effects
- Social sharing

## License

This project is open source and available for personal and educational use.

## Support

For issues related to:
- **InstantDB**: Check [InstantDB Documentation](https://www.instantdb.com/docs)
- **App-specific issues**: Check browser console for error messages and ensure all files are properly loaded

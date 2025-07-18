# Vision - Object Detection

Point your camera at things. AI draws boxes around them. It's pretty fast.

**Live Demo**: [https://svetimfm.github.io/vision-capture-app/](https://svetimfm.github.io/vision-capture-app/)

## What it does

- Detects up to 40 objects in real-time
- Takes photos with AI overlays
- Records videos with live detection
- Works entirely in your browser
- **100% client-side** - no data leaves your device

## Privacy & Security

Everything happens in your browser. We don't see your camera. We don't store your photos. We don't track what you detect. Your data never leaves your device.

The app uses TensorFlow.js to run a neural network locally. No servers, no uploads, no privacy concerns.

## Features

- 📸 **Snap** - High-res photos with detection overlays
- 🎥 **Record** - Videos with live object tracking
- 🎨 **Customize** - Colors, confidence thresholds, font sizes
- 🔄 **Smooth** - Interpolated animations between frames
- 📱 **Responsive** - Works on phones, tablets, desktops
- 🔒 **Private** - All processing happens on your device

## Tech Stack

- React
- TensorFlow.js with COCO-SSD model
- Tailwind CSS
- WebRTC for camera access
- Canvas API for overlays

## Local Development

```bash
# Clone the repo
git clone https://github.com/SvetimFM/vision-capture-app.git
cd vision-capture-app

# Install dependencies
npm install

# Start dev server
npm start

# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

## Browser Support

Works best in:
- Chrome/Edge 80+
- Safari 14+
- Firefox 78+

Needs camera permissions and a decent GPU for smooth performance.

## License

MIT - Do whatever you want with it.

---

Made with minimal fuss. [Report issues here](https://github.com/SvetimFM/vision-capture-app/issues).
import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const VisionCapture = () => {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [facingMode, setFacingMode] = useState('environment');
  const [showSettings, setShowSettings] = useState(false);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [capturedImage, setCapturedImage] = useState(null);
  const [showFlash, setShowFlash] = useState(false);
  const [detectionSettings, setDetectionSettings] = useState({
    lineWidth: 2,
    lineColor: '#FFFFFF',
    opacity: 0.9,
    minScore: 0.5,
    glowIntensity: 0,
    showLabels: true,
    pulseAnimation: false,
    detectionInterval: 100, // ms between detections
    fontSize: 18
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const animationIdRef = useRef(null);
  const streamRef = useRef(null);
  const lastDetectionTime = useRef(0);
  const frameCount = useRef(0);
  const lastFpsTime = useRef(Date.now());
  const currentPredictions = useRef([]);
  const previousPredictions = useRef([]);
  const transitionFrames = useRef(0);

  useEffect(() => {
    loadModel();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, []);

  const loadModel = async () => {
    try {
      await tf.ready();
      const loadedModel = await cocoSsd.load({
        base: 'lite_mobilenet_v2',
        maxDetections: 40
      });
      setModel(loadedModel);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to load model:', error);
      setIsLoading(false);
    }
  };

  const startCamera = async (mode = facingMode) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: mode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Camera access denied:', error);
    }
  };

  const flipCamera = async () => {
    const newMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(newMode);
    await startCamera(newMode);
  };

  const detect = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current || !isDetecting) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Update FPS
    frameCount.current++;
    const now = Date.now();
    if (now - lastFpsTime.current >= 1000) {
      setFps(frameCount.current);
      frameCount.current = 0;
      lastFpsTime.current = now;
    }

    // Only run detection at specified interval
    const timeSinceLastDetection = now - lastDetectionTime.current;
    
    if (timeSinceLastDetection >= detectionSettings.detectionInterval) {
      try {
        const predictions = await model.detect(video);
        previousPredictions.current = [...currentPredictions.current];
        currentPredictions.current = predictions;
        transitionFrames.current = 0;
        lastDetectionTime.current = now;
        setDetectionCount(predictions.filter(p => p.score >= detectionSettings.minScore).length);
      } catch (error) {
        console.error('Detection error:', error);
      }
    }

    // Get actual display dimensions
    const displayWidth = video.offsetWidth;
    const displayHeight = video.offsetHeight;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // Calculate scale and offset for object-cover behavior
    const videoAspectRatio = videoWidth / videoHeight;
    const displayAspectRatio = displayWidth / displayHeight;
    
    let scale, offsetX = 0, offsetY = 0;
    
    if (videoAspectRatio > displayAspectRatio) {
      // Video is wider than display - crop sides
      scale = displayHeight / videoHeight;
      offsetX = (displayWidth - videoWidth * scale) / 2;
    } else {
      // Video is taller than display - crop top/bottom
      scale = displayWidth / videoWidth;
      offsetY = (displayHeight - videoHeight * scale) / 2;
    }
    
    // Set canvas to match display size
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Smooth transition logic
    transitionFrames.current++;
    const transitionProgress = Math.min(transitionFrames.current / 10, 1); // 10 frames for transition
    
    // Draw predictions with smooth transitions
    const allPredictions = new Map();
    
    // Add previous predictions with reduced opacity
    previousPredictions.current.forEach(pred => {
      if (pred.score >= detectionSettings.minScore) {
        allPredictions.set(`${pred.class}_${Math.round(pred.bbox[0])}_${Math.round(pred.bbox[1])}`, {
          ...pred,
          opacity: 1 - transitionProgress
        });
      }
    });
    
    // Add current predictions
    currentPredictions.current.forEach(pred => {
      if (pred.score >= detectionSettings.minScore) {
        const key = `${pred.class}_${Math.round(pred.bbox[0])}_${Math.round(pred.bbox[1])}`;
        allPredictions.set(key, {
          ...pred,
          opacity: transitionProgress
        });
      }
    });

    // Draw all predictions
    allPredictions.forEach((prediction, index) => {
      if (prediction.score >= detectionSettings.minScore) {
        const [x, y, width, height] = prediction.bbox;
        
        // Transform coordinates to match display
        const scaledX = x * scale + offsetX;
        const scaledY = y * scale + offsetY;
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        
        ctx.save();
        
        // Calculate color based on confidence
        let boxColor = detectionSettings.lineColor;
        if (detectionSettings.lineColor === '#FFFFFF') {
          // If white is selected, use hue based on confidence
          const hue = prediction.score * 120; // 0 = red, 120 = green
          boxColor = `hsl(${hue}, 70%, 50%)`;
        }
        
        // Apply glow effect
        if (detectionSettings.glowIntensity > 0) {
          ctx.shadowColor = boxColor;
          ctx.shadowBlur = detectionSettings.glowIntensity;
        }
        
        // Draw bounding box with transition opacity
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = detectionSettings.lineWidth;
        ctx.globalAlpha = detectionSettings.opacity * (prediction.opacity || 1);
        
        if (detectionSettings.pulseAnimation) {
          const pulse = Math.sin(Date.now() * 0.003 + index) * 0.3 + 0.7;
          ctx.globalAlpha = detectionSettings.opacity * pulse;
        }
        
        // Draw rounded rectangle
        const radius = 8;
        ctx.beginPath();
        ctx.moveTo(scaledX + radius, scaledY);
        ctx.lineTo(scaledX + scaledWidth - radius, scaledY);
        ctx.quadraticCurveTo(scaledX + scaledWidth, scaledY, scaledX + scaledWidth, scaledY + radius);
        ctx.lineTo(scaledX + scaledWidth, scaledY + scaledHeight - radius);
        ctx.quadraticCurveTo(scaledX + scaledWidth, scaledY + scaledHeight, scaledX + scaledWidth - radius, scaledY + scaledHeight);
        ctx.lineTo(scaledX + radius, scaledY + scaledHeight);
        ctx.quadraticCurveTo(scaledX, scaledY + scaledHeight, scaledX, scaledY + scaledHeight - radius);
        ctx.lineTo(scaledX, scaledY + radius);
        ctx.quadraticCurveTo(scaledX, scaledY, scaledX + radius, scaledY);
        ctx.closePath();
        ctx.stroke();
        
        // Draw label with minimal design
        if (detectionSettings.showLabels) {
          const label = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
          
          // Save context state for text drawing
          ctx.save();
          
          // Reset transformation for text to prevent mirroring
          if (facingMode === 'user') {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
          }
          
          ctx.font = `bold ${detectionSettings.fontSize}px -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif`;
          
          // Measure text dimensions
          const textMetrics = ctx.measureText(label);
          const padding = 16;
          const labelHeight = detectionSettings.fontSize * 1.8;
          const boxWidth = textMetrics.width + padding * 2;
          
          // Adjust position for mirrored camera
          let labelX = scaledX;
          if (facingMode === 'user') {
            labelX = canvas.width - scaledX - scaledWidth;
          }
          
          // Draw colored background rectangle with rounded corners
          const bgRadius = 6;
          ctx.fillStyle = boxColor;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.moveTo(labelX + bgRadius, scaledY - labelHeight - 10);
          ctx.lineTo(labelX + boxWidth - bgRadius, scaledY - labelHeight - 10);
          ctx.quadraticCurveTo(labelX + boxWidth, scaledY - labelHeight - 10, labelX + boxWidth, scaledY - labelHeight - 10 + bgRadius);
          ctx.lineTo(labelX + boxWidth, scaledY - 10 - bgRadius);
          ctx.quadraticCurveTo(labelX + boxWidth, scaledY - 10, labelX + boxWidth - bgRadius, scaledY - 10);
          ctx.lineTo(labelX + bgRadius, scaledY - 10);
          ctx.quadraticCurveTo(labelX, scaledY - 10, labelX, scaledY - 10 - bgRadius);
          ctx.lineTo(labelX, scaledY - labelHeight - 10 + bgRadius);
          ctx.quadraticCurveTo(labelX, scaledY - labelHeight - 10, labelX + bgRadius, scaledY - labelHeight - 10);
          ctx.closePath();
          ctx.fill();
          
          // Draw text in contrasting color
          ctx.fillStyle = '#FFFFFF';
          ctx.globalAlpha = 1;
          ctx.fillText(label, labelX + padding, scaledY - labelHeight / 2 - 10 + detectionSettings.fontSize / 3);
          
          ctx.restore();
        }
        
        ctx.restore();
      }
    });

    animationIdRef.current = requestAnimationFrame(detect);
  }, [model, isDetecting, detectionSettings]);

  useEffect(() => {
    if (isDetecting) {
      detect();
    } else if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
    }
  }, [isDetecting, detect]);

  const toggleDetection = async () => {
    if (!isDetecting && !streamRef.current) {
      await startCamera();
    }
    setIsDetecting(!isDetecting);
  };

  const startRecording = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const compositeCanvas = document.createElement('canvas');
    const ctx = compositeCanvas.getContext('2d');

    compositeCanvas.width = video.videoWidth;
    compositeCanvas.height = video.videoHeight;

    const stream = compositeCanvas.captureStream(30);
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm',
      videoBitsPerSecond: 2500000
    });

    mediaRecorderRef.current = mediaRecorder;
    recordedChunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, {
        type: 'video/webm'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vision-capture-${Date.now()}.webm`;
      a.click();
    };

    const drawFrame = () => {
      ctx.drawImage(video, 0, 0);
      ctx.drawImage(canvas, 0, 0);
      if (isRecording) {
        requestAnimationFrame(drawFrame);
      }
    };

    mediaRecorder.start();
    setIsRecording(true);
    drawFrame();
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const capturePhoto = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame
    ctx.drawImage(video, 0, 0);

    // Create a second canvas for detections
    const detectionCanvas = document.createElement('canvas');
    const detectionCtx = detectionCanvas.getContext('2d');
    detectionCanvas.width = video.videoWidth;
    detectionCanvas.height = video.videoHeight;

    // Draw current detections at full resolution
    const scale = video.videoWidth / video.offsetWidth;
    
    currentPredictions.current.forEach((prediction) => {
      if (prediction.score >= detectionSettings.minScore) {
        const [x, y, width, height] = prediction.bbox;
        
        // Calculate color based on confidence
        let boxColor = detectionSettings.lineColor;
        if (detectionSettings.lineColor === '#FFFFFF') {
          const hue = prediction.score * 120;
          boxColor = `hsl(${hue}, 70%, 50%)`;
        }
        
        detectionCtx.strokeStyle = boxColor;
        detectionCtx.lineWidth = detectionSettings.lineWidth * scale;
        detectionCtx.globalAlpha = detectionSettings.opacity;
        
        // Draw rounded rectangle
        const radius = 8 * scale;
        detectionCtx.beginPath();
        detectionCtx.moveTo(x + radius, y);
        detectionCtx.lineTo(x + width - radius, y);
        detectionCtx.quadraticCurveTo(x + width, y, x + width, y + radius);
        detectionCtx.lineTo(x + width, y + height - radius);
        detectionCtx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        detectionCtx.lineTo(x + radius, y + height);
        detectionCtx.quadraticCurveTo(x, y + height, x, y + height - radius);
        detectionCtx.lineTo(x, y + radius);
        detectionCtx.quadraticCurveTo(x, y, x + radius, y);
        detectionCtx.closePath();
        detectionCtx.stroke();
        
        // Draw label
        if (detectionSettings.showLabels) {
          const label = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
          
          detectionCtx.font = `bold ${detectionSettings.fontSize * scale}px -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif`;
          const textMetrics = detectionCtx.measureText(label);
          const padding = 16 * scale;
          const labelHeight = detectionSettings.fontSize * 1.8 * scale;
          const boxWidth = textMetrics.width + padding * 2;
          
          // Draw colored background
          const bgRadius = 6 * scale;
          detectionCtx.fillStyle = boxColor;
          detectionCtx.globalAlpha = 0.9;
          detectionCtx.beginPath();
          detectionCtx.moveTo(x + bgRadius, y - labelHeight - 10 * scale);
          detectionCtx.lineTo(x + boxWidth - bgRadius, y - labelHeight - 10 * scale);
          detectionCtx.quadraticCurveTo(x + boxWidth, y - labelHeight - 10 * scale, x + boxWidth, y - labelHeight - 10 * scale + bgRadius);
          detectionCtx.lineTo(x + boxWidth, y - 10 * scale - bgRadius);
          detectionCtx.quadraticCurveTo(x + boxWidth, y - 10 * scale, x + boxWidth - bgRadius, y - 10 * scale);
          detectionCtx.lineTo(x + bgRadius, y - 10 * scale);
          detectionCtx.quadraticCurveTo(x, y - 10 * scale, x, y - 10 * scale - bgRadius);
          detectionCtx.lineTo(x, y - labelHeight - 10 * scale + bgRadius);
          detectionCtx.quadraticCurveTo(x, y - labelHeight - 10 * scale, x + bgRadius, y - labelHeight - 10 * scale);
          detectionCtx.closePath();
          detectionCtx.fill();
          
          // Draw text
          detectionCtx.fillStyle = '#FFFFFF';
          detectionCtx.globalAlpha = 1;
          detectionCtx.fillText(label, x + padding, y - labelHeight / 2 - 10 * scale + detectionSettings.fontSize * scale / 3);
        }
      }
    });

    // Composite the detections onto the main canvas
    ctx.drawImage(detectionCanvas, 0, 0);

    // Flash animation
    setShowFlash(true);
    setTimeout(() => setShowFlash(false), 150);

    // Download the image
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vision-capture-${Date.now()}.jpg`;
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Minimal Header */}
      <div className="absolute top-0 left-0 right-0 z-50 p-8">
        <div className="max-w-7xl mx-auto flex justify-between items-start">
          <div>
            <h1 className="text-xl font-extralight tracking-wide text-white/90">Vision</h1>
            <p className="text-xs font-light text-white/50 mt-1">Neural Detection Engine</p>
          </div>
          
          {isDetecting && (
            <div className="text-right">
              <div className="text-xs font-light text-white/50">
                <div>{fps} FPS</div>
                <div>{detectionCount} objects</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="relative h-screen flex items-center justify-center">
        {isLoading ? (
          <div className="text-center">
            <div className="relative w-20 h-20 mx-auto mb-6">
              <div className="absolute inset-0 border border-white/20 rounded-full"></div>
              <div className="absolute inset-0 border-t border-white rounded-full animate-spin"></div>
            </div>
            <p className="text-sm font-light text-white/50">Initializing Neural Engine</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
              style={{ 
                display: isDetecting ? 'block' : 'none',
                transform: facingMode === 'user' ? 'scaleX(-1)' : 'none'
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              style={{ 
                display: isDetecting ? 'block' : 'none',
                transform: facingMode === 'user' ? 'scaleX(-1)' : 'none'
              }}
            />
            
            {!isDetecting && (
              <div className="text-center z-10">
                <button
                  onClick={toggleDetection}
                  className="group relative px-12 py-5 bg-white/10 backdrop-blur-md text-white rounded-full font-light hover:bg-white/20 transition-all duration-300 border border-white/20"
                >
                  <span className="relative z-10">Begin Analysis</span>
                </button>
                <p className="text-xs text-white/40 mt-4 font-light">Tap to start real-time object detection</p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Control Interface */}
      {isDetecting && (
        <div className="absolute bottom-0 left-0 right-0 p-8">
          <div className="max-w-7xl mx-auto">
            {/* Primary Controls */}
            <div className="flex justify-center items-center gap-4 mb-8">
              {/* Camera Flip */}
              <button
                onClick={flipCamera}
                className="p-4 bg-white/10 backdrop-blur-md rounded-full hover:bg-white/20 transition-all duration-300 border border-white/20"
                title="Flip Camera"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V7a2 2 0 00-2-2h-1m-1 0l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </button>

              {/* Stop Button */}
              <button
                onClick={toggleDetection}
                className="px-8 py-4 bg-white/10 backdrop-blur-md rounded-full font-light hover:bg-white/20 transition-all duration-300 border border-white/20"
              >
                Stop Analysis
              </button>

              {/* Camera Button */}
              <button
                onClick={capturePhoto}
                className="p-4 bg-white/10 backdrop-blur-md rounded-full hover:bg-white/20 transition-all duration-300 border border-white/20"
                title="Capture Photo"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <circle cx="12" cy="13" r="3" strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} />
                </svg>
              </button>

              {/* Record Button */}
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="p-4 bg-white/10 backdrop-blur-md rounded-full hover:bg-white/20 transition-all duration-300 border border-white/20 flex items-center gap-3"
                  title="Start Recording"
                >
                  <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="p-4 bg-red-500/80 backdrop-blur-md rounded-full hover:bg-red-600/80 transition-all duration-300 animate-pulse"
                  title="Stop Recording"
                >
                  <span className="w-4 h-4 bg-white rounded-sm"></span>
                </button>
              )}

              {/* Settings Toggle */}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-4 bg-white/10 backdrop-blur-md rounded-full hover:bg-white/20 transition-all duration-300 border border-white/20"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </button>
            </div>

            {/* Settings Panel */}
            {showSettings && (
              <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 mb-4">
                <h3 className="text-sm font-light text-white/70 mb-4">Detection Settings</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  {/* Detection Interval */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Detection Speed</label>
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="50"
                      value={detectionSettings.detectionInterval}
                      onChange={(e) => setDetectionSettings({...detectionSettings, detectionInterval: parseInt(e.target.value)})}
                      className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <span className="text-xs text-white/40">{detectionSettings.detectionInterval}ms</span>
                  </div>

                  {/* Confidence Threshold */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Confidence</label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={detectionSettings.minScore}
                      onChange={(e) => setDetectionSettings({...detectionSettings, minScore: parseFloat(e.target.value)})}
                      className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-white/40">{Math.round(detectionSettings.minScore * 100)}%</span>
                  </div>

                  {/* Line Width */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Stroke Width</label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={detectionSettings.lineWidth}
                      onChange={(e) => setDetectionSettings({...detectionSettings, lineWidth: parseInt(e.target.value)})}
                      className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Opacity */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Opacity</label>
                    <input
                      type="range"
                      min="0.3"
                      max="1"
                      step="0.1"
                      value={detectionSettings.opacity}
                      onChange={(e) => setDetectionSettings({...detectionSettings, opacity: parseFloat(e.target.value)})}
                      className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Color Picker */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Detection Color</label>
                    <div className="flex items-center gap-2">
                      <input
                        type="color"
                        value={detectionSettings.lineColor}
                        onChange={(e) => setDetectionSettings({...detectionSettings, lineColor: e.target.value})}
                        className="w-10 h-10 rounded-lg cursor-pointer bg-transparent border border-white/20"
                      />
                      <span className="text-xs text-white/40">{detectionSettings.lineColor}</span>
                    </div>
                  </div>

                  {/* Font Size */}
                  <div>
                    <label className="text-xs text-white/50 block mb-2">Font Size</label>
                    <input
                      type="range"
                      min="12"
                      max="32"
                      value={detectionSettings.fontSize}
                      onChange={(e) => setDetectionSettings({...detectionSettings, fontSize: parseInt(e.target.value)})}
                      className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-white/40">{detectionSettings.fontSize}px</span>
                  </div>

                  {/* Toggle Switches */}
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="labels"
                      checked={detectionSettings.showLabels}
                      onChange={(e) => setDetectionSettings({...detectionSettings, showLabels: e.target.checked})}
                      className="w-4 h-4 bg-white/10 border-white/30 rounded"
                    />
                    <label htmlFor="labels" className="text-xs text-white/70">Show Labels</label>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="pulse"
                      checked={detectionSettings.pulseAnimation}
                      onChange={(e) => setDetectionSettings({...detectionSettings, pulseAnimation: e.target.checked})}
                      className="w-4 h-4 bg-white/10 border-white/30 rounded"
                    />
                    <label htmlFor="pulse" className="text-xs text-white/70">Pulse Effect</label>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Flash Animation */}
      {showFlash && (
        <div className="fixed inset-0 bg-white z-50 pointer-events-none animate-flash"></div>
      )}

      {/* Custom Styles */}
      <style jsx>{`
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          cursor: pointer;
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};

export default VisionCapture;
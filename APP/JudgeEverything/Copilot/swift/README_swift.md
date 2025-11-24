Swift client prototype

This folder contains Swift example snippets showing how an iOS app can:
- Capture camera frames (AVCaptureSession)
- Detect touches on the preview to get a click coordinate
- Send the current frame + click coordinate to the Python inference server
- Receive a base64 mask and overlay it on the preview

Notes:
- These are example snippets, not a full Xcode project. To produce an App Store-ready app you should convert the models to Core ML and run inference natively in Swift.
- For prototyping, you can run the Python server on your development machine and point the app to `http://<host-ip>:8000/segment`.

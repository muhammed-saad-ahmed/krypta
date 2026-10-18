import cv2, numpy as np
from config import MAX_IMAGE_MB, MAX_VIDEO_MB, MAX_FILES

class InvalidInput(ValueError): pass

def validate_upload(name, size_bytes, kind):
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    ok = {"image": {"jpg", "jpeg", "png"}, "video": {"mp4", "avi", "mov"}}[kind]
    if ext not in ok: raise InvalidInput(f"Unsupported {kind} type: .{ext}")
    lim = (MAX_IMAGE_MB if kind == "image" else MAX_VIDEO_MB) * 1024 * 1024
    if size_bytes > lim: raise InvalidInput("File too large")
    if size_bytes == 0: raise InvalidInput("Empty file")
    return True

def validate_count(n):
    if n > MAX_FILES: raise InvalidInput(f"Max {MAX_FILES} files")

def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    if img is None: raise InvalidInput("Corrupted or unreadable image")
    return img

def face_blur_available():
    """False on OpenCV >=5, which moved Haar's CascadeClassifier into opencv-contrib."""
    return hasattr(cv2, "CascadeClassifier")

def blur_faces(img):
    """Optional privacy step: Haar-cascade face blur (best effort; does not identify anyone).
    License plates are NOT reliably handled — see docs. Returns the image unchanged, rather
    than crashing, if this OpenCV build has no CascadeClassifier (see face_blur_available())."""
    if not face_blur_available():
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    out = img.copy()
    for (x, y, w, h) in casc.detectMultiScale(g, 1.1, 5):
        out[y:y+h, x:x+w] = cv2.GaussianBlur(out[y:y+h, x:x+w], (51, 51), 0)
    return out

def sample_frames(path, every_n=15, max_frames=60):
    cap = cv2.VideoCapture(path); frames = []; i = 0
    while cap.isOpened() and len(frames) < max_frames:
        ok, f = cap.read()
        if not ok: break
        if i % every_n == 0: frames.append(f)
        i += 1
    cap.release()
    if not frames: raise InvalidInput("Video has no readable frames")
    return frames

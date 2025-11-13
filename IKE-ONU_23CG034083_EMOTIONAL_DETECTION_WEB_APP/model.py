import os
import pickle
import cv2
import numpy as np

# Define constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'emotion_detect_model1.pkl')

EMOTION_CLASSES = ['Happy', 'Sad', 'Neutral']


class EmotionRecognizer:
    """Class to handle face detection and emotion prediction."""

    def __init__(self, model_file: str = None):
        # Initialize the model and Haar cascade
        self.model_file = model_file or MODEL_FILE
        self.classifier = None
        self.face_detector = None

        # Load model and cascade
        self._initialize_model()
        self._initialize_detector()

    def _initialize_model(self):
        """Load the trained emotion model from pickle."""
        try:
            with open(self.model_file, 'rb') as f:
                self.classifier = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as err:
            print(f"[Error] Could not load the model file: {self.model_file}\n{err}")
            self.classifier = None

    def _initialize_detector(self):
        """Set up the OpenCV Haar Cascade for face detection."""
        cascade_dir = getattr(cv2.data, "haarcascades", "")
        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")

        if not os.path.exists(cascade_path):
            print("[Warning] Haarcascade file not found. Ensure OpenCV is properly installed.")
        self.face_detector = cv2.CascadeClassifier(cascade_path)

    def _prepare_input(self, face_img):
        """Preprocess a cropped face image for model input."""
        if face_img is None:
            return None

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized.flatten().reshape(1, -1)

    def analyze_emotion(self, frame):
        """
        Detect the largest face in the image and predict the emotion.

        Returns:
            (emotion_label, probability_map)
        """
        if frame is None or self.classifier is None:
            return ('ModelNotReady', {})

        # Convert to grayscale for detection
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        faces = self.face_detector.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return ('NoFaceDetected', {})

        # Take the largest detected face
        x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        cropped_face = gray_img[y:y + h, x:x + w]

        features = self._prepare_input(cropped_face)
        if features is None:
            return ('PreprocessingError', {})

        try:
            probabilities = self.classifier.predict_proba(features)[0]
            prediction_idx = int(self.classifier.predict(features)[0])
        except Exception as e:
            print("[Error during prediction]:", e)
            return ('PredictionError', {})

        emotion = EMOTION_CLASSES[prediction_idx] if prediction_idx < len(EMOTION_CLASSES) else str(prediction_idx)
        prob_map = {EMOTION_CLASSES[i]: float(probabilities[i]) for i in range(min(len(probabilities), len(EMOTION_CLASSES)))}

        return (emotion, prob_map)


# Optional: Simple test
if __name__ == "__main__":
    print("Testing EmotionRecognizer class...")
    detector = EmotionRecognizer()
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy input
    result = detector.analyze_emotion(test_image)
    print("Prediction:", result)

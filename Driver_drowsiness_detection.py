import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# -----------------------------
# 1. LOAD DATASET
# -----------------------------

def load_data(folder, label):
    data = []
    labels = []

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        features = hog(
            img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        data.append(features)
        labels.append(label)

    return data, labels


# Dataset paths
closed_data, closed_labels = load_data(
    r"C:\Users\hp\Desktop\Internship training\Driver Drowsiness Dataset (DDD)\Drowsy", 1)

open_data, open_labels = load_data(
    r"C:\Users\hp\Desktop\Internship training\Driver Drowsiness Dataset (DDD)\Non Drowsy", 0)

X = np.array(open_data + closed_data)
y = np.array(open_labels + closed_labels)

# -----------------------------
# 2. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# -----------------------------
# 4. EVALUATION
# -----------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", model.score(X_test, y_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. SAVE MODEL
# -----------------------------

joblib.dump(model, "saved_model.pkl")
print("\nModel Saved Successfully!")

# -----------------------------
# 6. REAL-TIME WEBCAM DETECTION
# -----------------------------

# Load saved model
model = joblib.load("saved_model.pkl")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# 🔥 FIXED WEBCAM LINE FOR WINDOWS
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not cap.isOpened():
    print("❌ Error: Webcam not detected")
    exit()

print("\nPress ESC to Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        eye_img = gray[y:y+h, x:x+w]

        eye_img = cv2.resize(eye_img, (64, 64))

        features = hog(
            eye_img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        prediction = model.predict([features])[0]

        label = "Drowsy" if prediction == 1 else "Open"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



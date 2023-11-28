import cv2 as cv
import dlib
from scipy.spatial import distance as dist


def CalculateEyeRatioFormula(cordinates):
    p2_minus_p6 = dist.euclidean(cordinates[1], cordinates[5])
    p3_minus_p5 = dist.euclidean(cordinates[2], cordinates[4])
    p1_minus_p4 = dist.euclidean(cordinates[0], cordinates[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


class EyeDetector:
    def __init__(self):
        self.faces = None
        self.landmarks = None
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def findFaceDetector(self, img, gray_images):
        self.faces = self.face_detector(gray_images)
        for face in self.faces:
            # face detection
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def calculateEAR(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for face in self.faces:
            self.landmarks = self.predictor(gray, face)
            storeLandmarks = []
            for i in range(36, 48):
                eachPoint = (self.landmarks.part(i).x, self.landmarks.part(i).y)
                cv.circle(img, eachPoint, 2, (255, 0, 0), 1)
                storeLandmarks.append(eachPoint)
            rightEyeLandmarks, leftEyeLandmarks = storeLandmarks[:6], storeLandmarks[6:]

            leftEar = CalculateEyeRatioFormula(leftEyeLandmarks)
            rightEar = CalculateEyeRatioFormula(rightEyeLandmarks)

            avg = (leftEar + rightEar) / 2

            if avg < 0.21:
                cv.putText(img, f'Closed:{avg:.2f}', (30, 80), cv.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 255), 2)
            elif 0.21 < avg < 0.25:
                cv.putText(img, f'Drowsiness:{avg:.2f}', (30, 80), cv.FONT_HERSHEY_COMPLEX, 1.1, (0, 255, 0), 2)
            else:
                cv.putText(img, f'Open:{avg:.2f}', (30, 80), cv.FONT_HERSHEY_COMPLEX, 1.1, (0, 255, 0), 2)
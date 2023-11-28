import drowsinessModule as dM
import cv2 as cv


class DrowsyDetector:
    def __init__(self):
        self.detector = dM.EyeDetector()

    def monitor(self):
        cap = cv.VideoCapture(0)

        while True:
            _, img = cap.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self.detector.findFaceDetector(img, gray)
            self.detector.calculateEAR(img)
            cv.imshow("Driver Drowsiness Monitoring", img)

            if cv.waitKey(20) & 0xFF == ord('q'):
                break


def main():
    drowsy = DrowsyDetector()
    drowsy.monitor()


if __name__ == "__main__":
    main()

import timeit
import cv2
import glob
import os
import time
import timeit
import numpy as np
from keras.models import load_model
from cv_core.detector.face_inference import FaceLocationDetector
from argparse import ArgumentParser

IMG_SIZE = 64

test_path = 'batch_jpg_file/test_set/7/'
max_age = 70
model_name = 'models/138-0.08.h5'

parser = ArgumentParser()
parser.add_argument("--video", help="using video")


def init_detector():
    global detector

    detector = FaceLocationDetector()


def main():
    args = parser.parse_args()
    init_detector()
    model = load_model(model_name)

    if args.video:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = detector.predict(img)

            if 1 == len(face_locs):
                start_x, start_y, end_x, end_y = face_locs[0]
                img = img[start_y:end_y, start_x:end_x, :]
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img[np.newaxis, :]

                img = img.astype("float32") / 255

                start = timeit.default_timer()
                age_pred, gender_pred = model.predict(img)
                end = timeit.default_timer()

                gender_pred = 'female' if np.argmax(
                    gender_pred) == 0 else 'male'

                age_pred = (age_pred[0] * max_age)[0]

                cv2.putText(frame, 'age: {}'.format(int(round(age_pred))),
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            1, cv2.LINE_AA)
                cv2.putText(frame, 'gender: {}'.format(gender_pred), (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, 'Time cost: {}s'.format(
                    round((end - start),
                          3)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)

            else:
                cv2.putText(frame, 'age:', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'gender:', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, 'Time cost:', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)

            cv2.imshow('video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        img = cv2.imread('images/marco.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = detector.predict(img)
        if 1 == len(face_locs):
            start_x, start_y, end_x, end_y = face_locs[0]
            img = img[start_y:end_y, start_x:end_x, :]
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img[np.newaxis, :]

            img = img.astype("float32") / 255

            start = timeit.default_timer()
            age_pred, gender_pred = model.predict(img)
            end = timeit.default_timer()

            gender_pred = 'female' if np.argmax(gender_pred) == 0 else 'male'

            # print(gender)
            print(gender_pred)
            print((age_pred[0] * max_age)[0])
            print('Cost {}s'.format(end - start))


if __name__ == '__main__':
    main()
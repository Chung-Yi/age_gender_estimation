from cv_core.detector.face_inference import FaceLocationDetector
import glob
import os
import cv2

train_path = 'age_images'


def init_detector():
    global detector

    detector = FaceLocationDetector()


def main():

    init_detector()
    img_list = glob.glob(os.path.join(train_path, '*.jpg'), recursive=True)

    # for i in range(100):
    #     os.mkdir(os.path.join('faces', str(i + 1)))

    for img in img_list:
        img_name = img.split('/')[-1]
        age = img_name.split('_')[0]

        img = cv2.imread(img)
        face_locs = detector.predict(img)
        if not 1 == len(face_locs):
            continue
        start_x, start_y, end_x, end_y = face_locs[0]
        img = img[start_y:end_y, start_x:end_x, :]

        # cv2.imwrite(os.path.join('faces/train_set', img_name), img)
        cv2.imwrite(os.path.join('faces/' + str(age), img_name), img)
        # print(os.path.join('faces/train_set', img_name))


if __name__ == "__main__":
    main()
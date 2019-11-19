import cv2
import os
import glob
from uuid import uuid4
from pyagender import PyAgender
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--age_dir", help="number")
args = parser.parse_args()

# image_path = 'age_images/variance/*.jpg'.format(args.age_dir)
AGEIMAGE = 'age_images'

person_image_path = 'images/潘若迪/*.jpg'

# person_image_path = '20190924_asia_face_label/周迅/*.jpg'

images_path = os.path.join(AGEIMAGE, '**/*.jpg')


def _separate(images):

    for image in images:
        img = cv2.imread(image)
        faces = agender.detect_genders_ages(img)
        if len(faces) == 0:
            print('no face, {}'.format(image))
            continue

        age = int(round(faces[0]['age'], 0))
        gender = 'f' if faces[0]['gender'] > 0.5 else 'm'
        # print(age, image)

        if age < 15:
            age = 15
        # elif age > 40 and age < 60:
        #     age += 10
        elif age > 70:
            age = 70

        new_name = str(age) + '_' + gender + '_' + str(uuid4()) + '.jpg'

        # if not os.path.exists(os.path.join('faces', AGEIMAGE, str(age))):
        #     os.mkdir(os.path.join('faces', AGEIMAGE, str(age)))

        os.rename(image, os.path.join(AGEIMAGE, new_name))
        # print(age)

        # if age >= 40 and age <= 70:
        #     print(age, image)
        #     os.rename(image, os.path.join(AGEIMAGE, new_name))

        # elif age > 70:
        #     print(age)
        #     # print(new_name)
        #     os.rename(image, os.path.join(AGEIMAGE, new_name))


def main():

    global agender

    agender = PyAgender()

    images = glob.glob(person_image_path)

    _separate(images)


if __name__ == '__main__':
    main()
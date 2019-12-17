import boto3
import json
import io
import os
import csv
import glob
from PIL import Image


def main():
    images = glob.glob('20190924_asia_face_label/**/*.jpg')
    attr = [
        'IMAGEPATH', 'AgeRange', 'Beard', 'BoundingBox', 'Confidence',
        'Emotions', 'Eyeglasses', 'EyesOpen', 'Gender', 'Landmarks',
        'MouthOpen', 'Mustache', 'Pose', 'Quality', 'Smile', 'Sunglasses'
    ]

    with open('testdata_info.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=attr)
        writer.writeheader()

        for i, image_name in enumerate(images):

            image = Image.open(image_name)
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            byte_im = buf.getvalue()

            session = boto3.Session(profile_name='qa')
            client = session.client(
                'rekognition', region_name="ap-southeast-1")

            response = client.detect_faces(
                Image={'Bytes': byte_im}, Attributes=['ALL'])

            if len(response['FaceDetails']) == 0:
                print(image_name)
                continue

            faceDetail = response['FaceDetails'][0]
            faceDetail['IMAGEPATH'] = image_name
            writer.writerow(faceDetail)

            if i % 500 == 0:
                print('Finished {} images'.format(i))

            if i >= 1500:
                break


if __name__ == '__main__':
    main()
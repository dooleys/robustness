import argparse
from concurrent import futures
import os
import re
import sys
import tempfile

import boto3
import botocore
from botocore.exceptions import ClientError

from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials


import tqdm
import logging
import json

# Authenticate for Azure
# Authenticates your credentials and creates a client.
SUBSCRIPTION_KEY = "YOUR_KEY_HERE"
ENDPOINT = "YOUR_ENDPOINT_HERE"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY))

# AWS information
BUCKET_NAME = 'YOUR_BUCKET_HERE'
s3_client = boto3.client('s3')
rek_client = boto3.client('rekognition')

# where you do want the files to be save locally on disk?
JSON_PATH = '/data/'

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: Local ile to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name
    # Upload the file
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        sys.exit(f'ERROR when uploading json `{object_name}`: {str(exception)}')
        return False
    return True


def write_upload_response(response, photo, bucket, service, upload_to_s3=False):
    """Given a response from the detection API of `service`,
    save it locally and upload it to bucket if `upload_to_s3` is True
     by replacing `c_images` with `response/{service}`

    :param response (obj): Response objects from API call
    :param photo (str): File path in S3 bucket of image
    :param bucket (str): Bucket where image lives in bucket
    :param service (str): Name of the API service (aws, azure)
    :param upload_to_s3 (bool): Whether you want to upload the response json to s3
    """
    # get the photo's corruption, original_image name, and severity
    corruptions = ['gaussian-noise', 'shot-noise', 'impulse-noise', 'defocus-blur', 'glass-blur',
                    'motion-blur', 'zoom-blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic-transform', 'pixelate', 'jpeg-compression', 'clean']
    # which corruption is this?
    c = [c for c in corruptions if c in photo][0]
    original_photo = photo[:photo.index(c)]
    # remove trailing underscore if there
    if original_photo[-1] == '_':
        original_photo = original_photo[:-1]
    s = os.path.splitext(photo)[0].split('_')[-1]
    # format the saved json response
    response = {
        'original_photo': original_photo,
        'photo': photo,
        'corruption': c,
        'severity': s,
        'service': service,
        'response': response,
        }
    # replace in path c_images with response/{service}
    # e.g., if photo = 'foo/c_images/path/image.png'
    # then json_fn = 'response/{service}/path/image_{service}.json'
    # find where c_images is:
    i = photo.index('c_images/')
    # replace through that part with response/{service}
    json_fn = photo.replace(photo[:(i+len('c_images/'))], 'response/{}/'.format(service))
    json_fn = os.path.splitext(json_fn)[0] + '_{}.json'.format(service)
    temp_file_path = JSON_PATH + json_fn
    folder, _ = os.path.split(temp_file_path)
    # check to make sure the folder where you're writing the json exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    # write the json
    with open(temp_file_path, 'w') as json_file:
        json.dump(response, json_file)
    # upload it to S3. if you do, delete the json
    if upload_to_s3:
        success = upload_file(temp_file_path, bucket, json_fn)
        os.remove(temp_file_path)
    return success

def aws_detect_faces(photo, bucket):
    """Run AWS Rekognition detect_face API on photo from bucket

    :param photo (str): File path in S3 bucket of image
    :param bucket (str): bucket where image lives
    """
    try:
        response = rek_client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':photo}},Attributes=['ALL'])
        write_upload_response(response, photo, bucket, 'aws')
    except Exception as e:
        print(photo)
        print(e)
        return False
    return True

def azure_detect_faces(photo, bucket):
    """Run Azure's face detection API on photo from bucket

    :param photo (str): File path in S3 bucket of image
    :param bucket (str): bucket where image lives
    """
    try:
        bucket_url = 'https://{}.s3.us-east-2.amazonaws.com/'.format(bucket)
        single_face_image_url = bucket_url + photo
        # We use detection model 3 to get better performance on small faces
        detected_faces = face_client.face.detect_with_url(url=single_face_image_url,
                                                          return_face_landmarks=True,
                                                          return_face_attributes=['headpose','mask'],
                                                          detection_model='detection_03')
        response = [x.as_dict() for x in detected_faces]
        write_upload_response(response, photo, bucket, 'azure')
    except Exception as e:
        print(photo)
        print(e)
        return False
    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
            'image_list',
            type=str,
            default=None,
            help=('Filename that contains the paths of images on S3 to process'))
    parser.add_argument(
            '--num_processes',
            type=int,
            default=5,
            help='Number of parallel processes to use (default is 5).')
    parser.add_argument(
            '--download_bucket',
            type=str,
            default='inthewild',
            help='Bucket from which to download the image.')
    parser.add_argument(
            '--upload_folder',
            type=str,
            default=None,
            help='Folder where to upload the API responses.')
    parser.add_argument(
            '--service',
            type=str,
            default=None,
            help='Name of API service. Either aws or azure')
    args = vars(parser.parse_args())


    # Open the image list
    with open(args['image_list'], 'r') as f:
        image_list =  [line.strip() for line in f]


    # Set up progress bar
    progress_bar = tqdm.tqdm(
            total=len(image_list), desc='Processing images', leave=True)
    # Set up Threading environment and loop through images in image_list
    # based on the service which was specified in the arguments
    with futures.ThreadPoolExecutor(max_workers=args['num_processes']) as executor:
        if args['service'] == 'aws':
            all_futures = [
                    executor.submit(aws_detect_faces, image, args['download_bucket']) for image in image_list
            ]
        elif args['service'] == 'azure':
            # Create an authenticated FaceClient.
            all_futures = [
                    executor.submit(azure_detect_faces, image, args['download_bucket']) for image in image_list
            ]
        else:
            all_futures = []

        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()

import boto3
import pandas 

def get_folder_names(s3_client, bucket_name, directory_name):
    """
    Read all folder (datapoints) names in the images dataset
    """

    paginator = s3_client.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=directory_name,
        Delimiter='/'
    )

    folder_names = []
    for response in response_iterator:
        if response.get('CommonPrefixes') is not None:
            for prefix in response.get('CommonPrefixes'):
                # Extract the folder name from the prefix key
                folder_name = prefix.get('Prefix')
                # Removing the base directory and the trailing slash to get the folder name
                folder_name = folder_name[len(directory_name):].strip('/')
                folder_names.append(folder_name)

    return folder_names


def load_strongly_labelled_data(set='train'):
    """
    Loads data from S3 bucket
    """

    with open('secrets.txt', 'r') as f:
        lines = f.readlines()
        AWS_SECRET_ACCESS_KEY = lines[0].split('=')[1].strip()
        AWS_ACCESS_KEY = lines[1].split('=')[1].strip()
        bucket_name = lines[2].split('=')[1].strip()

    s3_client = boto3.client('s3',
                             aws_access_key_id = AWS_ACCESS_KEY,
                             aws_secret_access_key = AWS_SECRET_ACCESS_KEY)
    
    directory_name = f"{set}_strong/"


    folders = get_folder_names(s3_client, bucket_name, directory_name)

    
    


    pass
import pandas as pd
import boto3
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.io import read_video



class StronglyLabelledDataset(Dataset):
    """
    Video-audio dataset
    """
    def __init__(self, set='train', transform=None):
        """
        Custom dataset initializer.
        :param set: The dataset split, e.g. 'train', 'test', 'val'.
        :param transform: Transformations to be applied to the video frames.
        """
        self.transform = transform
        
        # Read credentials
        with open('secrets.txt', 'r') as f:
            lines = f.readlines()
            AWS_SECRET_ACCESS_KEY = lines[0].split('=')[1].strip()
            AWS_ACCESS_KEY = lines[1].split('=')[1].strip()
            self.bucket_name = lines[2].split('=')[1].strip()

        self.s3_client = boto3.client('s3',
                                      aws_access_key_id=AWS_ACCESS_KEY,
                                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        
        # List of folders (datapoints)
        self.directory_name = set + "_strong/"
        self.folders = self.get_folder_names(set + '_strong/')
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.folders)
    
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index `idx`.
        """
        folder = self.folders[idx]
        id = folder.strip('_')[0]
        
        # Keys for video and metadata
        video_key = f"{folder}/video.mp4"
        metadata_key = f"{folder}/metadata.csv"
        
        # Load video
        video_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=video_key)
        video_content = BytesIO(video_object['Body'].read())
        video, audio, info = read_video(video_content)

        if self.transform:
            video = self.transform(video)
        
        # Load metadata
        metadata_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
        metadata_content = BytesIO(metadata_object['Body'].read())
        metadata_df = pd.read_csv(metadata_content)

        return video, audio, metadata_df
    
    def get_folder_names(self, directory_name):
        """
        Read all folder (datapoints) names in the images dataset
        """

        paginator = self.s3_client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self.bucket_name,
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




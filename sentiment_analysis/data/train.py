import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# Set up SageMaker resources
sagemaker_session = sagemaker.Session()
role = get_execution_role()
bucket = '<your-sagemaker-bucket>'
prefix = 'sagemaker/sentiment_analysis'

# Download the IMDb dataset
!wget -O aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf aclImdb_v1.tar.gz

# Preprocess the dataset
from utils import preprocess_dataset
preprocess_dataset('aclImdb/train', 'train.csv')
preprocess_dataset('aclImdb/test', 'test.csv')

# Upload the preprocessed dataset to S3
train_data = sagemaker_session.upload_data('train.csv', bucket=bucket, key_prefix=prefix+'/input/train')
test_data = sagemaker_session.upload_data('test.csv', bucket=bucket, key_prefix=prefix+'/input/test')

# Create an Estimator
container = get_image_uri(boto3.Session().region_name, 'blazingtext')
estimator = sagemaker.estimator.Estimator(container,
                                          role,
                                          train_instance_count=1,
                                          train_instance_type='ml.c4.2xlarge',
                                          output_path='s3://{}/{}/output'.format(bucket, prefix),
                                          sagemaker_session=sagemaker_session)

# Set hyperparameters
estimator.set_hyperparameters(mode='supervised',
                              epochs=10,
                              learning_rate=0.01,
                              min_count=2,
                              vector_dim=300)

# Train the model
estimator.fit({'train': train_data})

# Deploy the model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

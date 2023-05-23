import boto3
import sagemaker

# Set up SageMaker resources
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
endpoint_name = '<your-endpoint-name>'

# Load the trained model artifacts
trained_model = sagemaker.Model(model_data='<s3-model-data-location>',
                                image='<your-image-uri>',
                                role=role)

# Create an inference endpoint configuration
endpoint_config = sagemaker.Session().create_endpoint_config(name='endpoint-config',
                                                             model_name=trained_model.name,
                                                             initial_instance_count=1,
                                                             instance_type='ml.m4.xlarge')

# Create the inference endpoint
endpoint = sagemaker.Session().create_endpoint(name=endpoint_name,
                                               config_name='endpoint-config')

# Wait for the endpoint to be ready
endpoint.wait()

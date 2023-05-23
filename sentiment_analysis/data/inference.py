import boto3
import sagemaker

# Set up SageMaker resources
sagemaker_session = sagemaker.Session()
endpoint_name = '<your-endpoint-name>'

# Create a predictor
predictor = sagemaker.predictor.RealTimePredictor(endpoint_name)

# Make predictions
def predict_sentiment(text):
    data = {'instances': [{'data': text}]}
    response = predictor.predict(data)
    result = response['predictions'][0]['predicted_label']
    return result

# Example usage
input_text = "This is a great movie!"
prediction = predict_sentiment(input_text)
print(f"Predicted sentiment: {prediction}")

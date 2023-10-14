import os
import sys
import boto3
import numpy as np
import pandas as pd
import sklearn
from awsglue.utils import getResolvedOptions
from io import StringIO

# Helper function to split dataset (80/19/1)
def split_data(df, train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42)

    # Calculate the split sizes
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    test_size = int(test_ratio * total_samples)
    validation_size = total_samples - train_size - test_size

    # Split the DataFrame into train, test, and validation
    train_data = df[:train_size]
    test_data = df[train_size:train_size + test_size]
    validation_data = df[train_size + test_size:]
    
    return [('train', train_data), ('test', test_data), ('validate', validation_data), ('baseline', train_data)]

# Get job args
args = getResolvedOptions(sys.argv, ['S3_INPUT_BUCKET', 'S3_INPUT_KEY_PREFIX', 'S3_OUTPUT_BUCKET', 'S3_OUTPUT_KEY_PREFIX'])

# Downloading the data from S3 into a Dataframe
column_names = ["sex", "length", "diameter", "height", "whole weight",  
                "shucked weight", "viscera weight", "shell weight", "rings"]
client = boto3.client('s3')
bucket_name = args['S3_INPUT_BUCKET']
object_key = os.path.join(args['S3_INPUT_KEY_PREFIX'], 'abalone.csv')
print("Downloading input data from S3 ...\n")
csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
data = pd.read_csv(StringIO(csv_string), sep=',', names=column_names)

# Re-order data to better separate features
data = data[["rings", "sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight"]]

# Create dummy variables for categorical `sex` feature using pandas
print("Encoding Features ...\n")
data = pd.get_dummies(data)
columns_to_replace = ['sex_F','sex_I','sex_M']
data[columns_to_replace] = data[columns_to_replace].replace({True: 1, False: 0})

# Create train, test and validate datasets
print("Creating dataset splits ...\n")
datasets = split_data(data)

# Upload data to S3 as .csv file while separating validation set
for file_name, partition_name in datasets:
    if file_name == 'test':
        print("Writing {} data ...\n".format(file_name))
        np.savetxt(file_name+'.csv', partition_name, delimiter=',')
        boto3.Session().resource('s3').Bucket(args['S3_OUTPUT_BUCKET']).Object(os.path.join(args['S3_OUTPUT_KEY_PREFIX'], 'testing', file_name+'.csv')).upload_file(file_name+'.csv')
    elif file_name == 'baseline':
        print("Writing {} data ...\n".format(file_name))
        np.savetxt(
            file_name+'.csv',
            partition_name,
            delimiter=',',
            header="rings,length,diameter,height,whole weight,shucked weight,viscera weight,shell weight,sex_F,sex_I,sex_M"
        )
        boto3.Session().resource('s3').Bucket(args['S3_OUTPUT_BUCKET']).Object(os.path.join(args['S3_OUTPUT_KEY_PREFIX'], 'baseline', file_name+'.csv')).upload_file(file_name+'.csv')
    else:
        print("Writing {} data ...\n".format(file_name))
        np.savetxt(file_name+'.csv', partition_name, delimiter=',')
        boto3.Session().resource('s3').Bucket(args['S3_OUTPUT_BUCKET']).Object(os.path.join(args['S3_OUTPUT_KEY_PREFIX'], 'training', file_name+'.csv')).upload_file(file_name+'.csv')

print("Done writing to S3 ...\n")
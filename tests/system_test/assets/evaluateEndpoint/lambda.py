import os
import io
import json
import re
import logging
import boto3
import time
import botocore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker-runtime")


def evaluate_model(bucket, key, endpoint_name):
    """
    Description:
    ------------
    Executes model predictions on the testing dataset.
    
    :bucket: (str) Pipeline S3 Bucket.
    :key: (str) Path to "testing" dataset.
    :endpoint_name: (str) Name of the 'Dev' endpoint to test.

    :returns: Lists of ground truth, prediction labels and response times.
    
    """
    column_names = ["rings", "length", "diameter", "height", "whole weight", "shucked weight",
                    "viscera weight", "shell weight", "sex_F", "sex_I", "sex_M"]
    response_times = []
    predictions = []
    y_test = []
    obj = s3.get_object(Bucket=bucket, Key=key)
    test_df = pd.read_csv(io.BytesIO(obj['Body'].read()), names=column_names)

    for col in column_names:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    y = test_df['rings'].to_numpy()
    X = test_df.drop(['rings'], axis=1)
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    arr_test = X
    
    # Cycle through each row of the data to get a prediction
    for row in range(len(arr_test)):
        payload = ",".join(map(str, arr_test[row]))
        elapsed_time = time.time()
        try:
            response = sm_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType = "text/csv",
                Body=payload
            )
        except ClientError as e:
            error_message = e.response["Error"]["Message"]
            logger.error(error_message)
            raise Exception(error_message)
        
        response_times.append(time.time() - elapsed_time)
        result = response['Body'].read()
        pattern = r'[^0-9.]+'
        result = re.split(pattern,result.decode("utf-8"))
        predictions += [float(r) for r in result if r != ""]
        y_test += [float(y[row])]

        #print(result)
        #print(y[row])

        #print(type(result))
        #print(type([row]))
    
    return y_test, predictions, response_times


def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    logger.debug("## Event ##")
    logger.debug(event)
    
    # Ensure variables passed from Model Evaluation Step
    if ("Bucket" in event):
        bucket = event["Bucket"]
    else:
        raise KeyError("S3 'Bucket' not found in Lambda event!")
    if ("Key" in event):
        key = event["Key"]
    else:
        raise KeyError("S3 'Key' not found in Lambda event!")
    if ("Output_Key" in event):
        output_key = event["Output_Key"]
    else:
        raise KeyError("S3 'Output_Uri' not found in Lambda event!")
    if ("Endpoint_Name" in event):
        endpoint_name = event["Endpoint_Name"]
    else:
        raise KeyError("'SageMaker Endpoint Name' not found in Lambda event!")
    
    # Get the evaluation results from SageMaker hosted model
    logger.info("Evaluating SageMaker Hosted Model ...")
    y, y_pred, times = evaluate_model(bucket, key, endpoint_name)

    print(y)
    print(y_pred)
    print(times)
    
    # Calculate the metrics
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    std = np.std(np.array(y) - np.array(y_pred))

    # Save Metrics to S3 for Model Package
    logger.info("Root Mean Square Error: {}".format(rmse))
    logger.info("Average Endpoint Response Time: {:.2f}s".format(np.mean(times)))
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }
    try:
        s3.put_object(
            Bucket=bucket,
            Key="{}/{}".format(output_key, "evaluation.json"),
            Body=json.dumps(report_dict, indent=4)
        )
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

    # Return results
    logger.info("Done!")
    return {
        "statusCode": 200,
        "Result": rmse,
        "AvgResponseTime": "{:.2f} seconds".format(np.mean(times))
    }
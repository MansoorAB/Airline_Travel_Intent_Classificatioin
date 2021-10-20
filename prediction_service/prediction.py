import os
import json
import pickle
import numpy as np
import pandas as pd
import ktrain
from datetime import datetime

from prediction_service import s3_methods


def get_s3_data(s3_bucket, local_folder):
        # This methods gets data from s3 bucket to local folder

        print(datetime.now(), 'starting to fetch s3 data')
        s3m_object = s3_methods.S3Methods()
        s3m_object.get_s3_bucket_data_to_local(s3_bucket, local_folder)
        print(datetime.now(), 's3 data fetch is complete now!')


def predict(data):
    
    model_path = os.path.join("prediction_service", "distilbert_atis_intent_classifier")

    if os.path.isdir(model_path):
        print('4a. model weights found, hence skipping download from s3 bucket.')
    else:
        print('4a. importing model weights from s3 bucket')
        get_s3_data('distilbert-atis-intent-classifier', model_path)

    print('4b. loading model')
    loaded_predictor = ktrain.load_predictor(model_path)

    print('4c. now predicting for the query: %s' %data)
    result = loaded_predictor.predict(data)
    print('4d. intent class predicted is %s' %result)
    
    return result


def form_response(query):
    try:
        print('3. inside form response')
        intent = predict(query)
        print('5. predicted intent is ', intent)
        response = "Query intent is %s" %intent
        return response
    except Exception as e:
        response = "Unexpected error has occured - %s" %str(e)
        return response


def api_response(query):

    try:  
        print('3. inside api_response')   
        intent = predict(query)          
        print('5. predicted intent is ', intent)
        response = {"Query Intent": intent}
        return response         

    except Exception as e:
        response = {"Unexpected Error ": str(e)}
        return response


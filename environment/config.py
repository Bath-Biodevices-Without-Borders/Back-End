import pymongo
from typing import Dict
from pydantic import ValidationError
from environment.models import Reading


# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://tomH:mbWS0uIrROmONBOv@biodevice.3jz8qms.mongodb.net/")
db = client["your_database"]
collection_accounts = db["user_accounts"]

# collection_accounts_list = collection_accounts.find().to_list()

# def add_user(details):
#     account = {
#         "username": details[0],
#         "email": details[1],
#         "password": details[2],
#         "verified": False
#     }
#     result = collection_accounts.insert_one(account)
#     return str(result.inserted_id)  # Return the user ID as a string
    
# collection_data = db["user_data"]

# collection_data_list = collection_data.find().to_list()
    
# data_reading = Reading

# def add_reading(details: Dict) -> str:
#     try:
#         # Validate details against the data_reading schema
#         reading = data_reading(**details)
        
#         # Convert to dictionary for MongoDB insertion
#         reading_dict = reading.dict(by_alias=True)
        
#         # Insert into MongoDB
#         result = collection_data.insert_one(reading_dict)
        
#         return f"Reading added successfully with ID: {result.inserted_id}"
#     except ValidationError as e:
#         return f"Validation error: {e}"
#     except Exception as e:
#         return f"An error occurred: {e}"
    
def get_reading():
    try:
        # Fetch all documents from the collection
        readings = list(collection_data.find())
        
        # Optionally, print the readings or process them
        for reading in readings:
            print(reading)
        
        return readings
    except Exception as e:
        return f"An error occurred: {e}"
    
# def get_reading_by_uid(uid: str):
#     """
#     Fetch readings from the database for a specific user based on their UID.

#     Args:
#         uid (str): The user ID to fetch readings for.

#     Returns:
#         List[Dict]: A list of readings for the specified user.
#     """
#     try:
#         # Query MongoDB for readings where 'uid' matches the given parameter
#         readings = list(collection_data.find({"uid": uid}))
        
#         # Optionally process or log the results
#         if not readings:
#             return f"No readings found for UID: {uid}"
        
#         return readings
#     except Exception as e:
#         return f"An error occurred while fetching readings: {e}"
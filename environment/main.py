from fastapi import FastAPI, HTTPException,Depends, Body
from fastapi.responses import FileResponse, HTMLResponse
from environment.schemas import signup,login,input_data
from dotenv import dotenv_values
from pymongo import MongoClient
from environment.config import add_user,add_reading,get_reading
import numpy as np


#config
config = dotenv_values(".env")
#create app
app = FastAPI()

#endpoint for registering
# @app.post("/register")
# def register_user():
#     details = signup()
#     add_user(details)
#     return {"message": "user registered"}

#base
@app.get("/")
async def root():
    return {"message": "home"}

#endpoint for adding a reading
# @app.post("/uploadreading")
# def upload_reading(
#     email: str = Body(..., embed=True),
#     password: str = Body(..., embed=True),
#     data: dict = Body(..., embed=True)
# ):
#     # Verify user credentials
#     user = login(email=email, password=password)
    
#     # Add reading
#     add_reading(data)
    
#     return {"message": "Data uploaded successfully"}

@app.get("/getreading")
def recieve_reading():
    readings = get_reading()
    # Extract latitude, longitude, and average chloride value
    results = []
    for reading in readings:
        #print(reading)
        latitude = reading['location']['latitude']
        longitude = reading['location']['longitude']
        
        # Find the chloride measurement and calculate its average
        chloride_measurement = next(
            (m for m in reading['measurements'] if m['name'] == 'chloride'), None
        )
        if chloride_measurement:
            chloride_avg = sum(chloride_measurement['values']) / len(chloride_measurement['values'])
        else:
            chloride_avg = None
        
        results.append((latitude,  longitude,chloride_avg))

    # Print the results
    #print(results)

    return results

# Base and Map View Endpoints
@app.get("/")
async def home():
    return HTMLResponse("""
        <html>
            <head>
                <title>Live Heatmap</title>
            </head>
            <body>
                <h1>Welcome to the Live Heatmap Application</h1>
                <a href="/view_map">View 2D Heatmap</a><br>
            </body>
        </html>
    """)

@app.get("/view_map")
async def view_map():
    return FileResponse("live_map.html")

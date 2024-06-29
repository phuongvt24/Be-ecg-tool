from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
from pretrain.predict import predict_eff, predict_cdil, predict_resnet18
import os
import sys
import numpy as np
import pandas as pd
from pretrain.fusion_test import predict_fusion
from pretrain.fusion_test_2 import predict_fusion_att

pretrain_path = os.path.join(os.path.dirname(__file__), 'pretrain')
if pretrain_path not in sys.path:
    sys.path.append(pretrain_path)
app = FastAPI()


@app.get("/hello")
def read_root():
    text = print_test()
    print(print_test)
    return {"text: ", text}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict-ecg")
async def predict_ecg(files: list[UploadFile] = File(...), type: str = Form(...)):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Two files are required: y_test and x_test.")
    
    df1 = df2 = None
    for file in files:
        # Check if the uploaded file is a CSV
        if file.content_type != 'text/csv':
            raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}. Only CSV files are allowed.")
        
        try:
            # Read the file content
            contents = await file.read()
            # Convert bytes data to StringIO for pandas
            csv_data = io.StringIO(contents.decode('utf-8'))
            # Use pandas to read the CSV data
            df = pd.read_csv(csv_data)

            # Determine the type of file and assign to df1 or df2
            if 'y_test' in file.filename:
                df2 = df
            elif 'x_test' in file.filename:
                df1 = df
            else:
                raise HTTPException(status_code=400, detail=f"Unexpected file name: {file.filename}. Expected 'y_test' or 'x_test'.")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process CSV file {file.filename}. Error: {str(e)}")
    
    # Ensure both DataFrames were successfully created
    if df1 is None or df2 is None:
        raise HTTPException(status_code=400, detail="Both 'y_test' and 'x_test' files must be provided.")

    if (type=='eff'):
        y_trues, y_preds = predict_eff(df1, df2)
    elif (type=='cdil'):
        y_trues, y_preds = predict_cdil(df1, df2)
    elif (type=='resnet18'):
        y_trues, y_preds = predict_resnet18(df1, df2)
    elif (type=='fusion'):
         y_trues, y_preds = predict_fusion(df1, df2)
    elif (type=='att-fusion'):
         y_trues, y_preds = predict_fusion_att(df1, df2)
         
    if isinstance(y_trues, np.ndarray):
        y_trues = y_trues.astype(int).tolist()
    else:
        y_trues = [int(value) for value in y_trues]

    if isinstance(y_preds, np.ndarray):
        y_preds = y_preds.astype(int).tolist()
    else:
        y_preds = [int(value) for value in y_preds]

    data = {
        'y_trues': y_trues,
        'y_preds': y_preds,
        'type': type
    }

    return JSONResponse(content=data)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, List
from fastapi import File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pretrain.predict import predict_eff, predict_cdil, predict_resnet18
from pretrain.fusion_test import predict_fusion
from pretrain.fusion_test_2 import predict_fusion_att
from fastapi.staticfiles import StaticFiles

pretrain_path = os.path.join(os.path.dirname(__file__), 'pretrain')
if pretrain_path not in sys.path:
    sys.path.append(pretrain_path)

app = FastAPI()

# Add CORS middleware to allow connections to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Specify the origin of the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/predict-ecg")
async def predict_ecg(files: List[UploadFile] = File(...), types: List[str] = Form(...)):
    df1 = df2 = None
    for file in files:
        if file.content_type != 'text/csv':
            raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}. Only CSV files are allowed.")
        
        try:
            contents = await file.read()
            csv_data = io.StringIO(contents.decode('utf-8'))
            df = pd.read_csv(csv_data)

            if 'true_label' in file.filename:
                df2 = df
            else:
                df1 = df
            # else:
            #     raise HTTPException(status_code=400, detail=f"Unexpected file name: {file.filename}. Expected 'y_test' or 'x_test'.")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process CSV file {file.filename}. Error: {str(e)}")
    
    # if df1 is None or df2 is None:
    #     raise HTTPException(status_code=400, detail="Both 'y_test' and 'x_test' files must be provided.")

    lead_mapping = {
        'channel-0': 'DI',
        'channel-1': 'DII',
        'channel-2': 'DIII',
        'channel-3': 'AVR',
        'channel-4': 'AVL',
        'channel-5': 'AVF',
        'channel-6': 'V1',
        'channel-7': 'V2',
        'channel-8': 'V3',
        'channel-9': 'V4',
        'channel-10': 'V5',
        'channel-11': 'V6'
    }

    output_dir = os.path.join(os.path.dirname(__file__), 'pretrain', 'images')
    os.makedirs(output_dir, exist_ok=True)

    for channel, lead in lead_mapping.items():
        plt.figure(figsize=(15, 4))
        plt.plot(df1[channel], color='black')
        plt.title(f'Lead {lead}', fontsize=12, loc='left')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('mV', fontsize=10)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='red')
        plt.gca().set_facecolor('#ffcccc')
        plt.xticks(ticks=range(0, 4101, 250), fontsize=8)
        min_val = df1[channel].min()
        max_val = df1[channel].max()
        y_ticks = [i * 0.5 for i in range(int(min_val * 2), int(max_val * 2) + 1)]
        plt.yticks(y_ticks, fontsize=8)
        plt.xlim(0, 4100)
        plt.minorticks_on()
        plt.savefig(f'{output_dir}/{lead}.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close()

    results = {}

    for model_type in types:
        print('y_trues: ', model_type)
        if model_type == 'efficientnet':
            y_trues, y_preds = predict_eff(df1, df2)
        elif model_type == 'cdil':
            y_trues, y_preds = predict_cdil(df1, df2)
        elif model_type == 'resnet18':
            y_trues, y_preds = predict_resnet18(df1, df2)
        elif model_type == 'lateFusion':
            y_trues, y_preds = predict_fusion(df1, df2)
        elif model_type == 'attFusion':
            y_trues, y_preds = predict_fusion_att(df1, df2)
        else:
            continue

        results[model_type] = {
            "y_trues": y_trues.tolist() if isinstance(y_trues, np.ndarray) else y_trues,
            "y_preds": y_preds.tolist() if isinstance(y_preds, np.ndarray) else y_preds
        }

    return JSONResponse(content=results)


app.mount("/images", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'pretrain', 'images')), name="images")

import base64

@app.get("/list-images")
async def list_images_base64():
    images_dir = os.path.join(os.path.dirname(__file__), 'pretrain', 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_data = []
    for file in image_files:
        file_path = os.path.join(images_dir, file)
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_data.append({
                'name': file,
                'data': f'data:image/png;base64,{encoded_string}'
            })
    return image_data
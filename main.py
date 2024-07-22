from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv
import tempfile
import traceback
import dlib
import numpy as np
import cv2


app = FastAPI()

# .envファイルから環境変数を読み込む（開発環境用）
load_dotenv()

# 環境変数からAPIキーを取得
API_KEY = os.getenv("FACE_AUTH_KEY")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# API認証
def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# リクエストされた画像を一時ファイルとして保存
async def save_temp_file(file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp.flush()                        # 全てのデータが書き込まれている事を確認
            temp_path = temp.name
        return temp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving temporary file: {e}")

# 顔認証を行うための準備処理(RGB変換)
def convert_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # BGRからRGBに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


@app.post("/get_face_fures")
async def get_face_features(
    account: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):


    # モデルファイルのパス
    base_dir = os.path.dirname(os.path.abspath(__file__))
    shape_predictor_path = os.path.join(base_dir, "models", "shape_predictor_68_face_landmarks.dat")
    face_rec_model_path = os.path.join(base_dir, "models", "dlib_face_recognition_resnet_model_v1.dat")

    # Dlibのモデルをロード
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(shape_predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    
    try:
        # 一時ファイルに画像を保存
        temp_path = await save_temp_file(file)
        print(f'画像の場所: {temp_path}')
            
        # 画像を読込み,RGBに変換
        img_rgb = convert_image(temp_path)

        # 画像のデータ型と形状を確認
        print(f"Image dtype: {img_rgb.dtype}, shape: {img_rgb.shape}")
        if img_rgb.dtype != np.uint8 or img_rgb.shape[2] != 3:
            raise HTTPException(status_code=500, detail="Unsupported image type. Image must be 8bit RGB.")
        
        # 顔検出
        dets = detector(img_rgb, 1)
        if len(dets) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        
        shape = sp(img_rgb, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
        face_descriptor_list = list(face_descriptor)
        
        # 一時ファイルを削除
        os.remove(temp_path)
        
    except Exception as e:
        # エラーが発生した場合、詳細なエラーメッセージをログに出力
        print(f"Error processing images with HOG+SVM: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing images with HOG+SVM: {e}")
    
    finally:
        if os.path.isfile(temp_path):
            # 一時ファイルを削除
            os.remove(temp_path)

    measure_results = {
        'account': account,                             # アカウント名
        'feature': face_descriptor_list                 # 特徴量
    }

    return measure_results



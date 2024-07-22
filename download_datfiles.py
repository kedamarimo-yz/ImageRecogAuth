import gdown
import os

def download_shape_predictor():
    url_68 = 'https://drive.google.com/file/d/10bagrgwEwoYaOHDijh8vkfNsIL36pmVM/view?usp=drive_link'
    output_68 = 'shape_predictor_68_face_landmarks.dat'
    url_dlibreset = 'https://drive.google.com/file/d/1ZF76zShtA0ieK5ic-rKPqm5WEL-BIO4s/view?usp=drive_link'
    output_dlibrest = 'dlib_face_recognition_resnet_model_v1.dat'
    
    # 68ダウンロード
    if not os.path.exists(output_68):
        gdown.download(url_68, output_68, quiet=False)
        print(f"Downloaded {output_68}")
    else:
        print(f"{output_68} already exists.")
        
    # dlibresetダウンロード
    if not os.path.exists(output_dlibrest):
        gdown.download(url_dlibreset, output_dlibrest, quiet=False)
        print(f"Downloaded {output_dlibrest}")
    else:
        print(f"{output_dlibrest} already exists.")

if __name__ == "__main__":
    download_shape_predictor()

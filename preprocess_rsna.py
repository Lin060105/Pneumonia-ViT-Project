"""
External Validation Preparer (RSNA Pneumonia Dataset)
此腳本用於將 RSNA 的 DICOM 醫療影像格式轉換為 PNG，以供 ViT 模型進行跨院外部驗證 (External Validation)。
"""
import os
import pydicom
import cv2
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    rsna_dir = 'data/rsna'
    labels_csv = os.path.join(rsna_dir, 'stage_2_train_labels.csv')
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    output_dir = 'data/rsna_processed'
    
    if not os.path.exists(labels_csv) or not os.path.exists(img_dir):
        logging.warning("⚠️ 找不到 RSNA 資料集。請先至 Kaggle 下載 RSNA Pneumonia Detection Challenge。")
        logging.warning("此腳本展示了本專案處理 DICOM 並進行跨院外部驗證的擴展能力。")
        return

    logging.info("🔄 開始處理 RSNA DICOM 影像...")
    labels = pd.read_csv(labels_csv)
    binary_labels = labels.groupby('patientId')['Target'].max()

    for patient_id, target in binary_labels.items():
        dcm_path = os.path.join(img_dir, f'{patient_id}.dcm')
        if not os.path.exists(dcm_path): continue
            
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        target_folder = os.path.join(output_dir, 'PNEUMONIA' if target == 1 else 'NORMAL')
        os.makedirs(target_folder, exist_ok=True)
        cv2.imwrite(os.path.join(target_folder, f'{patient_id}.png'), img)
        
    logging.info("✅ RSNA 外部資料集轉換完成！可以使用 evaluate_binary.py 進行跨院測試。")

if __name__ == '__main__':
    main()
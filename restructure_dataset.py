import os
import shutil
from glob import glob

def restructure_dataset(base_dir):
    """
    將 chest_xray 資料夾中的 PNEUMONIA 子資料夾根據檔名
    自動劃分為 BACTERIA 和 VIRUS 兩個新的子資料夾。
    """
    print("開始重組資料集...")
    
    # 遍歷 train, test, val 三個資料集
    for split in ['train', 'test', 'val']:
        pneumonia_path = os.path.join(base_dir, split, 'PNEUMONIA')
        
        if not os.path.exists(pneumonia_path):
            print(f"警告: 找不到路徑 {pneumonia_path}，跳過。")
            continue
            
        # 建立新的 BACTERIA 和 VIRUS 資料夾
        bacteria_path = os.path.join(base_dir, split, 'BACTERIA')
        virus_path = os.path.join(base_dir, split, 'VIRUS')
        os.makedirs(bacteria_path, exist_ok=True)
        os.makedirs(virus_path, exist_ok=True)
        
        # 獲取所有 PNEUMONIA 圖片
        image_files = glob(os.path.join(pneumonia_path, '*.jpeg'))
        moved_bacteria = 0
        moved_virus = 0
        
        for img_file in image_files:
            filename = os.path.basename(img_file)
            if 'bacteria' in filename:
                shutil.move(img_file, os.path.join(bacteria_path, filename))
                moved_bacteria += 1
            elif 'virus' in filename:
                shutil.move(img_file, os.path.join(virus_path, filename))
                moved_virus += 1
                
        print(f"在 {split} 資料集中:")
        print(f"  - 移動了 {moved_bacteria} 張細菌性肺炎圖片至 BACTERIA 資料夾。")
        print(f"  - 移動了 {moved_virus} 張病毒性肺炎圖片至 VIRUS 資料夾。")
        
        # 移除空的 PNEUMONIA 資料夾
        try:
            if not os.listdir(pneumonia_path):
                os.rmdir(pneumonia_path)
                print(f"  - 已移除空的 PNEUMONIA 資料夾。")
        except OSError as e:
            print(f"移除 {pneumonia_path} 時出錯: {e}")
            
    print("\n資料集重組完成！")

if __name__ == '__main__':
    # 專案根目錄下的 chest_xray 資料夾
    data_directory = 'chest_xray'
    restructure_dataset(data_directory)
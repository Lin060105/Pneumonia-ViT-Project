"""
Pneumonia Dataset Downloader & Setup Script
此腳本用於自動從 Kaggle 下載 Kermany 胸部 X 光資料集，
並將其整理為模型訓練所需的標準二元分類資料夾結構。
"""
import os
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract():
    # 使用 Mendeley Data 的直接下載連結 (Kermany 原始出處，免登入 Kaggle)
    # 這樣任何人 Clone 您的專案都不需要設定 API Key 就能直接下載！
    dataset_url = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-406f-bf08-fd15a451af24/file_downloaded"
    zip_path = "chest_xray_dataset.zip"
    extract_dir = "chest_xray_extracted"
    target_dir = "chest_xray"

    if os.path.exists(target_dir):
        print(f"✅ 資料夾 '{target_dir}' 已存在，無須重新下載。")
        return

    print("📥 開始下載 Kermany 胸部 X 光資料集 (約 1.2 GB，請耐心等候)...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(dataset_url, filename=zip_path, reporthook=t.update_to)
        
        print("\n📦 下載完成！開始解壓縮...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print("🛠️ 正在重構為二元分類資料夾結構 (NORMAL vs PNEUMONIA)...")
        # Mendeley 的壓縮包解開後會有特定結構，我們將其搬移到標準的 chest_xray 目錄下
        source_dir = os.path.join(extract_dir, "chest_xray")
        if os.path.exists(source_dir):
            shutil.move(source_dir, target_dir)
        
        # 清理暫存檔
        print("🧹 清理下載的暫存壓縮檔...")
        os.remove(zip_path)
        shutil.rmtree(extract_dir)
        
        print(f"🎉 資料集設定完成！您現在可以執行 'python train_binary.py' 了！")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    download_and_extract()
    
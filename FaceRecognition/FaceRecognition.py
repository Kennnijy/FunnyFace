import os
import re

import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image

# 1. 定義資料夾路徑（若在 Colab，記得改成 /content/drive/MyDrive/...）
BASE_DIR = "faceBase"
UNKNOWN_DIR = "faceUnknown"
RESULT_DIR = "faceResult"  # 儲存結果的資料夾

# 自動檢查並建立結果資料夾
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"已建立結果儲存資料夾：{RESULT_DIR}")

# 2. 取得未知照片檔案清單
unknown_files = [
    f
    for f in os.listdir(UNKNOWN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

total_files = len(unknown_files)
print(f"发现待辨识照片：{total_files} 张")
print(f"开始進行人臉精準搜尋，請稍候...\n")

# 3. 巡覽每一張未知的照片（一次全部衝完）
for index, unknown_file in enumerate(unknown_files, start=1):
    unknown_img_path = os.path.join(UNKNOWN_DIR, unknown_file)
    best_match_name = "Unknown"
    
    print(f"【{index}/{total_files}】正在辨識：{unknown_file} ...")

    try:
        # 使用 DeepFace.find 比對整個 faceBase 資料夾
        dfs = DeepFace.find(
            img_path=unknown_img_path,
            db_path=BASE_DIR,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True  # 讓 DeepFace 內建的進度條不要洗版終端機
        )

        # 如果有找到匹配的資料 (DataFrame 不為空)
        if len(dfs) > 0 and not dfs[0].empty:
            # 取得最匹配的第一筆資料路徑 (identity)
            match_path = dfs[0].iloc[0]["identity"]
            # 從完整路徑中切出純檔名
            base_file = os.path.basename(match_path)

            # 自動解析名字：移除數字與副檔名
            name_clean = os.path.splitext(base_file)[0]
            name_clean = re.sub(r"\d+", "", name_clean)
            name_clean = name_clean.replace("_", "").strip()

            best_match_name = name_clean

    except Exception as e:
        print(f" ➔ 【偵測提示】檔案 {unknown_file} 處理時跳過: {e}")
        best_match_name = "Unknown"

    # 4. 後台繪製結果視窗、儲存照片（不阻礙程式執行）
    try:
        img = Image.open(unknown_img_path)
        plt.figure()
        plt.imshow(img)

        # 設定圖片標題
        plt.title(f"Predicted Name: {best_match_name}", fontsize=14, color="blue")
        plt.axis("off")

        # 💡 修改：主檔名加上原檔名，避免同名覆蓋，並保留原副檔名識別度
        pure_name = os.path.splitext(unknown_file)[0]
        save_path = os.path.join(RESULT_DIR, f"result_{pure_name}_{best_match_name}.png")
        
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()  # 💡 關鍵：直接關閉畫布不呼叫 plt.show()，讓程式全速往下跑
        
    except Exception as img_err:
        print(f" ➔ 畫圖或存檔失敗: {img_err}")

    # 5. 終端機即時回報
    print(f" ➔ 辨識結果: {best_match_name}\n")

print("🎉 全部照片已一次性辨識完畢！請至 faceResult 資料夾查看所有結果圖。")
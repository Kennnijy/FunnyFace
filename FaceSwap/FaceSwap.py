import os  # 👈 新增：用於建立資料夾與處理路徑
import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image

# 1. 定義圖片路徑與儲存資料夾（若在 Colab，記得改成 /content/drive/MyDrive/...）
imgFile4 = "images.jpg"  # 確保圖片名稱與你的檔案一致
RESULT_DIR = "faceSwapResult"  # 👈 新增：儲存換臉結果的資料夾

# 👈 自動檢查並建立結果資料夾
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"已建立結果儲存資料夾：{RESULT_DIR}")

# 2. 讀取原始圖片與初始化
original_img = Image.open(imgFile4)
output_img = original_img.copy()

# 3. 使用 retinaface 偵測器抓取人臉
print("正在偵測人臉並下載/載入模型，請稍候...")
faces = DeepFace.extract_faces(
    imgFile4, detector_backend="retinaface", enforce_detection=False
)

face_clips = []
faces_positions = []

# 4. 裁剪所有偵測到的臉孔並記錄位置
for face in faces:
    area = face["facial_area"]
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    box = (x, y, x + w, y + h)
    faces_positions.append(box)
    face_clips.append(original_img.crop(box))

# 5. 進行臉部位置對調與貼上
if len(faces) >= 2:
    print(f"成功偵測到 {len(faces)} 張人臉，開始進行位置交換...")

    # 交換邏輯：最後一張移到最前面，其餘往後挪
    swapped_clips = [face_clips[-1]] + face_clips[:-1]

    for i, box in enumerate(faces_positions):
        # 計算目標框的寬度與高度
        target_w = box[2] - box[0]
        target_h = box[3] - box[1]

        # 調整對調後的臉部大小以符合目標區域
        resized_face = swapped_clips[i].resize((target_w, target_h))

        # 將調整後的臉部貼回輸出圖片的指定坐標 (左, 上)
        output_img.paste(resized_face, (box[0], box[1]))

    # 6. 顯示最終換臉結果並儲存照片
    plt.figure()  # 💡 建立新畫布
    plt.imshow(output_img)
    plt.axis("off")

    # 👈 新增：動態產生儲存路徑並存檔
    # 這裡直接用 PIL 的 save 功能保存高畫質的原圖結果
    pure_name = os.path.splitext(os.path.basename(imgFile4))[0]
    save_path = os.path.join(RESULT_DIR, f"swapped_{pure_name}.png")
    output_img.save(save_path)
    print(f"💾 換臉結果圖已儲存至：{save_path}")

    plt.show()
    plt.close()  # 關閉畫布
    print("換臉完成！")
else:
    print(f"偵測到的人臉數量不足（僅偵測到 {len(faces)} 個），無法進行交換。")
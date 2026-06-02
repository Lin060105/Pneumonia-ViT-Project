# 零基礎保母級教學手冊：胸腔 X 光肺炎 AI 專案到底在幹嘛？

這份文件是寫給「完全零基礎，但想真的理解這個專案」的你。你不需要先會寫程式，也不需要先懂醫學影像或機器學習。我會把整個專案想像成一間廚房、一台遊戲主機、一套考試制度，慢慢拆開講。

先講最重要的一句話：這個專案不是做一台可以取代醫師的診斷機器。它是一個研究原型，用 AI 看胸腔 X 光，輸出「這張片子像不像肺炎」的機率，並且用很多嚴格方法去檢查它到底可靠到什麼程度。

---

## 0. 這個專案一句話在做什麼？

這個專案做的是：

> 給電腦一張胸腔 X 光片，讓模型判斷它比較像 NORMAL 還是 PNEUMONIA，並輸出肺炎機率。

更白話一點：

> 我們教一個 AI 看很多張「正常」和「肺炎」的 X 光，讓它慢慢學會哪些影像特徵比較像肺炎。學完後，再拿它沒看過的新影像考它，看它答得準不準。

這裡有幾個角色：

| 角色 | 白話意思 |
|---|---|
| X 光影像 | 考題 |
| NORMAL / PNEUMONIA label | 正確答案 |
| AI model | 正在學習的學生 |
| Training | 讓學生刷題 |
| Validation | 小考，用來調整讀書方法 |
| Test | 期末考，不能偷看答案 |
| External validation | 換一間學校考，看是不是真的會 |

這個專案目前已經做完：

- 內部資料集：Kermany 兒童胸腔 X 光。
- 外部資料集：RSNA Pneumonia Detection Challenge。
- 模型：ResNet18、ResNet50、DenseNet121、EfficientNet-B0、ViT-B/16。
- 評估：Sensitivity、Specificity、AUC、PPV、NPV、Calibration、Decision Curve、Grad-CAM。
- 論文草稿：`docs/manuscript_draft.md`。
- 教學手冊：你正在看的這份 `docs/project_tutorial.md`。

---

## 1. 基礎環境與硬體大解密

### 1.1 什麼是終端機 Terminal？

終端機就是你用文字跟電腦下指令的地方。

平常你用電腦，多半是用滑鼠點圖示，例如：

- 點 Chrome 打開瀏覽器。
- 點資料夾看檔案。
- 點 Word 開文件。

終端機則是另一種操作方式。你不是用滑鼠點，而是直接打字告訴電腦：

```powershell
python train_binary.py
```

這句話的意思大概是：

> 電腦，請你用 Python 去執行 `train_binary.py` 這個檔案。

你可以把終端機想像成餐廳廚房的點餐口。

- 你打指令，就像把點菜單交給廚房。
- 電腦收到指令，就開始照著做。
- 終端機跑出文字，就是廚房回報「已開始做」、「做完了」、「材料不夠」、「這道菜名我看不懂」。

如果指令成功，通常會看到結果或回到下一行。

如果指令失敗，會看到錯誤訊息。錯誤訊息不是電腦在罵你，而是在告訴你：「我做到哪一步卡住了。」

### 1.2 我打指令進去時，電腦在幹嘛？

以這句為例：

```powershell
python evaluate_binary.py --model-path saved_models\baselines_full\vit-b_16_seed42_best.pth
```

電腦大概做了這些事情：

1. 找到 `python` 這個程式。
2. 用 Python 打開 `evaluate_binary.py`。
3. 讀取後面的參數，例如模型檔案在哪裡。
4. 載入模型。
5. 讀取測試影像。
6. 對每張影像做預測。
7. 算出 Accuracy、Sensitivity、Specificity、AUC 等指標。
8. 把結果寫到 `results/` 資料夾。

所以終端機不是神秘黑盒子，它只是用文字操作電腦。

### 1.3 什麼是 Python？

Python 是一種程式語言。

程式語言就是人類跟電腦溝通的語言。人類講中文、英文；電腦最後只懂 0 和 1。Python 的角色就是讓我們用比較接近人類邏輯的方式寫指令，再由電腦翻譯成能執行的動作。

例如 Python 可以寫：

```python
print("Hello")
```

意思是：

> 請在畫面上印出 Hello。

在這個專案中，Python 負責：

- 讀取 X 光影像。
- 建立 AI 模型。
- 訓練模型。
- 評估模型。
- 畫圖。
- 輸出 CSV 報表。
- 啟動 Streamlit demo app。

你可以把 Python 想像成這個專案的主要工作語言。

### 1.4 什麼是 PyTorch？

PyTorch 是一個用 Python 寫 AI 和深度學習的工具箱。

如果 Python 是中文，那 PyTorch 就像「醫學影像 AI 專用詞典加工具箱」。它幫我們處理很多很難的數學和 GPU 運算。

沒有 PyTorch 的話，你要自己從零寫：

- 神經網路每一層怎麼算。
- 圖片怎麼轉成矩陣。
- 模型怎麼更新權重。
- GPU 怎麼平行運算。
- loss 怎麼反向傳播。

有 PyTorch 後，你可以比較簡潔地寫：

```python
loss.backward()
optimizer.step()
```

這背後其實是在做大量微積分和矩陣運算，但 PyTorch 幫你包好了。

Python 和 PyTorch 的關係可以這樣理解：

| 名詞 | 比喻 |
|---|---|
| Python | 你會說的語言 |
| PyTorch | 用這個語言寫成的 AI 工具箱 |
| 模型 | 用工具箱組出來的機器 |
| GPU | 幫機器加速的大型引擎 |

### 1.5 什麼是 GPU？

GPU 是顯示卡。

平常你可能覺得顯示卡是用來打遊戲、跑 SolidWorks、顯示畫面。但對 AI 來說，GPU 更像是一台超會做大量重複計算的計算機。

AI 訓練時需要做很多矩陣運算。矩陣你可以想像成一大格表格，裡面都是數字。X 光影像也是一大格數字。神經網路也是一堆數字。訓練模型就是不停讓這些巨大表格互相計算。

CPU 像少數幾個很聰明的員工，適合處理複雜、零碎、需要判斷的工作。

GPU 像幾千個會同時做簡單加減乘除的員工，適合處理大量重複工作。

所以訓練 AI 時，GPU 通常比 CPU 快很多。

### 1.6 什麼是 CUDA？

CUDA 是 NVIDIA 顯示卡用來跑深度學習計算的技術。

你可以這樣想：

- GPU 是跑車。
- CUDA 是讓 Python / PyTorch 真的能踩油門開跑車的駕照和道路系統。

如果電腦有 NVIDIA GPU，但沒有安裝對應的 CUDA 版 PyTorch，模型可能還是只能用 CPU 跑。這就像你有跑車，但沒有鑰匙，只能牽著走。

這個專案後來已經安裝 CUDA 版 PyTorch，所以可以用你的 RTX 4060 Laptop GPU 訓練。

### 1.7 為什麼跑模型時電腦會到 80 度？

因為 GPU 正在滿載工作。

訓練模型時，GPU 會連續做大量矩陣運算，功耗會升高，風扇會變大聲，溫度也會上升。筆電 GPU 跑到 70-80 度並不罕見，尤其是長時間訓練 ViT 這種比較大的模型。

你可以想像成：

> 平常電腦只是散步，訓練模型是在跑長跑。

跑長跑會流汗，GPU 也會發熱。

但要注意：

- 放在通風處。
- 不要蓋住出風口。
- 長時間訓練時接電源。
- 若溫度長時間過高，可以降低 batch size 或改用 CPU / 短 epoch 測試。

---

## 2. Git 與 GitHub 時光機系統

### 2.1 什麼是 Git？

Git 是程式碼的版本控制系統。

白話說，Git 是專案的時光機。

你寫報告時可能會有這種檔名：

- 報告.docx
- 報告_新版.docx
- 報告_新版2.docx
- 報告_真的最後版.docx
- 報告_真的最後版_教授修改後.docx

這很混亂。你不知道哪個版本是最新，哪個版本能用，哪個版本改了什麼。

Git 幫你做的事情就是：

- 記錄每次重要修改。
- 讓你知道哪些檔案改過。
- 讓你回到以前的版本。
- 讓多人可以一起改同一個專案。

### 2.2 什麼是 GitHub？

Git 和 GitHub 很容易被搞混。

| 名詞 | 是什麼 | 比喻 |
|---|---|---|
| Git | 版本控制工具 | 你電腦裡的遊戲存檔系統 |
| GitHub | 放程式碼的平台 | 雲端存檔和作品展示櫃 |

Git 可以完全在你自己的電腦上運作，不一定需要網路。

GitHub 是網站，讓你把 Git 管理的專案上傳到雲端，方便備份、分享、合作。

一句話：

> Git 是工具，GitHub 是放 Git 專案的網路平台。

### 2.3 為什麼工程師一定要用 Git？

因為寫程式很容易改壞。

你今天可能改一個小功能，結果明天發現整個 app 打不開。如果沒有 Git，你只能憑記憶把檔案改回來，這很痛苦。

Git 像遊戲存檔：

- 打王前先存檔。
- 失敗了可以讀檔。
- 發現新路線比較好，可以另開分支。
- 最後保留最成功的路線。

在專案裡，每次 `commit` 就像一次正式存檔。

### 2.4 什麼是 `git add`？

`git add` 的意思是：

> 我想把這個檔案放到下一次存檔清單裡。

你可以想像你要寄包裹。

`git add` 就是把東西放進紙箱，但還沒有封箱寄出。

例如：

```powershell
git add docs\manuscript_draft.md
```

意思是：

> 下一次 commit 時，請把這份論文草稿的修改一起存起來。

### 2.5 什麼是 `git commit`？

`git commit` 就是正式存檔。

例如：

```powershell
git commit -m "Add manuscript and tutorial docs"
```

這句話的意思是：

> 把剛才 add 的內容正式記錄成一個版本，並附上說明：Add manuscript and tutorial docs。

commit message 很重要，因為以後你回頭看歷史紀錄時，會知道每次修改在做什麼。

### 2.6 什麼是 `git push`？

`git push` 是把本機 commit 推到 GitHub。

用包裹比喻：

- `git add`：把東西放進紙箱。
- `git commit`：封箱並貼標籤。
- `git push`：把箱子寄到雲端倉庫。

所以 `git push` 不是存檔本身，它是把已經存好的版本上傳。

### 2.7 什麼是 `.gitignore`？

`.gitignore` 是一份清單，告訴 Git：

> 這些檔案不要追蹤，也不要上傳。

為什麼需要它？

因為專案裡有些東西不適合放上 GitHub，例如：

- 資料集影像。
- 超大模型權重。
- 暫存檔。
- log 檔。
- 個人電腦環境設定。

### 2.8 為什麼 `.pth` 模型權重不能隨便傳上 Git？

`.pth` 是 PyTorch 模型 checkpoint，通常很大。

這個專案裡 ViT 模型權重大約幾百 MB。GitHub 一般 repo 不適合放這種大檔案，原因是：

- repo 會變很肥。
- 每次 clone 都很慢。
- Git 歷史會永久記住大檔。
- GitHub 對單檔大小有限制。

如果真的要管理大模型，通常用：

- Git LFS。
- Hugging Face Hub。
- 雲端硬碟。
- 實驗追蹤平台。

---

## 3. 專案檔案與資料庫

### 3.1 電腦怎麼看懂一張 X 光片？

電腦其實看不懂「肺」、「骨頭」、「發炎」這些概念。

電腦看到的是數字。

一張黑白 X 光可以想成一張超大的格子紙。每一格叫一個 pixel，也就是像素。

每個像素有一個亮度值：

- 0 可能代表很黑。
- 255 可能代表很白。
- 中間數字代表不同灰階。

所以一張 X 光在電腦眼中其實像這樣：

```text
12  15  18  20
10  14  90  95
8   12  88  91
```

真正的影像會大很多，例如 224 x 224，就有 50,176 個像素。模型學習的就是這些數字之間的規律。

### 3.2 什麼是矩陣？

矩陣就是一個數字表格。

一張影像可以是一個矩陣，一批影像可以是一疊矩陣。AI 模型做的事情，就是把這些矩陣丟進很多層數學運算，最後輸出一個機率。

在這個專案中，模型最後輸出：

```text
P(PNEUMONIA) = 0.87
```

意思是：

> 模型覺得這張影像像肺炎的程度是 0.87。

如果門檻 threshold 是 0.5，那 0.87 大於 0.5，所以判成 PNEUMONIA。

### 3.3 什麼是 Kaggle？

Kaggle 是一個資料科學和機器學習平台。

它有很多公開資料集，也有很多比賽。你可以把 Kaggle 想成：

> AI 界的練功場和資料集圖書館。

在這個專案裡，Kermany 和 RSNA 資料都可以透過 Kaggle 或相關公開來源取得。

### 3.4 什麼是 CSV？

CSV 是 Comma-Separated Values 的縮寫，意思是「用逗號分隔的表格」。

它看起來像這樣：

```csv
filename,label
image001.jpeg,NORMAL
image002.jpeg,PNEUMONIA
```

你可以用 Excel 打開 CSV，但 CSV 本身比 Excel 簡單很多。

| 格式 | 特點 |
|---|---|
| CSV | 純文字，輕量，適合程式讀寫 |
| Excel | 可以有公式、格式、顏色、工作表 |

在 AI 專案中，CSV 常用來存：

- 影像路徑。
- 標籤。
- 預測機率。
- 評估指標。

### 3.5 Label 是什麼？

Label 就是答案。

如果 X 光是考題，label 就是標準答案。

例如：

```csv
path,true_label
chest_xray/test/NORMAL/IM-0001.jpeg,NORMAL
chest_xray/test/PNEUMONIA/person10.jpeg,PNEUMONIA
```

模型訓練時會看影像和 label，慢慢學會：

> 什麼樣的數字模式通常對應 NORMAL，什麼樣的數字模式通常對應 PNEUMONIA。

### 3.6 什麼是 `.pth`？

`.pth` 是 PyTorch 常用的模型檔案格式。

它裡面通常存的是 model checkpoint。

checkpoint 可以想像成：

> AI 學生讀完書後，腦袋目前狀態的存檔。

模型訓練不是把圖片整張背起來，而是調整很多很多參數。這些參數叫 weights，也就是權重。

權重可以想像成模型腦中的旋鈕。

- 某些旋鈕負責看邊緣。
- 某些旋鈕負責看紋理。
- 某些旋鈕負責看影像區域關係。

訓練就是不停調整這些旋鈕。`.pth` 檔就是把調整好的旋鈕位置存起來。

---

## 4. 資料集與訓練機制

### 4.1 Kermany 和 RSNA 差在哪？

這是整個專案最重要的觀念之一。

Kermany 是內部資料集，主要是兒童胸腔 X 光。RSNA 是外部資料集，以成人為主，來自不同資料來源和標註任務。

| 項目 | Kermany | RSNA |
|---|---|---|
| 角色 | 內部資料 | 外部驗證 |
| 族群 | 兒童為主 | 成人為主 |
| 任務 | NORMAL vs PNEUMONIA 分類 | 肺炎 opacity 偵測挑戰轉二元分類 |
| 影像格式 | JPEG | DICOM |
| 肺炎盛行率 | internal test 62.5% | external 22.5% |

你可以想像：

> 模型在兒童醫院的考題練得很好，後來被拿去成人大醫院考試。題型、族群、影像風格都變了，所以分數下降是合理且重要的發現。

這就叫 domain gap。

### 4.2 Train、Validation、Test 是什麼？

這三個很像學生準備考試的三階段。

#### Train 訓練集

訓練集是給模型學習用的資料。

模型可以反覆看訓練集，做錯了就修正。

比喻：

> 課本習題。

#### Validation 驗證集

驗證集是小考。

它不是最終成績，但用來幫我們決定：

- 哪個模型比較好。
- 要不要停訓練。
- 哪個 checkpoint 最值得留下。

比喻：

> 模擬考。

#### Test 測試集

測試集是期末考。

模型不應該在訓練和調參時偷看 test set。否則成績會不公平。

比喻：

> 真正考試。

### 4.3 為什麼不能用 Test set 調參？

如果你一邊看期末考答案一邊改讀書方法，最後考高分就不代表你真的會，只代表你背了那份考卷。

AI 也一樣。

如果我們用 test set 來選模型或調 threshold，那 test 成績就不再客觀。

所以正確流程是：

1. Train set：學習。
2. Validation set：選模型和調參。
3. Test set：最後一次評估。
4. External validation：換資料集確認能不能推廣。

### 4.4 Epoch 是什麼？

Epoch 是模型把整個訓練集看完一輪。

如果訓練集有 5,216 張圖片，模型從第一張看到最後一張，這叫 1 epoch。

10 epochs 就是整份訓練集看 10 輪。

但不是看越多越好。看太少學不會，看太多可能變成死背訓練集，這叫 overfitting。

### 4.5 Batch Size 是什麼？

Batch size 是模型一次看幾張圖片。

如果 batch size = 32，意思是模型一次拿 32 張圖片算一次平均錯誤，再更新權重。

比喻：

> 老師不是每寫完一題就檢討，而是寫完一小疊題目後一起檢討。

Batch size 太大：

- GPU 記憶體可能不夠。
- 訓練可能快一點，但硬體壓力大。

Batch size 太小：

- 記憶體省。
- 訓練可能比較慢、比較不穩。

### 4.6 機器怎麼學習和修正錯誤？

模型學習大概分成四步：

1. 看圖片。
2. 猜答案。
3. 跟正確答案比較，算出錯多少。
4. 根據錯誤調整權重。

錯誤的大小叫 loss。

訓練的目標就是讓 loss 越來越小。

你可以想像模型像在調音：

- 一開始音很不準。
- 每次聽到自己唱錯，就微調一點。
- 調很多次後，越來越接近正確旋律。

---

## 5. 核心 AI 模型白話文解析

### 5.1 什麼是神經網路？

神經網路是一種由很多「小計算單元」組成的模型。

它不是大腦本身，但名字借用了神經元的概念。

一個神經網路會把輸入資料，例如影像像素，經過很多層計算，最後輸出答案。

你可以想像成工廠流水線：

- 第一站看簡單線條。
- 第二站看角落和紋理。
- 第三站看更大的形狀。
- 最後一站判斷像 NORMAL 還是 PNEUMONIA。

### 5.2 什麼是 CNN？

CNN 是 Convolutional Neural Network，中文叫卷積神經網路。

CNN 很擅長看圖片，因為它會用一個小視窗在圖片上滑來滑去，找局部特徵。

像你拿放大鏡看照片：

- 先看左上角。
- 再看右上角。
- 再看中間。
- 再看下方。

CNN 的強項是：

- 很會抓局部紋理。
- 很會找邊緣、形狀、陰影。
- 對醫學影像分類很常用。

### 5.3 ResNet18 是什麼？

ResNet 是 Residual Network。

它的核心概念是 shortcut，也就是捷徑。

一般很深的神經網路可能會有一個問題：層數太多，訊息傳到後面會變弱，訓練變困難。

ResNet 的想法是：

> 如果這一層學不到更好的東西，至少讓原本的訊息可以走捷徑傳下去。

ResNet18 是比較小的 ResNet，有 18 層左右的主要結構。

強項：

- 訓練快。
- 穩定。
- 適合當 baseline。

### 5.4 ResNet50 是什麼？

ResNet50 是更深的 ResNet。

它比 ResNet18 更大，表達能力更強，但也更吃資源。

你可以想像：

- ResNet18 像輕便筆記本。
- ResNet50 像比較厚的參考書。

參考書內容更多，但不一定每次考試都一定贏，因為資料量、訓練設定和題型都會影響結果。

### 5.5 DenseNet121 是什麼？

DenseNet 的特色是密集連接。

它不像一般網路只把上一層結果傳給下一層，而是讓很多前面層的資訊都能被後面層使用。

比喻：

> 每一位前面老師的筆記，後面的老師都可以拿來參考。

這樣的好處是：

- 特徵重複利用。
- 梯度比較容易傳遞。
- 在醫學影像任務常表現不錯。

### 5.6 EfficientNet-B0 是什麼？

EfficientNet 的設計目標是有效率。

它不是單純把模型變深或變寬，而是用比較平衡的方式同時調整：

- 深度。
- 寬度。
- 影像解析度。

EfficientNet-B0 是比較基礎的版本。

比喻：

> 不是把餐廳盲目蓋大，而是同時合理安排廚房大小、員工數量和出餐流程。

強項：

- 參數相對有效率。
- 表現常常很好。
- 在本專案內部測試中，AUC 幾乎追上 ViT。

### 5.7 什麼是 ViT-B/16？

ViT 是 Vision Transformer。

傳統 CNN 像拿放大鏡一小塊一小塊看圖片。ViT 的想法比較像：

> 先把圖片切成很多小拼圖，再讓每一塊拼圖互相討論，看看哪幾塊跟答案最有關。

ViT-B/16 裡面的 B 是 Base，代表模型大小；16 代表 patch size 是 16 x 16。

一張 224 x 224 圖片會被切成很多 16 x 16 小塊。模型會看這些小塊之間的關係。

CNN 和 ViT 的差異：

| 項目 | CNN | ViT |
|---|---|---|
| 看圖方式 | 局部滑動視窗 | 切成 patch 後看整體關係 |
| 強項 | 局部紋理、邊緣 | 長距離關係、全局注意力 |
| 資料需求 | 通常較省資料 | 通常更依賴大量資料或預訓練 |
| 本專案角色 | baseline | primary model |

---

## 6. 評估指標與圖表到底在畫什麼？

### 6.1 混淆矩陣是什麼？

混淆矩陣就是把模型答題結果分成四種。

| 名稱 | 意思 |
|---|---|
| TP | 真的肺炎，模型也說肺炎 |
| TN | 真的正常，模型也說正常 |
| FP | 其實正常，模型誤判肺炎 |
| FN | 其實肺炎，模型漏掉 |

在醫療篩檢中，FN 通常很重要，因為代表病人有病但模型沒抓到。

本專案 ViT 在內部測試：

- TP = 389
- TN = 145
- FP = 89
- FN = 1

外部 RSNA：

- TP = 5,740
- TN = 6,888
- FP = 13,784
- FN = 272

這表示外部資料上模型很少漏掉肺炎，但誤報很多。

### 6.2 Sensitivity 敏感度是什麼？

Sensitivity 又叫 recall。

它問的是：

> 所有真正肺炎的人裡，模型抓到了多少？

公式：

```text
Sensitivity = TP / (TP + FN)
```

高 sensitivity 代表不容易漏診。

本專案 primary ViT：

- Internal sensitivity = 0.997
- RSNA external sensitivity = 0.955

都很高。

### 6.3 Specificity 特異度是什麼？

Specificity 問的是：

> 所有真正正常的人裡，模型正確排除了多少？

公式：

```text
Specificity = TN / (TN + FP)
```

高 specificity 代表不容易誤報。

本專案 primary ViT：

- Internal specificity = 0.620
- RSNA external specificity = 0.333

外部特異度低，代表 RSNA 上模型把很多正常或非肺炎異常影像判成肺炎。

### 6.4 Precision / PPV 是什麼？

Precision 又叫 PPV，positive predictive value。

它問的是：

> 模型說肺炎的影像裡，真的有多少是肺炎？

公式：

```text
PPV = TP / (TP + FP)
```

本專案：

- Internal PPV = 0.814
- RSNA external PPV = 0.294

這代表在 RSNA 上，模型說「肺炎」時，真的肺炎比例只有約 29.4%。

### 6.5 為什麼 RSNA 的 PPV 會掉？

原因有兩個。

第一，盛行率下降。

內部 test set 肺炎比例是 62.5%，RSNA 是 22.5%。如果一個考場裡真的肺炎比較少，那模型只要有一點愛誤報，PPV 就會明顯下降。

第二，domain gap。

Kermany 是兒童資料，RSNA 多為成人資料，而且 RSNA 有很多「異常但不一定是肺炎」的影像。模型可能看到陰影、管線、其他疾病或成人影像特徵，就誤以為是肺炎。

用生活比喻：

> 你訓練一個人只看兒童醫院的題庫，結果拿去成人醫院考。他仍然很努力抓可疑影像，所以漏掉很少，但看到很多陌生情況時會過度緊張，於是誤報變多。

### 6.6 AUC 和 ROC 曲線是什麼？

ROC 曲線是在看：

> 如果我們把判定肺炎的門檻從很低一路調到很高，模型在 sensitivity 和 false positive rate 之間的取捨如何？

AUC 是 ROC 曲線下面積。

AUC 越接近 1，代表模型越能把正常和肺炎分開。

大概可以這樣看：

| AUC | 粗略解讀 |
|---|---|
| 0.5 | 跟亂猜差不多 |
| 0.7 | 有一定辨識力 |
| 0.8 | 不錯 |
| 0.9 以上 | 很好 |

本專案：

- Internal ROC AUC = 0.980
- RSNA external ROC AUC = 0.801

意思是模型在內部資料幾乎分得很好，但換到 RSNA 後仍有辨識能力，只是明顯下降。

### 6.7 PR AUC 是什麼？

PR 曲線看的是 Precision 和 Recall 的關係。

當資料很不平衡時，PR AUC 常比 ROC AUC 更能反映陽性預測品質。

本專案：

- Internal PR AUC = 0.986
- RSNA external PR AUC = 0.513

外部 PR AUC 明顯下降，和 PPV 下降一致。

### 6.8 Calibration 是什麼？

Calibration 是校準。

它問的是：

> 模型說 90% 有肺炎時，實際上真的約 90% 有肺炎嗎？

如果模型輸出 0.99，但實際只有 32% 是肺炎，那模型就過度自信。

本專案 RSNA 外部 ECE = 0.511，代表校準很差。外部最高機率區間 0.9-1.0 的平均預測機率約 0.994，但實際肺炎比例約 0.323。

白話：

> 模型在 RSNA 上太有自信了。它常常大喊「我超確定是肺炎」，但實際上沒那麼確定。

### 6.9 Grad-CAM 是什麼？

Grad-CAM 是一種熱力圖方法，用來看模型大概注意圖片哪裡。

紅色或亮色區域通常表示模型比較重視那裡。

但要小心：

Grad-CAM 不是模型的完整思考過程，也不是醫學證據。它只是幫我們檢查模型有沒有可能看錯地方。

例如：

- 如果模型注意肺野區域，比較合理。
- 如果模型一直注意文字標記、邊角、床板，就可能學到資料集偏差。

---

## 7. 上台 Demo 與解說指南

### 7.1 明天要展示，我該先打開什麼？

建議展示順序：

1. 打開專案資料夾。
2. 打開 `README.md` 或 `docs/project_tutorial.md`，說明專案目的。
3. 展示 `results/baselines_full_eval/internal_test_performance_summary.csv`。
4. 展示 `results/baselines_full_eval/vit_rsna_external/external_validation_metrics_report.csv`。
5. 展示 `results/gradcam_representative_cases.png`。
6. 啟動 Streamlit app 做單張或多張影像 demo。

### 7.2 如何啟動 demo app？

在終端機進到專案資料夾：

```powershell
cd D:\Pneumonia_Classification_PyTorch_L2_forCodex
```

啟動 app：

```powershell
streamlit run app_binary.py
```

如果要指定模型，可以設定環境變數：

```powershell
$env:MODEL_PATH="saved_models\baselines_full\vit-b_16_seed42_best.pth"
streamlit run app_binary.py
```

打開網頁後，你可以上傳胸腔 X 光影像，模型會輸出：

- Pneumonia probability。
- 判定結果。
- 是否落在人工複查區。
- Grad-CAM 熱力圖。
- 可下載 CSV 報告。

### 7.3 Demo 時可以怎麼說？

你可以這樣講：

> 這個專案是一個胸腔 X 光肺炎篩檢研究原型。我們使用 Kermany 兒童胸腔 X 光資料集訓練模型，並比較 ResNet、DenseNet、EfficientNet 和 Vision Transformer。主要模型 ViT-B/16 在內部測試集達到 ROC AUC 0.980、sensitivity 0.997，代表它在內部資料上非常擅長抓出肺炎影像。

接著講外部驗證：

> 但我們沒有只停在內部測試，因為醫學 AI 最怕只在自己熟悉的資料上表現好。所以我們用 RSNA 外部資料集做 validation。結果 ROC AUC 降到 0.801，PPV 降到 0.294，這說明模型遇到不同族群和不同標註定義時，會出現 domain gap。

再講價值：

> 這個結果反而是研究上重要的地方：它提醒我們，醫學 AI 不能只看高 accuracy，而要看外部驗證、校準、盛行率變化和臨床流程。這個模型目前適合定位為 high-sensitivity screening support prototype，而不是 standalone diagnostic device。

### 7.4 如果教授問：為什麼換資料集分數會變低？

你可以這樣回答：

> 這是 domain gap 的典型現象。內部 Kermany 資料集主要是兒童胸腔 X 光，而且是整理過的 NORMAL/PNEUMONIA 分類任務；RSNA 則以成人胸腔 X 光為主，標籤來自肺部 opacity 偵測挑戰，還包含許多 abnormal but not pneumonia 的影像。兒童和成人在肺部大小、影像拍攝條件、疾病表現和非肺炎異常比例上都不同，所以模型在 Kermany 學到的特徵不一定能完整轉移到 RSNA。這也是為什麼我們必須做 external validation，而不能只報 internal accuracy。

如果教授繼續問 PPV：

> PPV 下降也跟 prevalence shift 有關。Kermany internal test 的肺炎盛行率是 62.5%，RSNA 是 22.5%。當真實肺炎比例下降，只要模型 specificity 不夠高，false positives 就會大量增加，因此 PPV 會明顯下降。這不是單純公式問題，而是臨床部署時一定會遇到的真實問題。

### 7.5 如果教授問：那這模型還有價值嗎？

可以回答：

> 有，但價值不在於直接取代醫師診斷。它比較適合作為高敏感度的篩檢或分流輔助工具。因為外部 NPV 仍達 0.962，代表模型判為陰性時相對可靠；但 PPV 只有 0.294，代表陽性結果需要人工審查。未來若要臨床化，需要 local recalibration、threshold optimization、prospective validation 和 workflow impact analysis。

### 7.6 如果教授問：為什麼 ViT 沒有全面打爆 CNN？

可以回答：

> ViT 的優勢是 global attention，能看 patch 之間的長距離關係；但醫學影像資料量通常不像自然影像那麼大，而 CNN 對局部紋理有很強的 inductive bias，所以 CNN baseline 仍然很有競爭力。本研究中 ViT-B/16 的 internal AUC 是 0.980，EfficientNet-B0 是 0.979，兩者差異很小且不顯著，表示架構差異不是唯一重點，資料品質、外部驗證和校準同樣重要。

---

## 8. 專案裡重要檔案你該怎麼看？

| 檔案 | 用途 |
|---|---|
| `app_binary.py` | Streamlit demo app |
| `model_utils.py` | 建立模型、載入 checkpoint、影像前處理 |
| `train_all_baselines.py` | 訓練 5 種模型 x 3 seeds |
| `evaluate_binary.py` | 評估 internal / external performance |
| `audit_dataset.py` | 檢查資料集數量、病人分布、重複影像 |
| `explain_vit.py` | 產生 Grad-CAM 解釋圖 |
| `bias_analysis.py` | subgroup analysis 工具 |
| `docs/manuscript_draft.md` | 雙語論文草稿 |
| `docs/project_tutorial.md` | 你正在看的教學手冊 |
| `results/` | 所有評估結果、圖表、CSV |
| `saved_models/` | 訓練好的模型 checkpoint |

---

## 9. 你現在最該記住的 10 句話

1. AI 看 X 光，其實是在看像素數字矩陣。
2. Label 是答案，模型靠影像和答案學習。
3. Train 是練習，validation 是小考，test 是期末考。
4. External validation 是換學校考，最能看出模型能不能推廣。
5. Sensitivity 高代表不容易漏掉肺炎。
6. Specificity 高代表不容易把正常誤判成肺炎。
7. PPV 會受到盛行率影響，低盛行率時 PPV 很容易掉。
8. AUC 看的是模型區分兩類的能力，不等於某個固定門檻下的診斷準確率。
9. Calibration 看的是模型機率可不可信。
10. 這個專案是研究型篩檢輔助原型，不是臨床診斷醫療器材。

---

## 10. 最後用一句完整專業話總結

如果你只能背一段上台講，背這段：

> 本專案建立了一個胸腔 X 光肺炎二元分類研究原型，使用 Kermany 兒童資料集訓練並比較五種深度學習架構，主要模型 ViT-B/16 在內部測試達到 ROC AUC 0.980 與 sensitivity 0.997；但在 RSNA 外部驗證中 ROC AUC 降至 0.801、PPV 降至 0.294，顯示明顯 domain gap 與 prevalence shift。因此本模型目前適合被視為高敏感度篩檢輔助研究工具，而不是可直接臨床診斷的系統。

這段話聽起來很專業，而且每個重點都對：

- 有說模型任務。
- 有說內部成績。
- 有說外部成績。
- 有說為什麼下降。
- 有保守講臨床定位。

這就是醫學 AI 專案最重要的成熟態度：不是只秀高分，而是誠實說明模型在哪裡有效、在哪裡失準、未來要怎麼驗證。


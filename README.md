**📩 AI 詐騙簡訊偵測器**

<img width="1920" height="934" alt="image" src="https://github.com/user-attachments/assets/4305105a-2918-4206-a7cb-8690fdf553a9" />

使用 Multinomial Naive Bayes 模型訓練超過 100 萬筆簡訊資料，建立即時分類器，幫助使用者辨別「正常簡訊」與「詐騙簡訊」。結合 Streamlit 製作互動式網頁工具，支援即時預測、視覺化報表與信心分數顯示。

---

 **🚀 專案亮點**

- 📊 **模型訓練**：採用 TF-IDF + Multinomial Naive Bayes（MNB）建立分類模型  
- 💬 **即時預測**：輸入任意簡訊文字，立即回傳詐騙/正常預測與信心分數  
- 📈 **視覺化報告**：混淆矩陣、分類報告、準確率、折線圖一應俱全  
- 🎨 **介面美化**：整合背景圖片、動畫標題與自訂樣式  
- 🖥️ **執行方式多元**：支援本地運行，也可部署至 Hugging Face 或 Streamlit Cloud

---

**🧰 使用技術**

| 技術       | 說明                                     |
|------------|------------------------------------------|
| Python     | 程式語言                                 |
| Streamlit  | 打造互動式 Web App 的框架                 |
| Pandas     | 資料處理與載入                           |
| TfidfVectorizer | 簡訊文字轉向量                        |
| MultinomialNB | 分類模型（Multinomial Naive Bayes）   |
| Matplotlib / Seaborn | 圖形視覺化                      |

---

**📂 專案架構**
├── app.py # 主程式

├── set_bg.py # 背景圖片設定

├── 1000000.csv # 訓練用的簡訊資料集（含標籤）

├── 1.png # 背景圖片

├── requirements.txt # 套件清單

________________________________________

**▶️ 如何執行**

**一、確認你的環境（Python 已安裝）
打開你的電腦方式:**

1. VScode 執行程式碼也可以。
2.命令提示字元（CMD） 或 終端機（Terminal），輸入以下指令確認：
python --version

如果有顯示版本（例如 Python 3.10.8），代表你有安裝 Python。

________________________________________

** 二、建立一個資料夾**

例如你在桌面建立一個資料夾叫：

spam_detector_app

________________________________________

** 三、建立 app.py 檔案**

1.	打開 VSCode 或 記事本
2.	將剛才我提供的程式碼貼上
   
儲存為：app.py

放在你剛剛建立的 spam_detector_app 資料夾裡
________________________________________

 **四、安裝套件（只需一次）**
 
打開 CMD 或 Terminal，進入你的專案資料夾，輸入：

cd 路徑/spam_detector_app

範例（Windows）：

cd Desktop\spam_detector_app

然後安裝所需套件：

pip install streamlit pandas scikit-learn seaborn matplotlib
________________________________________

 **五、啟動 App**
在同一個資料夾中輸入：

streamlit run app.py

執行後，瀏覽器會自動打開你的應用程式

（預設網址是 [http://localhost:8501）

________________________________________

 **六、成功畫面長這樣**
🔵 標題：「AI 詐騙簡訊偵測器」

📩 有文字輸入框 → 貼入簡訊 → 按下「開始分析」

✅ 顯示結果：Spam / Ham（含信心分數）

________________________________________

**結果**

1.先選擇模型

 <img width="1920" height="882" alt="螢幕擷取畫面 2025-07-22 082608" src="https://github.com/user-attachments/assets/2929bbb0-1713-46bc-97ff-47c52b70e3bf" />


2.在按 “ 訓練模型 ” 

<img width="743" height="358" alt="螢幕擷取畫面 2025-07-22 082619" src="https://github.com/user-attachments/assets/0e0cca9e-858e-478e-9456-84e861ed2f24" />


3.輸入簡訊的內容，在按” 開始分析 ”，分析完，就會有預測結果的信心分數。

<img width="756" height="270" alt="螢幕擷取畫面 2025-07-22 082645" src="https://github.com/user-attachments/assets/0a6963c1-64e0-488b-8bfb-6d467db9342b" />

 
4.接下來，顯示完整評估報告也就是數據圖。

<img width="706" height="578" alt="螢幕擷取畫面 2025-07-22 082706" src="https://github.com/user-attachments/assets/f5cff303-05c3-42eb-a473-ff13b3a7846c" />

<img width="700" height="763" alt="螢幕擷取畫面 2025-07-22 082714" src="https://github.com/user-attachments/assets/2375cb69-093a-4174-8daa-fc89b280acff" />

<img width="700" height="599" alt="螢幕擷取畫面 2025-07-22 082725" src="https://github.com/user-attachments/assets/09a52db0-e4cb-468d-851d-a944dca05611" />

<img width="696" height="598" alt="螢幕擷取畫面 2025-07-22 082731" src="https://github.com/user-attachments/assets/b4b479f2-8d48-40e9-bb0f-94e46dfd8ce9" />

<img width="692" height="610" alt="螢幕擷取畫面 2025-07-22 082740" src="https://github.com/user-attachments/assets/8e6ccbc8-a160-42c1-bda9-1454db496d94" /> 
    

**「mAP顯示不出來，代表類別太少」**
 
 <img width="689" height="121" alt="螢幕擷取畫面 2025-07-22 082745" src="https://github.com/user-attachments/assets/c1f63f8c-9035-4577-9304-e6cab661f103" />



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re  # 引入 re 模組
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
from set_bg import set_bg_image
import joblib
import os



# -------------------- 頁面設定 --------------------
PAGE_TITLE = "AI 詐騙簡訊偵測器"
PAGE_ICON = "📩"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

set_bg_image("1.jpg")  # 替換成您圖片的路徑

st.markdown(f'<h1 class="glow">{PAGE_ICON} {PAGE_TITLE}</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='font-family:DFKai-SB; font-size:20px;'>
📩 請輸入簡訊內容，我們會用 AI 模型協助你辨識是否為詐騙訊息（Spam）。
</div>
""", unsafe_allow_html=True)

# -------------------- 資料載入 --------------------
@st.cache_data
def load_data():
    # 假設資料集為CSV格式
    df = pd.read_csv("1000000.csv")
    return df

df = load_data()
st.write(f"📊 資料集已載入：{len(df):,} 筆")

# -------------------- 文字預處理 --------------------
def preprocess_text(text):
    text = text.lower()  # 小寫化
    text = re.sub(r"[^\w\s]", "", text)  # 移除標點符號
    return text

df['processed_message'] = df['message'].apply(preprocess_text)

# -------------------- 目標與特徵 --------------------
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # scam=1, normal=0

X = df['processed_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------- 模型儲存與選擇 --------------------
MODEL_PATH = "spam_model.pkl"

def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        joblib.dump(model, f)

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
            # 檢查模型是否為正確的類型
            if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                return model
            else:
                st.warning("⚠️ 這不是一個有效的模型文件")
                return None
    return None


# 預設載入模型為 MNB
model = load_model()

# -------------------- 模型選擇 --------------------
model_option = st.selectbox("選擇模型", ["Naive Bayes (MNB)", "SVM"])

if model is None or st.button("訓練模型"):
    if model_option == "Naive Bayes (MNB)":
        model = MultinomialNB()
    elif model_option == "SVM":
        model = SVC(probability=True)
    
    model.fit(X_train_tfidf, y_train)
    save_model(model)
    st.success(f"模型訓練成功！使用 {model_option} 模型進行訓練。")

# -------------------- 預測函數 --------------------
def predict_message(msg):
    tfidf_msg = vectorizer.transform([msg])
    pred = model.predict(tfidf_msg)[0]
    prob = model.predict_proba(tfidf_msg)[0][pred]
    label_name = le.inverse_transform([pred])[0]
    return ("詐騙簡訊 (Scam)" if label_name == "scam" else "正常簡訊 (Normal)"), round(prob * 100, 2)

# -------------------- 使用者輸入區 --------------------
user_input = st.text_area("請輸入簡訊內容：", height=100)

if st.button("🔍 開始分析"):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入簡訊內容")
    else:
        result, confidence = predict_message(user_input)
        st.success(f"📌 預測結果：{result}（信心分數：{confidence}%）")

# -------------------- 評估報告 --------------------
with st.expander("🔍 顯示完整評估報告"):
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="🎯 測試集準確率", value=f"{acc * 100:.2f}%")

    report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
    st.subheader("📄 Classification Report")
    st.dataframe(report)

    # 混淆矩陣（數值）
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"實際:{cls}" for cls in le.classes_],
                         columns=[f"預測:{cls}" for cls in le.classes_])
    st.subheader("🔢 混淆矩陣（數值）")
    st.dataframe(cm_df)

    # 混淆矩陣（圖形）
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("預測標籤")
    ax.set_ylabel("實際標籤")
    ax.set_title("Confusion Matrix")
    st.subheader("🎨 混淆矩陣（視覺化）")
    st.pyplot(fig)

    # 類別分佈折線圖
    st.subheader("📈 簡訊類別分佈折線圖")
    label_counts = df['label'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.plot(label_counts.index, label_counts.values, marker='o', linestyle='-', color='darkorange')
    ax2.set_title("Scam / Normal 數量趨勢")
    ax2.set_xlabel("類別")
    ax2.set_ylabel("簡訊數量")
    ax2.grid(True)
    st.pyplot(fig2)

    # ROC 曲線
    st.subheader("📉 模型效能 ROC 曲線")
    y_scores = model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC 曲線 (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate (假陽性率)')
    ax3.set_ylabel('True Positive Rate (真陽性率)')
    ax3.set_title('ROC Curve')
    ax3.legend(loc="lower right")
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    # PR 曲線
    st.subheader("📈 Precision-Recall 曲線（PR Curve）")
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label=f'Average Precision = {avg_precision:.2f}', color='darkorange')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    ax_pr.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_pr)

     # === mAP（Mean Average Precision） ===
    st.subheader("📊 mAP（各類別 PR 曲線）")

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_score_all = model.predict_proba(X_test_tfidf)
    y_test_bin = label_binarize(y_test, classes=list(le.transform(le.classes_)))

    # 二元分類時需 reshape
    if y_test_bin.ndim == 1:
        y_test_bin = y_test_bin.reshape(-1, 1)

    if y_score_all.shape[1] == y_test_bin.shape[1]:  # 確保維度對應
        ap_dict = {}
        fig_map, ax_map = plt.subplots()
        for i, class_name in enumerate(le.classes_):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score_all[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_score_all[:, i])
            ap_dict[class_name] = ap
            ax_map.plot(recall, precision, lw=2, label=f'{class_name} (AP={ap:.2f})')

        mean_ap = sum(ap_dict.values()) / len(ap_dict)
        ax_map.set_title(f"Mean Average Precision (mAP) = {mean_ap:.2f}")
        ax_map.set_xlabel("Recall")
        ax_map.set_ylabel("Precision")
        ax_map.legend(loc="lower left")
        ax_map.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_map)
    else:
        st.warning("⚠️ 分類數量不符，無法產生 mAP 曲線")
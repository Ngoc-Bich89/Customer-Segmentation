# app_customer_segmentation_kmeans.py

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScalerModel, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeansModel, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import plotly.express as px
import base64
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from pdf2image import convert_from_path
#import java_bootstrap

# ---------------- Spark Session ----------------
@st.cache_resource
def init_spark():
    # Khởi động Java trên Cloud
    #java_bootstrap.ensure_java()
    
    # SparkSession
    spark = SparkSession.builder \
        .appName("CustomerSegmentation") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    return spark

spark = init_spark()

# ---------------- Banner ----------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #2E86C1;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #566573;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Customer Segmentation App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze customers with RFM & KMeans clustering</div>', unsafe_allow_html=True)
st.write("---")

# ---------------- Tabs ----------------
tabs = st.tabs(["📖 Introduction", "📂 Upload Data", "🔍 EDA", "📊 Clustering & Visualization", "🔮 Prediction", "📑 Final Report"])

# ---------------- 1. Introduction ----------------
with tabs[0]:
    st.header("Welcome")
    st.markdown("""
    Xin chào 👋, đây là **ứng dụng phân tích khách hàng** được xây dựng bằng Python + PySpark + Streamlit.

    Ứng dụng này giúp cửa hàng/doanh nghiệp:
    - Phân tích dữ liệu giao dịch và sản phẩm.
    - Tính toán **RFM (Recency - Frequency - Monetary)** cho từng khách hàng.
    - Thực hiện phân cụm khách hàng bằng **KMeans (PySpark)**.
    - Xác định các nhóm khách hàng **Loyal Customers", "Potential Customers", "Occasional Customers", "At-Risk Customers**.
    - Trực quan hóa dữ liệu: Histogram, Scatter Plot, Cluster Distribution.
    - Dự đoán **cluster cho từng khách hàng** hoặc nhập RFM để dự đoán.
    - Xuất báo cáo cuối cùng (Final Report) về hành vi khách hàng.
    """)

# ---------------- 2. Upload Data ----------------
with tabs[1]:
    st.header("Upload Your Data")
    file_product = st.file_uploader("Upload Product CSV (productId, productName, price, Category)", type=["csv"])
    file_trans = st.file_uploader("Upload Transactions CSV (Member_number, Date, productId, items)", type=["csv"])
    if file_product and file_trans:
        product_df = pd.read_csv(file_product)
        trans_df = pd.read_csv(file_trans)
        st.success("✅ Files uploaded successfully!")
        st.write("📦 Product Sample")
        st.dataframe(product_df.head())
        st.write("🛒 Transactions Sample")
        st.dataframe(trans_df.head())

# =========================
# Load trained pipeline model
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_models():
    scaler_model = StandardScalerModel.load(os.path.join(BASE_DIR, "output/scaler_model"))
    kmeans_model = KMeansModel.load(os.path.join(BASE_DIR, "output/kmeans_model"))
    return scaler_model, kmeans_model

scaler_model, kmeans_model = load_models()
print("BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR))
print("Output exists?", os.path.exists(os.path.join(BASE_DIR, "output")))
if os.path.exists(os.path.join(BASE_DIR, "output")):
    print("Output content:", os.listdir(os.path.join(BASE_DIR, "output")))

# ---------------- 3. EDA ----------------
with tabs[2]:
    st.header("Exploratory Data Analysis")
    if file_product and file_trans:
        trans_df['Date'] = pd.to_datetime(trans_df['Date'])
        df = trans_df.merge(product_df, on='productId', how='left')
        df["amount"] = df["price"] * df["items"]

        # RFM
        max_date = df['Date'].max().date()
        Recency = lambda x : (max_date - x.max().date()).days
        Frequency  = lambda x: x.count()
        Monetary = lambda x : round(sum(x), 2)
        df_RFM = df.groupby('Member_number').agg({'Date': Recency,'Member_number': Frequency,'amount': Monetary }).rename(columns={"Date": "Recency", "Member_number": "Frequency", "amount": "Monetary"}).reset_index()
        df_merge = df.merge(df_RFM, on='Member_number', how='left')

        st.subheader("📊 RFM Table")
        st.dataframe(df_merge.head())
        st.write(df_merge.describe())
        # -------------------- Key Metrics Cards --------------------
        st.subheader("🔑 RFM Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recency (days)", f"{df_RFM['Recency'].mean():.1f}", f"Min: {df_RFM['Recency'].min()} / Max: {df_RFM['Recency'].max()}")
        col2.metric("Frequency", f"{df_RFM['Frequency'].mean():.1f}", f"Min: {df_RFM['Frequency'].min()} / Max: {df_RFM['Frequency'].max()}")
        col3.metric("Monetary ($)", f"{df_RFM['Monetary'].mean():.2f}", f"Min: {df_RFM['Monetary'].min()} / Max: {df_RFM['Monetary'].max()}")
        # -------------------- Mini-summary --------------------
        st.markdown(f"""
        **📌 Summary:**  
        - Total customers: {df_RFM['Member_number'].nunique()}  
        - Avg Recency: {df_RFM['Recency'].mean():.1f} days  
        - Avg Frequency: {df_RFM['Frequency'].mean():.1f} times  
        - Avg Monetary: ${df_RFM['Monetary'].mean():.2f}
        """)

# ---------------- 4. Clustering & Visualization ----------------
with tabs[3]:
    st.header("KMeans Clustering & Visualization")
    if file_product and file_trans:
        rfm_spark = spark.createDataFrame(df_merge)

        # Features
        feature_cols = ["Recency", "Frequency", "Monetary"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
        rfm_spark = assembler.transform(rfm_spark)
        scaler = StandardScaler(inputCol="features_vec", outputCol="features", withStd=True, withMean=True)
        #scaler_model = scaler.fit(rfm_spark)
        rfm_spark = scaler_model.transform(rfm_spark)

        # KMeans dùng model đã load
        rfm_clustered = kmeans_model.transform(rfm_spark)
        # Pandas
        rfm_pdf = rfm_clustered.toPandas()

        # Label clusters by Monetary mean
        cluster_mean = rfm_pdf.groupby("cluster")["Monetary"].mean().sort_values(ascending=False)
        labels = ["Loyal Customers", "Potential Customers", "Occasional Customers", "At-Risk Customers"]
        cluster_label_map = {cluster: labels[i] for i, cluster in enumerate(cluster_mean.index)}
        rfm_pdf["Cluster_Label"] = rfm_pdf["cluster"].map(cluster_label_map)

        #rfm_pdf["Cluster_Label"] = rfm_pdf["cluster"].map(cluster_label_map)
        # Merge Category
        #trans_tmp = trans_df[["Member_number", "productId"]].merge(product_df[["productId", "Category"]], on="productId", how="left")
        #rfm_pdf = rfm_pdf.merge(trans_tmp[["Member_number", "Category"]], on="Member_number", how="left")

        # ---------------- Filter by Category ----------------
        st.subheader("📂 Filter by Category")
        category_list = ["All"] + sorted(product_df["Category"].dropna().unique().tolist())
        selected_category = st.selectbox("Select Category", category_list)

        if selected_category != "All":
            df_filtered = rfm_pdf[rfm_pdf["Category"] == selected_category]
        else:
            df_filtered = rfm_pdf.copy()

        # Histogram
        st.subheader("📊 Histogram of RFM")
        fig, axs = plt.subplots(3, 1, figsize=(7, 9))

        sns.histplot(df_filtered['Recency'], bins=20, kde=True, ax=axs[0], color="skyblue")
        axs[0].set_title("Distribution of Recency", fontsize=12, fontweight="bold")

        sns.histplot(df_filtered['Frequency'], bins=20, kde=True, ax=axs[1], color="lightgreen")
        axs[1].set_title("Distribution of Frequency", fontsize=12, fontweight="bold")

        sns.histplot(df_filtered['Monetary'], bins=20, kde=True, ax=axs[2], color="salmon")
        axs[2].set_title("Distribution of Monetary", fontsize=12, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)

        # Pie chart (Cluster Distribution)
        st.subheader("🥧 Customer Segments")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        df_filtered["Cluster_Label"].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", df_filtered["Cluster_Label"].nunique()),
            ax=ax2
        )
        ax2.set_ylabel("")
        ax2.set_title("Customer Segments", fontsize=12, fontweight="bold")
        st.pyplot(fig2)

        # Scatter
        st.subheader("📈 Scatter Plot Recency vs Monetary")
        fig3 = px.scatter(df_filtered,x="Recency", y="Frequency",color="Cluster_Label",hover_data=["cluster", "Category"], color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig3, use_container_width=True)

        # Scatter plot categorical color
        rfm_summary = (df_filtered.groupby(["cluster", "Cluster_Label"]).agg({"Recency": "mean","Monetary": "mean","Frequency": "mean"}).reset_index().rename(columns={"Recency": "Avg_Recency","Monetary": "Avg_Monetary","Frequency": "Avg_Frequency"}))       
        fig4 = px.scatter(rfm_summary,x="Avg_Recency",y="Avg_Monetary",size="Avg_Frequency",color="Cluster_Label",hover_data=["cluster"],size_max=60,color_discrete_sequence=px.colors.qualitative.Set2)

        st.subheader("📊 Scatter Plot RFM Clusters")
        st.plotly_chart(fig4, use_container_width=True)


# ---------------- 5. Prediction ----------------
with tabs[4]:
    st.header("Prediction")
    if file_product and file_trans:
        # -------------------- Giải thích các Cluster --------------------
        st.subheader("💡 Giải thích 4 Cluster")
        st.markdown("""
        - **Loyal Customers (Khách hàng trung thành)** 🏆: Mua thường xuyên, chi tiêu cao, gần đây vẫn hoạt động.  
          → Đây là nhóm khách hàng quan trọng, nên giữ chân và chăm sóc đặc biệt.
          
        - **Potential Customers (Khách hàng tiềm năng)** 🌱: Mua đều đặn nhưng chi tiêu trung bình, gần đây vẫn hoạt động.  
          → Có khả năng trở thành khách trung thành nếu tăng cường marketing phù hợp.
          
        - **Occasional Customers (Khách hàng thỉnh thoảng)** 🛍️: Mua ít, chi tiêu trung bình, không đều.  
          → Cần khuyến mãi hoặc nhắc nhở để tăng tần suất mua.
          
        - **At-Risk Customers (Khách hàng rủi ro / gần mất)** ⚠️: Mua ít, chi tiêu thấp, lâu không hoạt động.  
          → Cần chiến lược giữ chân hoặc khuyến mãi đặc biệt để tránh mất khách.
        """)
        # ------------------- 1. Predict from existing Member_number -------------------
        with st.expander("🔽 Predict from existing Member_number", expanded=True):
            st.subheader("Select Member_number")
            member_list = rfm_pdf["Member_number"].unique().tolist()
            member_input = st.selectbox("Choose Member_number", member_list)

            if st.button("Predict Cluster for Member_number"):
                if member_input:
                    customer = rfm_pdf[rfm_pdf["Member_number"] == member_input]
                    cluster = int(customer["cluster"].values[0])
                    label = customer["Cluster_Label"].values[0]
                    st.success(f"Member **{member_input}** → Cluster {cluster} ({label})")
                else:
                    st.warning("Please select a Member_number")

        # ------------------- 2. Predict from RFM input -------------------
        with st.expander("✍️ Predict from RFM Input", expanded=True):
            st.subheader("Input RFM values")
            recency = st.number_input("Recency", min_value=0, value=30)
            frequency = st.number_input("Frequency", min_value=0, value=5)
            monetary = st.number_input("Monetary", min_value=0, value=100)

            if st.button("Predict Cluster from RFM input"):
                from pyspark.sql import Row
                new_df = spark.createDataFrame([Row(Recency=float(recency),
                                                   Frequency=float(frequency),
                                                   Monetary=float(monetary))])
                new_df = assembler.transform(new_df)
                new_df = scaler_model.transform(new_df)
                cluster_pred = kmeans_model.transform(new_df).select("cluster").collect()[0][0]
                label_pred = cluster_label_map[cluster_pred]
                st.success(f"Predicted Cluster: {cluster_pred} ({label_pred})")

# ---------------- 6. Final Report ----------------
with tabs[5]:
    st.header("Final Report")
    report_file = "final_report.pptx"  # em để file pptx trong repo
    if os.path.exists(report_file):
        # Convert PPTX -> PDF bằng LibreOffice CLI
        os.system(f"libreoffice --headless --convert-to pdf {report_file} --outdir .")
        pdf_file = report_file.replace(".pptx", ".pdf")

        if os.path.exists(pdf_file):
            slides = convert_from_path(pdf_file)
            st.subheader("📑 Slide Preview")
            for i, slide in enumerate(slides, 1):
                st.image(slide, caption=f"Slide {i}", use_container_width=True)
        else:
            st.error("❌ Không convert được PPTX sang PDF.")
    else:
        st.warning("⚠️ Report file not found. Please add `final_report.pptx` to the repo.")
# ---------------- Footer ----------------
image_path = Path("output/ava.png")
with open(image_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()
st.write("---")
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; gap:20px; padding:20px; background-color:#f9f9f9; border-radius:10px;">   
    <!-- Ảnh người làm -->
    <img src="data:image/png;base64,{img_base64}" 
         width="120" style="border-radius:50%; border:3px solid #4CAF50;"/>
    <!-- Thông tin người làm và GVHD -->
    <div style="display:flex; flex-direction:column; justify-content:center; gap:5px;">
        <div style="display:flex; gap:40px;">
            <p style="margin:0; font-size:16px;"><b>Nguyễn Lê Ngọc Bích</b> - <a href="mailto:ngocbich.892k1@gmail.com">ngocbich.892k1@gmail.com</a></p>
            <p style="margin:0; font-size:16px;"><b>GVHD:</b> Khuất Thùy Phương</p>
        </div>
        <!-- Icon minh họa -->
        <img src="https://img.icons8.com/color/96/000000/customer-insight.png" width="60"/>
    </div>
</div>
""", unsafe_allow_html=True)

# 📊 Customer Segmentation using RFM & K-Means

## 🎯 Objective
Mục tiêu của dự án là phân khúc khách hàng (Customer Segmentation) dựa trên dữ liệu giao dịch.  
Điều này giúp doanh nghiệp:

- Hiểu rõ hành vi khách hàng  
- Nhận diện nhóm khách hàng trung thành, tiềm năng, có nguy cơ rời bỏ  
- Đưa ra chiến lược marketing cá nhân hóa nhằm tăng doanh thu và giữ chân khách hàng  

---

## 🔎 Quy trình thực hiện
1. **Data Preparation**
   - Thu thập dữ liệu giao dịch  
   - Làm sạch dữ liệu (loại bỏ null, trùng lặp, xử lý ngày tháng & số liệu)  

2. **RFM Analysis**
   - **Recency (R):** Số ngày kể từ lần mua gần nhất  
   - **Frequency (F):** Tổng số lần mua  
   - **Monetary (M):** Tổng giá trị chi tiêu  
   - Chuẩn hóa dữ liệu RFM  

3. **K-Means Clustering**
   - Xác định số cụm tối ưu (Elbow Method, Silhouette Score)  
   - Chạy mô hình K-Means  
   - Gán nhãn cluster cho từng khách hàng  

4. **Cluster Profiling & Insights**
   - Phân tích đặc điểm từng cluster theo RFM  
   - Xác định Top 3 Category cho mỗi nhóm (dù trùng nhau nhưng tỷ trọng khác nhau)  

---

## 📊 Kết quả phân cụm

### Theo RFM Analysis (3 nhóm)
- **High (~50%)**  
  Mua thường xuyên, chi tiêu cao  
  → Nhóm cốt lõi  

- **Medium (~40%)**  
  Hành vi mua vừa phải, có tiềm năng tăng trưởng  
  → Cần khuyến khích mua thêm  

- **Low (~10%)**  
  Ít quay lại, chi tiêu thấp  
  → Cần chiến dịch tái kích hoạt  

### Theo RFM + K-Means (4 nhóm)
- **Cluster 0 (~50%)**  
  Mua thường xuyên, chi tiêu cao → nhóm cốt lõi  

- **Cluster 1 (~8.83%)**  
  Ít quay lại, chi tiêu thấp → cần tái kích hoạt  

- **Cluster 2 (~29.19%)**  
  Mua vừa phải, có tiềm năng → cần khuyến khích mua thêm  

- **Cluster 3 (~19.52%)**  
  Ít quay lại, chi tiêu thấp (giảm 1/2 so với Cluster 2)  
  → Nếu không can thiệp sẽ rơi vào Cluster 1, khó quay lại  

### 🔍 Phân tích Category
- Top 3 Category ở tất cả cluster đều giống nhau  
- Khác nhau ở **tỷ trọng chi tiêu**  
- Insight: khách hàng có cùng sở thích sản phẩm, khác nhau ở **cường độ mua và chi tiêu**  

---

## 📌 Next Steps / Business Implications

1. **Hành động cho từng nhóm**
   - **Trung thành:** VIP program, referral, tri ân → Giữ chân lâu dài, tăng CLV  
   - **Bình thường:** Voucher, combo, cross-sell/upsell → Đẩy khách lên nhóm trung thành  
   - **Nguy cơ rời bỏ:** Re-activation campaign (email, SMS, ưu đãi mạnh) → Giảm churn rate  

2. **Theo dõi KPI**
   - Retention rate  
   - Customer Lifetime Value (CLV)  
   - Conversion rate từ các campaign  

3. **Mở rộng mô hình**
   - Thêm dữ liệu demographic, kênh mua hàng  
   - Thử các mô hình phân cụm khác (Hierarchical, DBSCAN)  
   - Tạo dashboard trực quan (Tableau, PowerBI, Streamlit)  

---

## 🛠️ Công nghệ sử dụng
- **Python:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Jupyter Notebook**  
- **Visualization:** matplotlib, seaborn  

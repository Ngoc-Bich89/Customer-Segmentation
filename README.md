📊 Customer Segmentation using RFM & K-Means
🎯 Objective
Mục tiêu của dự án là phân khúc khách hàng (Customer Segmentation) dựa trên dữ liệu giao dịch.
Điều này giúp doanh nghiệp:
•	Hiểu rõ hành vi khách hàng
•	Nhận diện nhóm khách hàng trung thành, tiềm năng, có nguy cơ rời bỏ
•	Đưa ra chiến lược marketing cá nhân hóa nhằm tăng doanh thu và giữ chân khách hàng
________________________________________
🔎 Quy trình thực hiện
1.	Data Preparation
o	Thu thập dữ liệu giao dịch
o	Làm sạch dữ liệu (loại bỏ null, trùng lặp, xử lý ngày tháng & số liệu)
2.	RFM Analysis
o	Recency (R): Số ngày kể từ lần mua gần nhất
o	Frequency (F): Tổng số lần mua
o	Monetary (M): Tổng giá trị chi tiêu
o	Chuẩn hóa dữ liệu RFM
3.	K-Means Clustering
o	Xác định số cụm tối ưu (Elbow Method, Silhouette Score)
o	Chạy mô hình K-Means
o	Gán nhãn cluster cho từng khách hàng
4.	Cluster Profiling & Insights
o	Phân tích đặc điểm từng cluster theo RFM
o	Xác định Top 3 Category cho mỗi nhóm (dù trùng nhau nhưng tỷ trọng khác nhau)
________________________________________
📊 Kết quả phân cụm
Dữ liệu khách hàng được chia thành 3 nhóm chính theo RFM analysis
•	Nhóm High (~50%)
o	Mua thường xuyên, chi tiêu cao
o	Chiếm tỉ lệ khách hàng lớn nhất → nhóm cốt lõi
•	Nhóm Medium (~40%)
o	Hành vi mua vừa phải, có tiềm năng tăng trưởng
o	Cần khuyến khích mua thêm
•	Nhóm Low (~10%)
o	Ít quay lại, chi tiêu thấp
o	Cần chiến dịch tái kích hoạt
Dữ liệu khách hàng được chia thành 4 nhóm chính theo RFM analysis + Kmeans
•	Nhóm Cluster 0 (~50%)
o	Mua thường xuyên, chi tiêu cao
o	Chiếm tỉ lệ khách hàng lớn nhất → nhóm cốt lõi
•	Nhóm Cluster 1 (~8.83%)
o	Ít quay lại, chi tiêu thấp
o	Cần chiến dịch tái kích hoạt
•	Nhóm Cluster 2 (~29.19%)
o	Hành vi mua vừa phải, có tiềm năng tăng trưởng
o	Cần khuyến khích mua thêm
•	Nhóm Cluster 3 (~19.52%)
o	Ít quay lại, chi tiêu thấp, giảm 1 nửa so với Cluster 2 
o	Cần kéo nhóm này quay lại Cluster 2 nếu để lâu dài không có chiến dịch thì sẽ rơi vào Cluster 1 gần như không để nhóm khách hàng này quay lại được nữa.
🔍 Phân tích Category:
•	Top 3 Category ở tất cả cluster đều giống nhau, nhưng tỷ trọng tiêu dùng khác nhau
•	Insight: khách hàng có cùng sở thích sản phẩm, khác nhau ở cường độ mua và chi tiêu
________________________________________
📌 Next Steps / Business Implications
1. Hành động cho từng nhóm và Cluster +
•	Trung thành 
o	Chăm sóc đặc biệt: VIP program, referral, tri ân khách hàng
o	Mục tiêu: Giữ chân lâu dài, tăng CLV
•	Bình thường 
o	Tăng động lực mua: voucher, combo, cross-sell/upsell
o	Mục tiêu: Đẩy khách lên nhóm trung thành
•	Nguy cơ rời bỏ 
o	Re-activation campaign: email, SMS, ưu đãi mạnh
o	Mục tiêu: Kéo khách quay lại, giảm churn rate
2. Theo dõi KPI
•	Retention rate
•	Customer Lifetime Value (CLV)
•	Conversion rate từ các campaign
3. Mở rộng mô hình
•	Thêm dữ liệu demographic, kênh mua hàng
•	Thử các mô hình phân cụm khác (Hierarchical, DBSCAN)
•	Tạo dashboard trực quan (Tableau, PowerBI, Streamlit)
________________________________________
🛠️ Công nghệ sử dụng
•	Python: pandas, numpy, matplotlib, seaborn, scikit-learn
•	Jupyter Notebook
•	Visualization: matplotlib, seaborn


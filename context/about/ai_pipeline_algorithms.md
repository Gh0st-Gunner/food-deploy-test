# 3. Quy Trình Xử Lý AI & Thuật Toán Ước Lượng Khẩu Phần 3D (AI Pipeline & 3D Portion Estimation)

Tài liệu này giải thích chi tiết cơ sở khoa học, công nghệ học sâu và các thuật toán hình học đằng sau bộ quét món ăn 3D của Munchin'.

---

## 1. Các Mô Hình Học Sâu Trong Pipeline AI

Hệ thống kết hợp ba mô hình học sâu chuyên biệt trong lĩnh vực thị giác máy tính:

### A. Grounding DINO (Zero-Shot Object Detection)
*   **Mục tiêu:** Phát hiện vị trí (hộp giới hạn - Bounding Box) của các thành phần nguyên liệu trong đĩa ăn dựa trên các gợi ý bằng văn bản (text prompts).
*   **Cơ chế hoạt động:** Thay vì giới hạn trong các lớp nhãn cứng nhắc như các mô hình YOLO thông thường, Grounding DINO liên kết trực tiếp giữa văn bản mô tả nguyên liệu (ví dụ: `"thịt bò, bún, giá đỗ, hành lá"`) với các đặc trưng hình ảnh của bức ảnh. Kết quả trả về là một danh sách tọa độ hộp giới hạn $[x_{min}, y_{min}, x_{max}, y_{max}]$ kèm theo điểm số tin cậy (confidence score) cho từng nhãn nguyên liệu.

### B. SAM 2 (Segment Anything Model 2)
*   **Mục tiêu:** Cắt phân vùng đối tượng (Instance Segmentation) ở cấp độ pixel để xác định diện tích bề mặt chính xác của từng nguyên liệu.
*   **Cơ chế hoạt động:** SAM 2 là mô hình phân đoạn nền tảng (Foundation Model). Hệ thống sử dụng kết quả hộp giới hạn từ Grounding DINO làm gợi ý đầu vào (bounding box prompts) cho SAM 2. SAM 2 sẽ sinh ra các mặt nạ nhị phân (binary mask) bao quanh chính xác biên dạng của từng loại thức ăn.

### C. Depth Anything V2 (Monocular Depth Estimation)
*   **Mục tiêu:** Dựng bản đồ chiều sâu (Depth Map) từ một bức ảnh 2D duy nhất để thu thập thông tin chiều cao 3D của khối thức ăn.
*   **Cơ chế hoạt động:** Depth Anything V2 phân tích các chi tiết đổ bóng, phối cảnh và tiêu cự trong ảnh để dự đoán khoảng cách tương đối của từng pixel so với camera. Bản đồ chiều sâu trả về là một ma trận 2D chứa các giá trị chuẩn hóa $[0.0, 1.0]$, trong đó giá trị lớn hơn tương ứng với các đối tượng nằm gần camera hơn (nhô cao hơn).

---

## 2. Thuật Toán Ước Lượng Khẩu Phần 3D (Portion Sizing Algorithm)

Quy trình tính toán thể tích và khối lượng món ăn được thực hiện qua các bước hình học sau:

### Bước 1: Hiệu chỉnh góc chụp bằng Ellipse Fitting
Các ảnh chụp đĩa thức ăn thường được chụp ở góc nghiêng từ $30^\circ$ đến $60^\circ$ thay vì chụp thẳng từ trên xuống (top-down). Điều này làm đĩa tròn hiển thị dưới dạng hình bầu dục (ellipse) trên ảnh 2D, gây sai lệch nghiêm trọng về diện tích thực tế.
1.  Hệ thống lấy mặt nạ đĩa ăn (`dish_mask`), tìm đường viền lớn nhất bằng OpenCV (`cv2.findContours`).
2.  Khớp hình ellipse bao quanh đĩa bằng thuật toán bình phương tối thiểu (`cv2.fitEllipse`). Thuật toán trả về tọa độ tâm, chiều dài trục lớn ($Major\_Axis$) và trục nhỏ ($Minor\_Axis$).
3.  Tính toán tỷ lệ nghiêng của camera (`tilt_ratio`):
    $$tilt\_ratio = \frac{Major\_Axis}{Minor\_Axis}$$
    Tỷ lệ này được giới hạn từ $1.0$ (góc chụp thẳng đứng $90^\circ$) đến tối đa $2.5$ (góc nghiêng khoảng $66^\circ$) để tránh sai lệch cực đoan.

### Bước 2: Tính toán tỷ lệ chuyển đổi vật lý (Physical Scale)
Hệ thống sử dụng đường kính đĩa ăn tiêu chuẩn làm vật tham chiếu (mặc định $25 \text{ cm}$):
$$Scale_{cm/px} = \frac{PlateDiameter_{real} (25\text{ cm})}{Major\_Axis_{px}}$$

### Bước 3: Tính diện tích thức ăn thực tế (Actual Area)
Diện tích bề mặt thực tế của thức ăn ($Area_{cm^2}$) sau khi bù trừ góc nghiêng camera được tính theo công thức:
$$Area_{cm^2} = FoodPixels \times (Scale_{cm/px})^2 \times tilt\_ratio$$
Trong đó, $FoodPixels$ là tổng số pixel được phân loại là thức ăn trong mặt nạ.

### Bước 4: Xác định chiều cao thức ăn bằng Depth Map
Để biết chiều cao thực tế của khối thức ăn, ta cần xác định mặt phẳng đáy (mặt đĩa) làm mốc tham chiếu không gian:
1.  **Tìm vùng biên đĩa:** Thực hiện phép toán xói mòn hình thái học (`cv2.erode` với nhân kích thước $5 \times 5$) trên mặt nạ đĩa ăn, lấy hiệu của mặt nạ gốc và mặt nạ xói mòn để thu được đường biên dạng đai (`boundary_mask`).
2.  **Xác định độ sâu mốc đĩa ($base\_depth$):** Tính toán giá trị trung vị (median) của bản đồ chiều sâu tại các điểm thuộc đường biên này. Vùng biên tiếp xúc trực tiếp với mặt phẳng đĩa, do đó giá trị này đại diện cho độ cao của đáy đĩa.
3.  **Tính chiều cao tương đối:** Chiều cao tương đối tại mỗi pixel thức ăn $j$ được tính bằng:
    $$Height_{rel, j} = \max(Depth_j - base\_depth, 0.0)$$
4.  **Chuyển đổi sang cm thực tế:** Chiều cao thực tế của pixel được tính bằng cách nhân chiều cao tương đối với hệ số chiều sâu tỉ lệ với kích thước đĩa:
    $$Height_{cm, j} = Height_{rel, j} \times (PlateDiameter_{real} \times 0.3)$$
    Độ cao trung bình của món ăn ($avg\_height_{cm}$) được giới hạn trong khoảng hợp lý từ $0.5 \text{ cm}$ (đồ ăn mỏng như trứng rán) đến $8.0 \text{ cm}$ (khối cơm/tô mì lớn).

### Bước 5: Tính Thể tích & Khối lượng (Volume & Weight)
1.  **Thể tích ($Volume_{ml}$):** Được tính bằng tích phân diện tích nhân chiều cao trung bình (quy đổi $1 \text{ cm}^3 = 1 \text{ ml}$):
    $$Volume_{ml} = Area_{cm^2} \times avg\_height_{cm}$$
2.  **Khối lượng ($Weight_{g}$):** Quy đổi thể tích sang khối lượng dựa trên mật độ khối lượng điển hình ($density_{g/ml}$) của từng nhóm món ăn (lấy từ dữ liệu mật độ dinh dưỡng định nghĩa sẵn):
    $$Weight_{g} = Volume_{ml} \times density_{g/ml}$$
    Nếu món ăn không khớp với bất kỳ mật độ khối lượng đặc thù nào, hệ thống mặc định sử dụng mật độ của nước/súp ($1.0 \text{ g/ml}$). Khối lượng sau cùng được giới hạn trong khoảng an toàn $15\% - 250\%$ của một khẩu phần ăn tiêu chuẩn để tránh các sai số đo đạc cực đoan.

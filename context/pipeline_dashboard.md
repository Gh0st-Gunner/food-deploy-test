# Pipeline Bảng Điều Khiển Dinh Dưỡng & Nhật Ký Bữa Ăn (Dashboard & Meal Diary Pipeline)

Tài liệu này đặc tả cơ chế tính toán calo mục tiêu (BMR & TDEE), quy trình ghi nhận nhật ký bữa ăn hàng ngày, và luồng đồng bộ dữ liệu hiển thị biểu đồ báo cáo tiến trình sức khỏe trên giao diện Dashboard.

---

## 1. Các Tính Năng & Trường Hợp Sử Dụng (Use Cases)

Hệ thống hỗ trợ người dùng theo dõi sức khỏe và quản lý khẩu phần ăn hàng ngày thông qua các ca sử dụng sau:

```plantuml
@startuml
left to right direction
skinparam packageStyle rectangle

actor User as "Người dùng"

rectangle "Hệ thống Dashboard & Nhật ký" {
    usecase "Khảo sát Onboarding ban đầu" as UC_onboarding
    usecase "Tính toán chỉ số mục tiêu (Calo & Macros)" as UC_calc_targets
    usecase "Thêm món ăn thủ công vào Nhật ký" as UC_add_manual
    usecase "Xóa món ăn khỏi Nhật ký" as UC_delete_meal
    usecase "Xem tiến trình nạp Calo hàng ngày" as UC_view_progress
    usecase "Xem biểu đồ báo cáo lịch sử sức khỏe" as UC_view_charts
    usecase "Lọc biểu đồ theo khoảng thời gian (1, 3, 7, 30 ngày)" as UC_filter_chart
}

User --> UC_onboarding
User --> UC_add_manual
User --> UC_delete_meal
User --> UC_view_progress
User --> UC_view_charts

UC_onboarding ..> UC_calc_targets : <<include>>
UC_view_charts ..> UC_filter_chart : <<include>>
@enduml
```

---

## 2. Quy Trình Hoạt Động (Flow of Operation)

### A. Quy trình Onboarding & Tính toán Calo/Macros mục tiêu
Khi đăng ký tài khoản lần đầu, hệ thống thực hiện thu thập dữ liệu chỉ số cơ thể của người dùng để tính toán năng lượng tiêu thụ tự nhiên và thiết lập mục tiêu calo tối ưu.

```plantuml
@startuml
start
:Thu thập thông tin cá nhân: Giới tính, Tuổi, Chiều cao (cm), Cân nặng (kg/lbs), Tần suất hoạt động, Mục tiêu sức khỏe;

if (Đơn vị cân nặng là lbs?) then (yes)
  :Quy đổi cân nặng sang kg (Weight_kg = Weight_lbs * 0.453592);
else (no)
  :Weight_kg = Cân nặng nhập vào;
endif

partition "Tính chỉ số BMR (Harris-Benedict)" {
  if (Giới tính là Nam?) then (yes)
    :BMR = 88.362 + (13.397 * Weight_kg) + (4.799 * Chiều cao) - (5.677 * Tuổi);
  else (no)
    :BMR = 447.593 + (9.247 * Weight_kg) + (3.098 * Chiều cao) - (4.330 * Tuổi);
  endif
}

partition "Tính chỉ số TDEE" {
  :TDEE = BMR * Hệ số hoạt động (Activity Factor);
  note right
    - Sedentary (Ít hoạt động): 1.2
    - Lightly active (Nhẹ): 1.375
    - Moderately active (Vừa): 1.55
    - Very active (Nhiều): 1.725
  end note
}

partition "Tính Calo Mục Tiêu" {
  if (Mục tiêu sức khỏe?) then (Giảm cân)
    :Target_Calories = TDEE - 500;
  elseif (Tăng cân)
    :Target_Calories = TDEE + 400;
  else (Duy trì)
    :Target_Calories = TDEE;
  endif
}

partition "Phân bổ Macros mục tiêu (Tỷ lệ năng lượng)" {
  :Target_Carbs (45% Calo) = (Target_Calories * 0.45) / 4 (gram);
  :Target_Protein (25% Calo) = (Target_Calories * 0.25) / 4 (gram);
  :Target_Fat (30% Calo) = (Target_Calories * 0.30) / 9 (gram);
}

:Lưu các mục tiêu calo và macros vào dữ liệu người dùng (DB / LocalStorage);
stop
@enduml
```

### B. Luồng Nghiệp vụ Ghi Nhật ký Bữa ăn & Cập nhật Dashboard
Nhật ký bữa ăn được chia làm các nhóm chính (Bữa sáng, Bữa trưa, Bữa tối, Ăn vặt). Mỗi lần người dùng thêm mới (hoặc xóa) một món ăn, Dashboard và biểu đồ tiến trình sẽ lập tức tính toán lại giá trị cộng dồn.

```plantuml
@startuml
autonumber
actor User as "Người dùng"
participant UI as "Giao diện SPA (Dashboard)"
participant Storage as "Trình lưu trữ (DB/LocalStorage)"

User -> UI: Chọn Thêm món ăn (Thủ công / Kết quả quét AI)
UI -> Storage: Ghi nhận bản ghi nhật ký bữa ăn mới
Storage -->> UI: Xác nhận đã lưu thành công

group Cập nhật Dashboard (Thời gian thực)
    UI -> UI: Lấy danh sách các món ăn đã ăn trong ngày
    UI -> UI: Cộng dồn Calo nạp vào = Sum(Calo_mon_an)
    UI -> UI: Cộng dồn Macros = Sum(Carbs), Sum(Protein), Sum(Fat)
    UI -> UI: Vẽ biểu đồ Arc tiến trình SVG: Calo nạp vào / Calo mục tiêu
    UI -> UI: Cập nhật các thanh tiến trình chất dinh dưỡng đa lượng
end

User -> UI: Chọn Xóa món ăn khỏi nhật ký
UI -> Storage: Xóa bản ghi món ăn đó
Storage -->> UI: Xác nhận đã xóa thành công
UI -> UI: Thực hiện recalculate cộng dồn và cập nhật lại giao diện Dashboard
@enduml
```

---

## 3. Cơ Chế Hoạt Động Của Biểu Đồ Lịch Sử Tuần / Tháng

Biểu đồ báo cáo lịch sử (`progress-line-chart`) được thiết kế linh hoạt với khả năng hiển thị đa thông số và đa khoảng thời gian:

### A. Các Chỉ Số Sức Khỏe Hỗ Trợ (Metrics)
Tùy thuộc vào chỉ số được người dùng nhấp chọn trên Dashboard, biểu đồ sẽ cấu hình động:
- **Calories**: Màu chủ đề Coral (`#F05C3B`), Đơn vị `kcal`.
- **Carbs / Protein / Fat**: Màu chủ đề Xanh lá/Xanh nhạt/Cam, Đơn vị `g`.
- **Weight**: Màu chủ đề Xanh dương (`#007AFF`), Đơn vị `kg` hoặc `lbs`.
- **Active Burn (Exercise)**: Màu chủ đề Xanh lá tươi (`#30D158`), Đơn vị `kcal`.

### B. Bộ Lọc Khoảng Thời Gian (Duration Filter)
Giao diện cung cấp hộp thoại chọn khoảng thời gian lọc:
- **1 ngày / 3 ngày / 7 ngày / 30 ngày**:
  - Khi người dùng thay đổi lựa chọn thời gian, biểu đồ sẽ gửi yêu cầu truy xuất dữ liệu lịch sử tương ứng.
  - Hệ thống tính toán giá trị trung bình (`Avg`) dựa trên tổng số ngày trong khoảng lọc và hiển thị lên tiêu đề thẻ.
  - Trục hoành (X-Axis) tự động co giãn và điều chỉnh nhãn hiển thị phù hợp (ví dụ: hiển thị nhãn cách quãng 6 ngày một lần đối với bộ lọc 30 ngày để chống tràn văn bản trên màn hình di động).

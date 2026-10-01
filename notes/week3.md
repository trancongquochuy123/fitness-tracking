# Handnote — Part 3: Data Visualization

> **Mục tiêu:** trực quan hóa dữ liệu accelerometer + gyroscope để hiểu pattern của từng bài tập, từng participant và từng loại set, từ đó chuẩn bị cho **outlier detection, feature engineering và modeling**. 

---

## 1. Fix bug từ Part 2

### Vấn đề

Trong bước rename columns ở Part 2, thứ tự các column bị nhầm giữa:

* `participant`
* `label`
* `category`

Nếu sai thứ tự → dữ liệu bị gán nhãn sai.

### Kiểm tra

Sau khi chạy lại script tạo dataset:

```text
participant → participant ID
label       → exercise
category    → heavy / medium
```

Sau đó export lại:

```text
01_data_processed.pkl
```

File này được sử dụng làm input cho bước visualization. 

---

# 2. Data Visualization Workflow

Workflow tổng quát:

```text
Define objective
      ↓
Collect data
      ↓
Clean / preprocess
      ↓
Explore data
      ↓
Create figures
      ↓
Style figures
      ↓
Export figures
```

Trong Part 3 tập trung chủ yếu vào:

```text
Create figures
      ↓
Style figures
      ↓
Export figures
```

Quan trọng nhất: **trước khi plot phải biết mình muốn figure trả lời câu hỏi gì.**

Ví dụ:

* Muốn biết set kéo dài bao lâu → giữ timestamp.
* Muốn biết có bao nhiêu samples → reset index và dùng sample number. 

---

# 3. Download & chuẩn bị Python file

File sử dụng:

```text
src/
└── visualization/
    └── visualize.py
```

File gồm:

* imports
* các phần visualization theo thứ tự
* code để tạo và export figures. 

---

# 4. Loading Data

Dataset đã được lưu dưới dạng **pickle**.

```python
import pandas as pd

df = pd.read_pickle(
    "../../data/interim/01_data_processed.pkl"
)
```

Kiểm tra:

```python
df.head()
```

Cần đảm bảo:

| Column         | Ý nghĩa                   |
| -------------- | ------------------------- |
| `participant`  | người thực hiện           |
| `label`        | exercise                  |
| `category`     | `heavy` / `medium`        |
| sensor columns | accelerometer / gyroscope |
| `set`          | từng set                  |



---

# 5. Plot một column

## 5.1. Tại sao cần chọn từng `set`?

Nếu plot toàn bộ dataset cùng lúc:

```text
Participant A
Participant B
Participant C
...
Exercise 1
Exercise 2
...
```

→ figure trở nên rất rối.

Vì vậy, trước tiên isolate một set.

```python
set_df = df[
    df["set"] == 1
]
```

Ý tưởng:

```text
Original DataFrame
        ↓
filter set == 1
        ↓
Subset
        ↓
Plot
```



---

## 5.2. Plot một sensor column

Ví dụ `acc_y`:

```python
plt.plot(set_df["acc_y"])
plt.show()
```

Khi index là timestamp, matplotlib có thể sử dụng timestamp làm trục X.

Điều này phù hợp nếu muốn quan sát:

> **Set kéo dài bao lâu?**

Nhưng chưa phù hợp nếu muốn biết:

> **Set có bao nhiêu samples?**



---

## 5.3. Reset index

Nếu muốn X-axis biểu diễn **sample number**:

```python
set_df["acc_y"].reset_index(
    drop=True
)
```

Sau đó plot:

```python
plt.plot(
    set_df["acc_y"].reset_index(drop=True)
)
```

Kết quả:

```text
X-axis
0 ────────────────> number of samples
```

Thay vì:

```text
X-axis
timestamp ────────> elapsed time
```

### Nhớ

**Timestamp → duration/time**

**Reset index → sample count/order**



---

# 6. Plot tất cả Exercises

Một set đơn lẻ chưa cho biết:

* exercise nào?
* participant nào?
* pattern giữa các exercise có khác nhau không?

Vì vậy cần lấy tất cả `label`.

## 6.1. Lấy unique labels

```python
labels = df["label"].unique()
```

Ví dụ:

```text
bench
squat
deadlift
row
overhead_press
rest
```

`unique()` trả về các giá trị khác nhau trong column. 

---

## 6.2. Loop qua từng exercise

```python
for label in labels:

    subset = df[
        df["label"] == label
    ]

    display(subset.head(2))
```

Logic:

```text
labels
  ↓
for từng label
  ↓
filter DataFrame
  ↓
subset
  ↓
plot
```

---

## 6.3. Tạo figure

Thay vì chỉ:

```python
plt.plot(...)
```

có thể tạo figure + axes:

```python
fig, ax = plt.subplots()

ax.plot(
    subset["acc_y"],
    label=label
)

plt.legend()
plt.show()
```

### Lợi ích

Có `fig` và `ax` → dễ:

* set title
* set labels
* customize legend
* thay đổi kích thước
* combine nhiều plots.



---

## 6.4. Plot 100 samples đầu

Để nhìn rõ pattern:

```python
ax.plot(
    subset["acc_y"][:100]
)
```

Ta có thể quan sát các pattern khác nhau:

* Bench press → sharp peaks
* Overhead press → double sharp peaks
* Squat → peak pattern khác
* Deadlift → pattern khác
* Row → pattern khác

Đây là insight quan trọng cho ML:

```text
Different exercises
        ↓
Different sensor patterns
        ↓
Potential features
        ↓
Classification
```



---

# 7. Adjust Plot Settings

Plot mặc định thường chưa tối ưu cho time-series.

Với sensor data, figure thường nên **rộng hơn** để dễ nhìn pattern theo thời gian. 

## 7.1. Dùng `style`

Matplotlib có thể áp dụng style:

```python
plt.style.use("seaborn-v0_8-deep")
```

Tên style phụ thuộc phiên bản matplotlib.

---

## 7.2. Dùng `rcParams`

Thay vì mỗi plot đều viết:

```python
figsize=...
font=...
dpi=...
```

có thể thiết lập một lần:

```python
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
```

Sau đó các figure tạo tiếp theo sẽ dùng settings này.

### Ý tưởng

```text
Set global plotting settings
          ↓
All plots inherit settings
          ↓
Less duplicated code
          ↓
Consistent figures
```



---

# 8. Compare Medium vs Heavy Sets

Dataset có:

```text
category
├── medium
└── heavy
```

Mục tiêu:

> Kiểm tra visually xem medium và heavy set có pattern khác nhau không.

---

## 8.1. `query()`

Có thể filter DataFrame bằng:

```python
category_df = df.query(
    "label == 'squat'"
)
```

Thêm participant:

```python
category_df = df.query(
    "label == 'squat' and participant == 'A'"
)
```

Sau đó:

```python
category_df = category_df.reset_index(drop=True)
```



---

## 8.2. `groupby()`

Group theo category:

```python
category_df.groupby("category")[
    "acc_y"
].plot()
```

Kết quả:

```text
             acc_y
               │
 heavy    ─────┤
 medium   ─────┤
               └──────── samples
```

`groupby()` giúp phân biệt các nhóm trong cùng figure.



---

## 8.3. Thêm labels

```python
fig, ax = plt.subplots()

category_df.groupby("category")["acc_y"].plot(
    ax=ax
)

ax.set_ylabel("Accelerometer Y")
ax.set_xlabel("Samples")
ax.legend(["heavy", "medium"])
```

Điểm cần nhớ:

> Visualization không chỉ để "vẽ đẹp", mà phải giúp giải thích data.

Trong video, pattern acceleration của medium/heavy được dùng để quan sát sự khác biệt trong chuyển động của squat. 

---

# 9. Compare Participants

ML model không chỉ cần nhận diện exercise của **một người**.

Mục tiêu là:

```text
Training participants
        ↓
Learn exercise patterns
        ↓
New participant
        ↓
Model can generalize
```

Do đó cần kiểm tra xem pattern giữa participants có tương tự nhau không. 

---

## 9.1. Filter exercise

Ví dụ bench press:

```python
participant_df = df.query(
    "label == 'bench'"
)
```

---

## 9.2. Sort theo participant

```python
participant_df = (
    participant_df
    .sort_values("participant")
    .reset_index(drop=True)
)
```

### Tại sao cần `sort_values()`?

Nếu không sort:

```text
A
C
B
A
E
C
...
```

→ khó đọc.

Sau khi sort:

```text
A A A
B B B
C C C
D D D
E E E
```

---

## 9.3. Tại sao `reset_index()`?

Dataset ban đầu sử dụng timestamp.

Các participant có thể thực hiện exercise ở những thời điểm khác nhau.

Nếu dùng timestamp trực tiếp:

```text
time
 │
 │       A
 │
 │             B
 │
 │                    C
 └──────────────────────
```

→ visualization khó so sánh.

Reset index giúp tập trung vào:

> **sample sequence**

thay vì absolute timestamp. 

---

# 10. Plot Multiple Axes

Accelerometer có 3 axes:

```text
X → left / right
Y → up / down
Z → front / back
```

Gyroscope cũng có:

```text
X
Y
Z
```

Mục tiêu:

> xem toàn bộ movement pattern của một exercise.



---

## 10.1. Dùng biến để query linh hoạt

```python
label = "squat"
participant = "A"
```

Dùng **f-string**:

```python
all_axes_df = df.query(
    f"label == '{label}' and participant == '{participant}'"
)
```

### F-string

```python
f"label == '{label}'"
```

Cho phép đưa biến vào string:

```python
name = "Huy"

f"Hello {name}"
```

→

```text
Hello Huy
```



---

# 11. Series vs DataFrame

Đây là điểm Python/Pandas rất quan trọng.

### Một cặp `[]`

```python
df["acc_x"]
```

→ `Series`

### Hai cặp `[]`

```python
df[["acc_x", "acc_y", "acc_z"]]
```

→ `DataFrame`

| Syntax                   | Result    |
| ------------------------ | --------- |
| `df["acc_x"]`            | Series    |
| `df[["acc_x", "acc_y"]]` | DataFrame |

Khi muốn chọn **nhiều columns**, phải dùng double brackets. 

---

## 11.1. Plot 3 accelerometer axes

```python
all_axes_df[
    ["acc_x", "acc_y", "acc_z"]
].plot(ax=ax)
```

Kết quả:

```text
acc_x ─────────────
acc_y ───────╮─────
acc_z ───╮───╰─────
           samples →
```

Từ đó có thể quan sát cách ba axes tương tác trong cùng một movement.



---

# 12. Loop tất cả combinations

Thay vì manually:

```text
squat + A
squat + B
bench + A
bench + B
...
```

ta lấy unique:

```python
labels = df["label"].unique()
participants = df["participant"].unique()
```

Sau đó **nested loop**:

```python
for label in labels:

    for participant in participants:

        ...
```

Logic:

```text
label
 ├── participant A
 ├── participant B
 ├── participant C
 └── ...

next label
 ├── participant A
 ├── participant B
 └── ...
```



---

## 12.1. Query động

```python
all_axes_df = (
    df.query(
        f"label == '{label}' "
        f"and participant == '{participant}'"
    )
    .reset_index(drop=True)
)
```

Mỗi vòng loop:

```text
label thay đổi
+
participant thay đổi
        ↓
query thay đổi
        ↓
plot tương ứng
```

---

## 12.2. Title động

```python
ax.set_title(
    f"{label} ({participant})".title()
)
```

Ví dụ:

```text
Squat (A)
Bench (B)
Deadlift (C)
```

---

## 12.3. Xử lý combination không có data

Không phải participant nào cũng thực hiện tất cả exercises.

Vì vậy:

```python
if len(all_axes_df) > 0:
    # create plot
```

Nếu:

```python
len(all_axes_df) == 0
```

→ bỏ qua combination đó.

Tránh tạo figure rỗng. 

---

# 13. Accelerometer → Gyroscope

Sau khi plot accelerometer, có thể dùng **cùng logic** cho gyroscope.

Chỉ cần thay sensor columns:

```text
Accelerometer
acc_x
acc_y
acc_z

        ↓

Gyroscope
gyr_x
gyr_y
gyr_z
```

Gyroscope cũng có 3 axes nhưng pattern khác accelerometer. 

---

# 14. Combine Accelerometer + Gyroscope

Mục tiêu cuối cùng:

```text
┌──────────────────────────────┐
│ Accelerometer X/Y/Z          │
├──────────────────────────────┤
│ Gyroscope X/Y/Z              │
└──────────────────────────────┘
```

Một figure → một exercise + một participant + cả hai sensor.

---

## 14.1. Tạo nhiều axes

```python
fig, ax = plt.subplots(
    nrows=2,
    sharex=True,
    figsize=(20, 10)
)
```

Ý nghĩa:

| Parameter         | Ý nghĩa                |
| ----------------- | ---------------------- |
| `nrows=2`         | 2 plots theo chiều dọc |
| `sharex=True`     | dùng chung X-axis      |
| `figsize=(20,10)` | figure lớn hơn         |



---

## 14.2. Plot từng sensor

Accelerometer:

```python
combined_df[
    ["acc_x", "acc_y", "acc_z"]
].plot(
    ax=ax[0]
)
```

Gyroscope:

```python
combined_df[
    ["gyr_x", "gyr_y", "gyr_z"]
].plot(
    ax=ax[1]
)
```

### `ax[0]` và `ax[1]`

Python index bắt đầu từ `0`:

```text
ax[0] → plot phía trên
ax[1] → plot phía dưới
```



---

# 15. Styling Combined Figure

Có thể customize legend cho từng axes:

```python
ax[0].legend(...)
ax[1].legend(...)
```

Một số options được sử dụng trong video:

* `loc`
* `bbox_to_anchor`
* `ncol`
* `fancybox`
* `shadow`

Ví dụ mục tiêu:

```text
          Accelerometer X   Y   Z
────────────────────────────────────
              plot

          Gyroscope X   Y   Z
────────────────────────────────────
              plot
```

`sharex=True` giúp không cần lặp lại X-axis label ở cả hai plot. 

---

# 16. Export tất cả Figures

Sau khi visualization hoàn chỉnh, không chỉ `plt.show()` mà cần lưu figure.

```python
plt.savefig(...)
```

Trong project:

```text
reports/
└── figures/
```

Đây là nơi lưu figures phục vụ:

* EDA
* report
* sharing với team
* phân tích tiếp theo. 

---

## 16.1. Dynamic filename

Có thể dùng f-string:

```python
f"{label.title()} ({participant}).png"
```

Ví dụ:

```text
Squat (A).png
Squat (B).png
Bench (A).png
Bench (B).png
```

Như vậy tên file tự động phản ánh:

```text
exercise + participant
```



---

## 16.2. DPI

Khi export:

```python
plt.savefig(
    path,
    dpi=100
)
```

`dpi` ảnh hưởng đến resolution của output.

Mục tiêu:

```text
Visualization
     ↓
High-resolution PNG
     ↓
Report / presentation / team sharing
```

Video sử dụng DPI 100 và ghi nhận output có resolution lớn, phù hợp để đưa vào report. 

---

# 17. Toàn bộ Pipeline của Part 3

Đây là phần **quan trọng nhất để nhớ**:

```text
Load processed data
        ↓
Fix / verify columns
        ↓
Select subset
        ↓
Plot single sensor
        ↓
Understand individual exercises
        ↓
Compare medium vs heavy
        ↓
Compare participants
        ↓
Plot X/Y/Z axes
        ↓
Loop over exercise × participant
        ↓
Plot accelerometer
        ↓
Plot gyroscope
        ↓
Combine both sensors
        ↓
Export figures
```

---

# 18. Các Pandas / Matplotlib concepts cần nhớ

| Concept          | Dùng để                       |
| ---------------- | ----------------------------- |
| `df["column"]`   | lấy một Series                |
| `df[["a","b"]]`  | lấy nhiều columns → DataFrame |
| `df[...]`        | filter bằng điều kiện         |
| `.query()`       | filter bằng expression        |
| `.unique()`      | lấy các giá trị khác nhau     |
| `.groupby()`     | chia data thành groups        |
| `.sort_values()` | sắp xếp data                  |
| `.reset_index()` | tạo lại index                 |
| `plt.plot()`     | tạo line plot                 |
| `plt.subplots()` | tạo figure + axes             |
| `ax[0]`          | truy cập subplot đầu          |
| `plt.rcParams`   | global plot settings          |
| `f"..."`         | dynamic string                |
| `plt.savefig()`  | export figure                 |

---

# 19. Key Takeaways

### ① Visualization phải có mục tiêu

Không phải:

> "Có data → plot."

Mà là:

> "Tôi muốn figure này giúp tôi hiểu điều gì?"

---

### ② Sensor data có pattern

Mỗi exercise có movement pattern khác nhau.

```text
Exercise
   ↓
Movement
   ↓
Sensor signal
   ↓
Pattern
   ↓
Potential ML features
```

Đây là lý do visualization quan trọng trước modeling. 

---

### ③ Cần kiểm tra variation giữa participants

Model cần generalize, không chỉ memorize movement của một người.

```text
Participant A ─┐
Participant B ─┤
Participant C ─┼→ ML model
Participant D ─┤
Participant E ─┘
                  ↓
             New participant
```

---

### ④ Accelerometer và gyroscope bổ sung cho nhau

```text
Accelerometer
→ acceleration / movement

Gyroscope
→ rotational / orientation-related information
```

Kết hợp hai sensor giúp có cái nhìn đầy đủ hơn về movement data. 

---

### ⑤ Visualization là bước chuẩn bị cho ML

Part 3 chưa xây model.

Nó giúp trả lời:

```text
Data có pattern không?
        ↓
Các exercise có khác nhau không?
        ↓
Heavy / medium có khác nhau không?
        ↓
Participants có khác nhau không?
        ↓
Sensor nào chứa thông tin hữu ích?
        ↓
→ Feature engineering
→ Outlier detection
→ Modeling
```

Đây chính là vai trò của Part 3 trong toàn bộ fitness-tracker project. 

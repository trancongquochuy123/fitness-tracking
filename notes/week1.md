Dưới đây là **handnote đầy đủ cho Part 1**, mình chuyển transcript thành dạng ghi chú học Data Science/ML, giữ lại các ý quan trọng nhưng bỏ phần nói lan man của video.

# Handnote — Fitness Tracker with Python

## Part 1: Introduction

### 1. Project Overview

**Mục tiêu:** xây dựng một **Fitness Tracker bằng Python + Machine Learning** có khả năng:

1. Nhận dữ liệu từ **accelerometer** và **gyroscope**.
2. Xử lý và phân tích sensor data.
3. Nhận diện bài tập đang thực hiện.
4. Đếm số repetitions.

Pipeline tổng quát:

```text
Sensor Data
    ↓
Data Processing
    ↓
Visualization
    ↓
Outlier Detection
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Exercise Classification
    ↓
Repetition Counting
```

Ví dụ:

```text
Accelerometer + Gyroscope
        ↓
   Sensor Time Series
        ↓
   Feature Extraction
        ↓
     ML Classifier
        ↓
    "Squat"
        ↓
   Repetition Counter
        ↓
      "10 reps"
```

---

# 2. Bài toán Machine Learning

Project tập trung vào **barbell exercises**.

Các exercise trong dataset:

| Exercise       | Nhóm cơ chính          |
| -------------- | ---------------------- |
| Bench Press    | Chest                  |
| Deadlift       | Back / posterior chain |
| Overhead Press | Shoulders              |
| Barbell Row    | Back                   |
| Squat          | Legs                   |

### Classification problem

Input:

```text
Sensor time-series data
```

Output:

```text
Exercise class
```

Ví dụ:

```text
Accelerometer + Gyroscope
        ↓
       Model
        ↓
     "Squat"
```

Sau đó cần giải quyết thêm:

```text
"Squat" + sensor data
        ↓
  Repetition Counter
        ↓
      10 reps
```

Vì vậy project thực chất gồm **2 bài toán**:

### Problem 1 — Exercise Classification

> Người dùng đang thực hiện bài tập nào?

```text
Sensor data → Exercise type
```

### Problem 2 — Repetition Counting

> Người dùng đã thực hiện bao nhiêu lần?

```text
Sensor data → Number of repetitions
```

---

# 3. Project Background

Dataset được thu thập bằng **MetaMotion sensor** của **MbientLab**.

Đây là một research-oriented wearable sensor, không phải smartwatch hoàn chỉnh.

Sensor có thể:

* thu thập dữ liệu chuyển động
* kết nối với điện thoại qua Bluetooth
* ghi sensor measurements
* export dữ liệu thành CSV

Các sensor quan trọng trong project:

```text
Accelerometer
Gyroscope
```

### Accelerometer

Đo **gia tốc** theo các trục.

Thông thường:

```text
X-axis
Y-axis
Z-axis
```

Có thể dùng để phát hiện:

* chuyển động
* hướng
* thay đổi tốc độ
* pattern của exercise

### Gyroscope

Đo **angular velocity / tốc độ quay**.

Cũng thường có:

```text
X-axis
Y-axis
Z-axis
```

Gyroscope đặc biệt hữu ích khi chuyển động liên quan đến:

* rotation
* orientation
* wrist movement

---

# 4. Dataset

Dataset được thu thập trong quá trình tập gym.

Có tổng cộng:

```text
5 participants
```

bao gồm:

* tác giả
* 4 người khác

Mỗi participant thực hiện các barbell exercises.

Sensor được đeo trên **wrist**.

Mỗi exercise tạo ra một **distinct pattern** trong sensor data.

Ví dụ về mặt ý tưởng:

```text
Squat
────────────────────
acceleration pattern
       /\      /\
      /  \    /  \
_____/    \__/    \____


Bench Press
────────────────────
acceleration pattern
     /\    /\    /\
____/  \__/  \__/  \___
```

Machine Learning sẽ học các pattern này để phân biệt exercise.

---

# 5. Quantified Self

## Definition

**Quantified Self** là khái niệm về việc một cá nhân:

> tự theo dõi các thông tin sinh học, vật lý, hành vi hoặc môi trường của bản thân nhằm đạt một mục tiêu cụ thể và sử dụng dữ liệu thu thập được để đưa ra hành động.

Có thể hiểu đơn giản:

```text
Self
 ↓
Collect Data
 ↓
Analyze Data
 ↓
Understand Yourself
 ↓
Take Action
```

Ví dụ:

```text
Sleep tracking
Heart rate
Exercise
Calories
Steps
Stress
Recovery
```

Các thiết bị phổ biến:

* Apple Watch
* WHOOP
* Oura Ring
* các wearable devices khác

---

# 6. Why Quantified Self Matters

Ngày càng nhiều thiết bị có khả năng thu thập dữ liệu liên tục.

Ví dụ:

```text
Wearable
   ↓
Sensor
   ↓
Time-series data
   ↓
Machine Learning
   ↓
Insight
   ↓
Action
```

Ví dụ:

```text
Heart rate + Sleep
        ↓
     Analysis
        ↓
"Sleep quality is low"
        ↓
Change sleep behavior
```

Điểm quan trọng:

> Quantified Self không chỉ là **collect data**.

Mà phải có:

```text
Measurement
    +
Goal
    +
Action
```

---

# 7. Sensor Data ≈ Time-Series Data

Kiến thức trong project không chỉ áp dụng cho fitness.

Sensor data thường là **time-series data**.

Ví dụ:

```text
Timestamp    Acc_X    Acc_Y    Acc_Z
10:00:01     0.12     1.04     9.72
10:00:02     0.21     1.10     9.61
10:00:03     0.45     1.31     9.42
...
```

Mỗi observation gắn với một timestamp.

Vì vậy:

```text
Sensor / IoT
      ↓
Time Series
      ↓
Data Processing
      ↓
Feature Engineering
      ↓
Machine Learning
```

---

# 8. General Applications

Các kỹ thuật trong project có thể áp dụng cho nhiều loại sensor data.

### Fitness

```text
Accelerometer
+
Gyroscope
      ↓
Exercise recognition
```

### Predictive Maintenance

Sensor gắn trên máy móc:

```text
Machine
   ↓
Vibration Sensor
   ↓
Time-series data
   ↓
Anomaly Detection
   ↓
Predictive Maintenance
```

Có thể phát hiện:

* vibration bất thường
* machine degradation
* potential failure

### IoT

```text
IoT Device
   ↓
Sensors
   ↓
Continuous measurements
   ↓
Time-series database
   ↓
ML / Analytics
```

=> Kiến thức của project có tính transferable sang:

* IoT
* industrial monitoring
* predictive maintenance
* wearable technology
* activity recognition
* anomaly detection

---

# 9. Seven-Week Project Roadmap

Project gồm **7 tuần / 7 phần chính**.

```text
Week 1
Introduction
     ↓
Week 2
Raw Data Processing
     ↓
Week 3
Visualization
     ↓
Week 4
Outlier Detection
     ↓
Week 5
Feature Engineering
     ↓
Week 6
Predictive Modeling
     ↓
Week 7
Repetition Counting
```

---

# 10. Week 1 — Introduction

Đây là phần hiện tại.

Cần hiểu:

* project objective
* dataset
* sensor
* Quantified Self
* time-series sensor data
* project roadmap

Chưa đi sâu vào coding.

---

# 11. Week 2 — Raw Data Processing

Mục tiêu:

> biến raw sensor data thành dataset có thể sử dụng cho ML.

Các bước:

```text
CSV files
   ↓
Read data
   ↓
Inspect data
   ↓
Split data
   ↓
Clean data
   ↓
Prepare dataset
```

Thư viện chính:

```python
import pandas as pd
```

### Pandas

Pandas rất quan trọng trong Data Science vì được sử dụng để:

* đọc CSV
* manipulate data
* filtering
* cleaning
* grouping
* aggregation
* transforming data

Ví dụ:

```python
df = pd.read_csv("data.csv")
```

Kiểm tra dataset:

```python
df.head()
df.info()
df.describe()
df.shape
```

---

# 12. Week 3 — Data Visualization

Mục tiêu:

> hiểu sensor data bằng visualization trước khi xây dựng model.

Đặc biệt quan trọng với time-series data.

Ví dụ:

```text
Time
 ↓
Sensor measurements
 ↓
Plot
 ↓
Identify patterns
```

Có thể quan sát:

* exercise pattern
* noise
* spikes
* missing data
* abnormal movement
* differences giữa các exercises

Một nguyên tắc quan trọng:

> Không nên chỉ nhìn vào dataframe. Hãy visualize data.

---

# 13. Week 4 — Outlier Detection

Sensor data thường **noisy**.

Ví dụ người dùng đang squat:

```text
Squat
↓
Squat
↓
Squat
↓
Rest
↓
Adjust wrist
↓
Squat
```

Sensor vẫn ghi nhận:

```text
Squat data
+
unwanted movement
```

Movement không thuộc exercise chính sẽ trở thành noise / outlier.

### Problem

Nếu đưa toàn bộ data vào model:

```text
Useful signal + Noise
       ↓
      ML
       ↓
Poorer model
```

Vì vậy cần phát hiện và xử lý outliers.

Các kỹ thuật outlier detection sẽ được học ở Week 4.

---

# 14. Week 5 — Feature Engineering

Đây là một phần rất quan trọng.

Raw sensor data:

```text
X(t)
Y(t)
Z(t)
```

không nhất thiết là input tốt nhất cho ML.

Ta cần **extract meaningful features**.

```text
Raw Time Series
      ↓
Feature Engineering
      ↓
Numerical Features
Frequency Features
Time Features
Filtered Signals
PCA
Clustering
      ↓
ML Dataset
```

---

## 14.1 Numerical Features

Có thể extract các statistical properties như:

```text
Mean
Standard deviation
Minimum
Maximum
Variance
Range
...
```

Ví dụ:

```python
mean = signal.mean()
std = signal.std()
maximum = signal.max()
minimum = signal.min()
```

Mục tiêu:

> chuyển một đoạn time series thành một vector đặc trưng có ý nghĩa.

---

# 15. Frequency Features

Sensor signal không chỉ có thông tin trong **time domain**.

Có thể phân tích nó trong **frequency domain**.

Concept:

```text
Time Domain
     ↓
Frequency Analysis
     ↓
Frequency Features
```

Ví dụ có thể tìm:

* dominant frequency
* frequency components
* energy ở các frequency bands

Điều này hữu ích vì các exercise khác nhau có thể có movement frequency khác nhau.

Ví dụ:

```text
Exercise A
→ movement frequency ≈ X

Exercise B
→ movement frequency ≈ Y
```

---

# 16. Time Features

Ngoài numerical và frequency features, có thể tạo các feature liên quan đến thời gian.

Ví dụ:

```text
Duration
Timestamp
Time between movements
Window length
```

Time-series thường được chia thành các **windows** để tạo feature.

Concept:

```text
Continuous Signal
────────────────────────────────────

        Window 1
     ┌───────────┐
─────┤           ├──────────────────

                   Window 2
              ┌───────────┐
──────────────┤           ├──────────

                           Window 3
                      ┌───────────┐
──────────────────────┤           ├──
```

Mỗi window có thể trở thành một training sample.

---

# 17. Filtering

Sensor data thường có noise.

Một kỹ thuật được đề cập là:

### Low-pass filtering

Concept:

```text
Raw Signal
   ↓
Low-pass Filter
   ↓
Smoothed Signal
```

Low-pass filter giữ lại các thành phần có tần số thấp và giảm các high-frequency components.

Ứng dụng:

```text
Noisy movement
     ↓
Filtering
     ↓
Cleaner signal
     ↓
Better features
```

---

# 18. Principal Component Analysis — PCA

PCA cũng được sử dụng trong feature engineering / dimensionality reduction.

Mục tiêu:

```text
Many correlated features
        ↓
       PCA
        ↓
Fewer principal components
```

Ví dụ:

```text
Feature 1
Feature 2
Feature 3
Feature 4
Feature 5
Feature 6
        ↓
       PCA
        ↓
   PC1 + PC2
```

PCA giúp:

* giảm dimensionality
* visualize data
* tìm các directions có variance lớn
* giảm redundancy

---

# 19. Clustering

Project cũng đề cập đến **clustering**.

Clustering là **unsupervised learning**.

Không cần label trực tiếp.

Concept:

```text
Sensor samples
      ↓
   Clustering
      ↓
Groups / Patterns
```

Ví dụ:

```text
● ● ●       ▲ ▲ ▲       ■ ■ ■
● ● ●       ▲ ▲ ▲       ■ ■ ■
```

Có thể giúp khám phá:

* các movement patterns
* similarity giữa samples
* structure của dataset

---

# 20. Week 6 — Predictive Modeling

Sau khi data đã được:

```text
Cleaned
   ↓
Visualized
   ↓
Outliers handled
   ↓
Features extracted
```

ta bắt đầu xây dựng ML models.

Mục tiêu:

> phân loại barbell exercise.

```text
Features
   ↓
Classifier
   ↓
Exercise
```

Ví dụ:

```text
Sensor features
      ↓
   Classifier
      ↓
"Deadlift"
```

---

# 21. Multiple Classifiers

Không chỉ train một model.

Project sẽ xây dựng và so sánh **nhiều classifiers**.

Concept:

```text
Dataset
   ↓
 ┌───────────────┐
 ↓       ↓       ↓
Model A Model B Model C
 ↓       ↓       ↓
Performance comparison
        ↓
Evaluation
```

Các model cụ thể sẽ được học ở phần predictive modeling.

Điểm quan trọng là:

> Không nên mặc định rằng một algorithm luôn tốt nhất cho mọi dataset.

Cần train, evaluate và compare dựa trên dữ liệu thực tế.

---

# 22. Final ML Pipeline

Sau Week 6, hệ thống về cơ bản sẽ có:

```text
Sensor Data
     ↓
Data Processing
     ↓
Outlier Detection
     ↓
Feature Engineering
     ↓
ML Classifier
     ↓
Exercise Prediction
```

Ví dụ:

```text
Accelerometer
+
Gyroscope
       ↓
   Processing
       ↓
    Features
       ↓
    Classifier
       ↓
    "Squat"
```

---

# 23. Week 7 — Repetition Counting

Sau khi biết exercise là gì, cần biết:

> Có bao nhiêu repetitions?

Ví dụ:

```text
Sensor Data
     ↓
Exercise Classifier
     ↓
Squat
     ↓
Repetition Algorithm
     ↓
10 repetitions
```

Đây là một bài toán khác với classification.

### Classification

```text
"What exercise?"
```

### Repetition counting

```text
"How many times?"
```

---

# 24. Final Fitness Tracker

Mục tiêu cuối cùng:

```text
           Wearable Sensor
                 ↓
      Accelerometer + Gyroscope
                 ↓
          Sensor Monitoring
                 ↓
         Data Processing
                 ↓
        Feature Engineering
                 ↓
        Exercise Classifier
                 ↓
        Exercise Recognition
                 ↓
        Repetition Counter
                 ↓
        Workout Summary
```

Ví dụ:

```text
User enters gym
      ↓
Does squats
      ↓
App automatically detects:
      ↓
Exercise = Squat
Reps = 10
Sets = 3
      ↓
User enters weight manually
      ↓
Workout saved
```

---

# 25. "Other Way Around" Fitness Tracking

Một ý tưởng UX quan trọng của project:

### Traditional approach

Người dùng phải:

```text
Open app
 ↓
Select exercise
 ↓
Enter weight
 ↓
Start workout
 ↓
Do exercise
 ↓
Enter reps
```

### ML-based approach

```text
Enter gym
    ↓
Start exercising
    ↓
Sensor automatically monitors movement
    ↓
ML detects exercise
    ↓
ML counts reps
    ↓
Finish workout
    ↓
Review detected workout
    ↓
Fill missing information
```

Ví dụ:

```text
"I noticed you did:

Squat
3 sets
10 reps each

Is that correct?"
```

Người dùng chỉ cần xác nhận và nhập những thông tin sensor không thể biết, ví dụ:

```text
Weight = 100 kg
```

---

# 26. Why This Is a Good ML Project

Project kết hợp nhiều phần quan trọng của Data Science:

```text
Python
   +
Pandas
   +
Time Series
   +
Sensor Data
   +
Data Cleaning
   +
Visualization
   +
Outlier Detection
   +
Feature Engineering
   +
Dimensionality Reduction
   +
Clustering
   +
Classification
   +
Signal Processing
```

Đây không chỉ là một project:

```text
CSV → sklearn → accuracy
```

mà là một **end-to-end ML project**.

---

# 27. Generalizable Knowledge

Điều quan trọng nhất không phải chỉ là xây Fitness Tracker.

Kiến thức có thể áp dụng cho nhiều domain:

### Wearable

```text
Accelerometer
Gyroscope
Heart rate
      ↓
Activity recognition
```

### IoT

```text
IoT sensors
      ↓
Time-series
      ↓
Anomaly detection
```

### Manufacturing

```text
Machine vibration
      ↓
Feature extraction
      ↓
Failure prediction
```

### Healthcare

```text
Sensor measurements
      ↓
Pattern recognition
      ↓
Activity / condition detection
```

### Smart devices

```text
Sensor data
      ↓
ML
      ↓
Automatic activity recognition
```

---

# 28. Tools / Technologies

Các công cụ chính:

```text
Python
Pandas
Visualization libraries
Scikit-learn
Signal processing techniques
VS Code
CSV
Git / GitHub
```

Đặc biệt project sử dụng:

### VS Code

Không sử dụng Jupyter Notebook làm môi trường chính.

Thay vào đó:

> **Interactive Python trong Visual Studio Code**

Điều này phù hợp với workflow của project.

---

# 29. Project Setup

Để follow project cần:

### Step 1 — VS Code

Setup Visual Studio Code cho Data Science.

### Step 2 — Project Template

Download project template.

Có thể:

```bash
git clone <repository>
```

hoặc download ZIP.

### Step 3 — Workspace

Mở project template thành VS Code workspace.

### Step 4 — Dataset

Dataset sẽ được cung cấp ở phần tiếp theo.

---

# 30. Resources

Các resource được nhắc tới:

### Book

**Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data**

Nội dung tập trung vào:

* machine learning
* quantified self
* sensory data
* time-series
* sensor-based ML

Book **không bắt buộc** để follow project.

Project thiên về:

> **Hands-on applied machine learning**

hơn là theoretical ML.

---

# 31. Key Concepts to Remember

## Concept 1 — Sensor Data

```text
Sensor
 ↓
Measurements
 ↓
Time Series
```

---

## Concept 2 — Time Series

Data phụ thuộc vào thời gian:

```text
t1 → measurement
t2 → measurement
t3 → measurement
...
```

---

## Concept 3 — Signal

Sensor measurements tạo thành signal.

```text
Signal = measurement changing over time
```

---

## Concept 4 — Noise

Không phải mọi measurement đều chứa information hữu ích.

```text
Signal + Noise
```

Cần xử lý noise trước ML.

---

## Concept 5 — Features

Raw sensor data → meaningful numerical representation.

```text
Raw data
   ↓
Features
   ↓
Machine Learning
```

---

## Concept 6 — Classification

Predict category:

```text
Input → Class
```

Trong project:

```text
Sensor data → Exercise
```

---

## Concept 7 — Repetition Counting

Predict số lần lặp:

```text
Movement signal
      ↓
Repetition detection
      ↓
Number of reps
```

---

# 32. Complete Project Architecture

```text
                ┌───────────────────┐
                │   Wearable Sensor │
                └─────────┬─────────┘
                          │
              Accelerometer + Gyroscope
                          │
                          ▼
                ┌───────────────────┐
                │    Raw Dataset    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Data Processing   │
                │ Pandas / Cleaning │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │  Visualization    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Outlier Detection │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Feature Engineering│
                │                   │
                │ Numerical         │
                │ Frequency         │
                │ Time              │
                │ Filtering         │
                │ PCA               │
                │ Clustering        │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Predictive Model  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │Exercise Classifier│
                └─────────┬─────────┘
                          │
                          ▼
                  "Squat / Deadlift
                   / Bench Press..."
                          │
                          ▼
                ┌───────────────────┐
                │ Repetition Counter│
                └─────────┬─────────┘
                          │
                          ▼
                  "Squat — 10 reps"
```

---

# 33. Week 1 — Checklist

Trước khi sang Part 2, cần nắm được:

### Project

* [x] Fitness Tracker
* [x] Python
* [x] Machine Learning
* [x] Sensor data

### Sensor

* [x] Accelerometer
* [x] Gyroscope
* [x] Wearable sensor
* [x] Time-series data

### ML Problem

* [x] Exercise classification
* [x] Repetition counting

### Dataset

* [x] 5 participants
* [x] Wrist sensor
* [x] Barbell exercises
* [x] CSV data

### Data Science Pipeline

* [x] Data processing
* [x] Visualization
* [x] Outlier detection
* [x] Feature engineering
* [x] Predictive modeling

### Advanced concepts

* [x] Frequency features
* [x] Filtering
* [x] PCA
* [x] Clustering

### Environment

* [x] VS Code
* [x] Interactive Python
* [x] Project template

---

# 34. One-Minute Summary

Nếu cần ghi nhớ Part 1 bằng **một flow duy nhất**:

```text
Wearable Sensor
      ↓
Accelerometer + Gyroscope
      ↓
Time-Series Data
      ↓
Clean Data
      ↓
Visualize
      ↓
Detect Outliers
      ↓
Feature Engineering
      ↓
Train ML Classifiers
      ↓
Recognize Exercise
      ↓
Count Repetitions
      ↓
Automatic Fitness Tracker
```

**Core idea:**

> Dùng **sensor time-series data + signal processing + feature engineering + machine learning** để biến chuyển động của người dùng thành thông tin có ý nghĩa như **“đang tập squat”** và **“đã thực hiện 10 reps”**.

**Part 2 sẽ bắt đầu từ phần quan trọng nhất của workflow: lấy raw CSV sensor data và thực hiện data processing bằng Pandas.**

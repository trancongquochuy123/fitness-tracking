import os
import sys
import pandas as pd
from glob import glob

# Console Windows (cp1252) không encode được tiếng Việt có dấu trong
# các câu print() -> ép stdout sang UTF-8 để tránh UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
# Bước đầu tiên: đọc thử MỘT file để hiểu cấu trúc dữ liệu trước khi
# tự động hóa cho toàn bộ 187 file (xem notes/week2.md - mục 13, 14).
# Mỗi recording tạo ra 2 file riêng biệt: Accelerometer và Gyroscope.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path trỏ tới thư mục chứa raw CSV, dùng để ghép với tên file
# khi đọc single file / glob toàn bộ *.csv.
data_path = os.path.join(BASE_DIR, "raw", "MetaMotion") + os.sep

single_file_acc = pd.read_csv(
    data_path
    + "A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    data_path
    + "A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# Kiểm tra nhanh để chắc chắn đọc đúng file trước khi tự động hóa.
print("single_file_acc:", single_file_acc.shape)
print(single_file_acc.head())
print("single_file_gyr:", single_file_gyr.shape)
print(single_file_gyr.head())


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
# Dùng glob để lấy danh sách toàn bộ file .csv trong thư mục raw.
# "*.csv" nghĩa là lấy mọi file có đuôi .csv, bỏ qua các file khác
# (ví dụ .txt, .png) nếu có trong cùng thư mục.
files = glob(data_path + "*.csv")
print(
    "Tổng số file CSV tìm được:", len(files)
)  # kỳ vọng ra 187 file theo notes/week2.md


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
# Metadata (participant, exercise label, category) KHÔNG nằm trong
# nội dung CSV mà nằm trong TÊN FILE, ví dụ:
#   "A-bench-heavy2-rpe8_MetaWear_2019-01-11T..._Accelerometer_..."
#   -> participant = A, label = bench, category = heavy2 (-> heavy)
# Đây là dữ liệu "semi-structured" (xem notes/week2.md - mục 12, 19-24).

f = files[0]

# QUAN TRỌNG: phải lấy os.path.basename(f) (chỉ tên file) rồi mới
# split theo "-", KHÔNG được split trực tiếp trên full path (f).
# Lý do: full path là
#   "...\Fitness-tracking\data\raw\MetaMotion\A-bench-...csv"
# và chính thư mục project "Fitness-tracking" cũng chứa dấu "-"!
# Nếu split("-") trên cả path, phần tử đầu tiên sẽ bị cắt ngay tại
# dấu "-" đó (ra "...Fitness"), làm lệch toàn bộ index phía sau ->
# participant/label/category đều sai (participant lẫn luôn path,
# label/category bị dịch chuyển 1 vị trí). Đây là lỗi thực tế đã gặp
# khi kiểm tra output ở Part 3 (notes/week3.md - mục 1).
filename = os.path.basename(f)

participant = filename.split("-")[0]
label = filename.split("-")[1]
# "heavy2".rstrip("123") -> "heavy": bỏ số thứ tự set (1/2/3) ở cuối.
# .rstrip("_MetaWear_2019") xử lý trường hợp không có "-rpeN" phía sau,
# khi đó phần category bị dính liền với "_MetaWear_2019" (ví dụ file
# "A-dead-heavy_MetaWear_2019-..."), cần cắt bỏ phần đuôi dư này.
category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# Kiểm tra metadata extract đúng chưa trước khi áp dụng cho toàn bộ file.
print("participant:", participant, "| label:", label, "| category:", category)
print(df[["participant", "label", "category"]].head())


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
# Lặp qua toàn bộ 187 file: đọc CSV, extract metadata từ filename,
# gắn thêm "set" id để phân biệt từng recording, rồi tách riêng
# accelerometer và gyroscope vào 2 DataFrame khác nhau (notes - mục
# 27-33).

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# Set id tăng dần, mỗi loại sensor có counter riêng. Đây KHÔNG phải
# "set tập thứ mấy" trong thực tế, chỉ là id kỹ thuật để reference
# nhanh một recording cụ thể (notes - mục 39, 40).
acc_set = 1
gyr_set = 1

for f in files:
    # Luôn lấy basename trước khi split("-") - xem giải thích chi
    # tiết ở section "Extract features from filename" phía trên.
    filename = os.path.basename(f)
    participant = filename.split("-")[0]
    label = filename.split("-")[1]
    category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    # Dùng "in" để kiểm tra loại sensor dựa trên tên file (partial
    # string match), vì bản thân nội dung CSV không ghi rõ sensor type.
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

# Kiểm tra số lượng record và số set sau khi gộp toàn bộ file.
print("acc_df shape:", acc_df.shape, "| số set:", acc_set - 1)
print("gyr_df shape:", gyr_df.shape, "| số set:", gyr_set - 1)


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
# Cột "epoch (ms)" là Unix timestamp tính bằng milliseconds. Cần
# convert sang datetime thật để có thể dùng các thao tác time-series
# (resample, index theo thời gian...) ở bước sau (notes - mục 2-7).

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# Sau khi đã có DatetimeIndex, các cột thời gian gốc (epoch, time,
# elapsed) không còn cần thiết -> xóa để DataFrame gọn hơn.
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# Kiểm tra DatetimeIndex đã được gán đúng chưa.
print(acc_df.info())
print(gyr_df.info())

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
# Đóng toàn bộ logic đọc + extract metadata + xử lý datetime ở trên
# thành một function duy nhất để dễ tái sử dụng, tránh duplicate code
# khi cần đọc lại dữ liệu ở file/notebook khác (notes - mục 9, 10).


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        filename = os.path.basename(f)
        participant = filename.split("-")[0]
        label = filename.split("-")[1]
        category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


files = glob(data_path + "*.csv")
acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
# Gộp accelerometer + gyroscope vào MỘT DataFrame duy nhất theo cột
# (axis=1), dựa trên DatetimeIndex đã tạo ở bước trước (notes - mục
# 11-13).
#
# acc_df và gyr_df đều có cùng các cột metadata (participant, label,
# category, set) vì được gắn từ cùng logic -> chỉ cần giữ lại 3 cột
# đầu tiên (x, y, z) của acc_df, metadata còn lại lấy từ gyr_df để
# tránh trùng lặp cột.
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# Kiểm tra sau khi merge: số cột, số dòng, và NaN do 2 sensor không
# cùng sampling frequency (sẽ được xử lý ở bước resample tiếp theo).
print("data_merged shape:", data_merged.shape)
print(data_merged.isna().sum())
print(data_merged.head())


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# Vì 2 sensor có sampling frequency khác nhau, sau khi merge sẽ xuất
# hiện rất nhiều NaN (timestamp của 2 sensor hiếm khi trùng nhau).
# Giải pháp: resample cả 2 về cùng một frequency chung là 200ms
# (~5Hz) - đủ chi tiết cho bài toán nhưng không tạo quá nhiều dữ liệu
# dư thừa (notes - mục 14-22).

# Mỗi loại cột cần một kiểu aggregation khác nhau khi resample:
# - Cột numerical (acc_x, acc_y, ... gyr_z): lấy "mean" vì là giá trị
#   đo liên tục, có thể tính trung bình trong mỗi khoảng 200ms.
# - Cột categorical (participant, label, category, set): lấy "last"
#   vì tính trung bình không có ý nghĩa với dữ liệu dạng nhãn.
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# Dataset trải dài nhiều ngày khác nhau. Nếu resample trực tiếp trên
# toàn bộ khoảng thời gian, Pandas sẽ tạo record cho MỌI khoảng 200ms
# kể cả lúc không ghi dữ liệu (giữa các ngày) -> DataFrame phình to
# không cần thiết. Giải pháp: tách theo từng ngày trước, resample
# riêng từng ngày, rồi ghép lại (notes - mục 27-29).
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

# Sau resample, cột "set" có thể bị chuyển thành float (do NaN từng
# xuất hiện trước khi dropna) -> convert lại về int cho đúng ý nghĩa
# dữ liệu (set là id, không phải số thực).
data_resampled["set"] = data_resampled["set"].astype("int")

# Kiểm tra kết quả sau resample: không còn NaN, dtype đúng, số dòng
# giảm đáng kể so với data_merged (do gộp nhiều sample vào mỗi 200ms).
print("data_resampled shape:", data_resampled.shape)
print(data_resampled.info())
print(data_resampled.head())


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
# Lưu DataFrame đã xử lý thành file pickle (.pkl) trong data/interim.
# Dùng Pickle thay vì CSV vì đây là dữ liệu trung gian (intermediate)
# chỉ dùng lại trong Python: giữ nguyên dtype, DatetimeIndex... mà
# không cần convert lại từ đầu mỗi lần load (notes - mục 34-36).
interim_path = os.path.join(BASE_DIR, "interim", "01_data_processed.pkl")
data_resampled.to_pickle(interim_path)

# Xác nhận file đã được lưu thành công.
print("Đã lưu dataset đã xử lý tại:", interim_path)
print("Kích thước dataset cuối cùng:", data_resampled.shape)

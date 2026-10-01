import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Console Windows (cp1252) không encode được tiếng Việt có dấu trong
# các câu print() -> ép stdout sang UTF-8 để tránh lỗi UnicodeEncodeError
# (giống cách đã xử lý trong data/make_dataset.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# File này nằm cùng thư mục với plot_settings.py. Khi chạy trực tiếp,
# Python tự thêm thư mục chứa script vào sys.path, nên import được
# luôn như một module cùng cấp (không cần sys.path.append thủ công).
#
# Chỉ cần IMPORT là đủ để áp dụng toàn bộ cấu hình (style, màu sắc,
# kích thước figure...) đã định nghĩa sẵn trong plot_settings.py cho
# MỌI figure tạo ra sau dòng này trong file. Đây là "side-effect
# import": không cần gọi gì thêm từ module đó, chỉ cần import 1 lần
# ở đầu file (notes/week3.md - mục 7: "set global settings 1 lần,
# mọi figure sau đó tự động dùng chung").
import plot_settings  # (import chỉ để kích hoạt cấu hình, không gọi hàm nào từ nó)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "01_data_processed.pkl")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


# --------------------------------------------------------------
# 4. Nạp dữ liệu (Loading Data)
# --------------------------------------------------------------
# Đọc lại dataset đã xử lý ở Part 2 (data/make_dataset.py). Dùng
# pickle (.pkl) vì cần giữ nguyên DatetimeIndex + dtype của từng cột,
# không phải đọc CSV và xử lý lại từ đầu mỗi lần chạy visualize.py
# (notes/week3.md - mục 4).
df = pd.read_pickle(DATA_PATH)

print("Kích thước dataset:", df.shape)
print(df.head())
print("Các cột:", list(df.columns))
# Kỳ vọng: participant, label (tên exercise), category (heavy/medium),
# acc_x/y/z, gyr_x/y/z, set — đúng như bảng mô tả trong notes/week3.md.


# --------------------------------------------------------------
# 5. Plot một cột dữ liệu của một set
# --------------------------------------------------------------
# 5.1 Cô lập (isolate) MỘT set trước khi vẽ.
#
# Vì sao? Nếu vẽ thẳng toàn bộ dataset (9000+ dòng, 5 participant,
# 6 exercise trộn lẫn), đường vẽ ra sẽ như một mớ chỉ rối, không nhìn
# ra được pattern nào cả:
#
#   Toàn bộ dataset (KHÔNG filter):
#   acc_y │ ╱╲╱╲╲╱╲╱╲╲╱╲╲╱╲╱╲╲╱╲╱╲╲╱╲╱╲╲╱╲╱╲╲  ← không đọc được gì
#         └──────────────────────────────────
#
#   Chỉ 1 set (CÓ filter set == 1):
#   acc_y │     ╱╲  ╱╲  ╱╲  ╱╲         ← thấy rõ 4 lần lặp động tác
#         └──────────────────────────
#
# Nên luôn bắt đầu bằng cách cô lập 1 set để quan sát pattern rõ ràng.
set_df = df[df["set"] == 1]

# 5.2 Vẽ trực tiếp theo DatetimeIndex -> trục X là THỜI GIAN THẬT
# (ví dụ 15:08:05, 15:08:06...). Phù hợp khi câu hỏi là:
#       "Set này kéo dài bao lâu (bao nhiêu giây)?"
fig, ax = plt.subplots()
ax.plot(set_df["acc_y"])
ax.set_ylabel("acc_y")
ax.set_xlabel("Thời gian")
ax.set_title("Set 1 - acc_y theo thời gian thực")
plt.show()

# 5.3 reset_index(drop=True) -> bỏ timestamp, đánh lại index từ
# 0, 1, 2, 3... Trục X khi đó là SỐ THỨ TỰ SAMPLE, không còn là thời
# gian. Phù hợp khi câu hỏi là:
#       "Set này có bao nhiêu sample (điểm đo)?"
#
#   Trước reset_index          Sau reset_index
#   X = 15:08:05.2  ...        X = 0, 1, 2, 3...
#   X = 15:08:05.4              (chỉ đếm số điểm, bỏ qua mốc giờ)
#   X = 15:08:05.6
#
# Ghi nhớ: Timestamp -> trả lời câu hỏi về THỜI LƯỢNG.
#          reset_index -> trả lời câu hỏi về SỐ LƯỢNG sample.
fig, ax = plt.subplots()
ax.plot(set_df["acc_y"].reset_index(drop=True))
ax.set_ylabel("acc_y")
ax.set_xlabel("Số thứ tự sample")
ax.set_title("Set 1 - acc_y theo số thứ tự sample")
plt.show()


# --------------------------------------------------------------
# 6. Vẽ tất cả các Exercise
# --------------------------------------------------------------
# Một set đơn lẻ (ví dụ set 1) chỉ cho biết pattern của MỘT exercise.
# Để biết các exercise khác nhau (bench, squat, dead, ohp, row, rest)
# có hình dạng sóng khác nhau không, cần lấy từng label và vẽ riêng.
labels = df["label"].unique()
print("Các exercise (label) có trong dataset:", labels)

for label in labels:
    subset = df[df["label"] == label]

    fig, ax = plt.subplots()
    # Chỉ lấy 100 sample ĐẦU TIÊN để nhìn rõ pattern của vài rep đầu.
    # Nếu vẽ cả set (có thể vài trăm sample), các đỉnh dao động sẽ bị
    # "nén" lại quá sát nhau, nhìn không rõ hình dạng sóng.
    ax.plot(subset["acc_y"].reset_index(drop=True)[:100], label=label)
    ax.set_ylabel("acc_y")
    ax.set_xlabel("Số thứ tự sample")
    ax.set_title(f"Exercise: {label}")
    plt.legend()
    plt.show()
    # Insight quan trọng cho ML: mỗi exercise tạo ra một "chữ ký"
    # dao động khác nhau trên acc_y, ví dụ:
    #   Bench press     -> đỉnh nhọn, đều
    #   Overhead press  -> 2 đỉnh nhọn liên tiếp
    #   Squat / Deadlift/ Row -> hình dạng sóng khác nhau
    # Đây chính là tín hiệu (feature) mà model classification sẽ học.
    #
    # Ý nghĩa business: đây là bước "feasibility check" trước khi đầu
    # tư công sức build model. Nếu các exercise KHÔNG có pattern khác
    # biệt rõ ràng (nhìn bằng mắt còn không phân biệt được), thì việc
    # xây một app "tự động nhận diện bài tập" gần như chắc chắn sẽ
    # thất bại hoặc cho độ chính xác thấp -> nên phát hiện sớm ở bước
    # visualization này, tránh lãng phí thời gian train model trên dữ
    # liệu không có tín hiệu phân biệt.


# --------------------------------------------------------------
# 7. Tinh chỉnh cấu hình Plot (Plot Settings)
# --------------------------------------------------------------
# Đã xử lý ngay ở đầu file bằng "import plot_settings" (áp dụng
# mpl.style.use, rcParams["figure.figsize"], bảng màu theo cycler...).
# Mọi figure được tạo SAU dòng import đó sẽ tự động dùng chung các
# cấu hình này -> không cần lặp lại figsize/style ở từng plt.subplots().


# --------------------------------------------------------------
# 8. So sánh set Medium vs Heavy
# --------------------------------------------------------------
# Mục tiêu: với CÙNG một exercise + CÙNG một participant, set "heavy"
# (tập nặng) và set "medium" (tập nhẹ hơn) có pattern vận động khác
# nhau không? (trực giác: tập nặng thường chuyển động chậm, kiểm soát
# hơn tập nhẹ).
category_df = df.query("label == 'squat' and participant == 'A'").reset_index(
    drop=True
)

fig, ax = plt.subplots()
# groupby("category") tách dữ liệu thành 2 nhóm (heavy/medium), rồi
# .plot() vẽ MỖI NHÓM thành 1 đường riêng nhưng trên CÙNG một axes
# (ax) -> đặt 2 đường cạnh nhau để so sánh trực tiếp bằng mắt.
#
#   acc_y │   heavy  ──╮  ╭──
#         │  medium ───╯╲╱
#         └──────────────────── Số thứ tự sample
category_df.groupby("category")["acc_y"].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("Số thứ tự sample")
ax.set_title("Squat (A) - So sánh Heavy vs Medium")
ax.legend()
plt.show()

# Ý nghĩa business: nếu heavy/medium thực sự có pattern khác biệt rõ
# (ví dụ biên độ dao động, tốc độ chuyển động khác nhau), đây là cơ
# sở để sau này xây thêm tính năng "ước lượng mức độ gắng sức" (RPE)
# hoặc "phân loại cường độ set" tự động từ sensor, chứ không chỉ dừng
# ở việc nhận diện TÊN bài tập. Đây là một hướng mở rộng sản phẩm
# (product feature), không phải chỉ là bài tập kỹ thuật thống kê.


# --------------------------------------------------------------
# 9. So sánh giữa các Participant
# --------------------------------------------------------------
# Mục tiêu: model cần nhận diện exercise cho NHIỀU người, không chỉ
# "học vẹt" cách di chuyển của một participant duy nhất. Vì vậy cần
# kiểm tra: cùng 1 exercise, các participant khác nhau có pattern
# tương tự nhau không?
participant_df = (
    df.query("label == 'bench'")
    .sort_values("participant")  # gom các dòng cùng participant lại gần nhau,
    # thay vì để lẫn lộn A-C-B-A-E-C... rất khó đọc khi groupby/plot.
    .reset_index(drop=True)  # mỗi participant tập ở THỜI ĐIỂM khác nhau
    # (ngày/giờ khác nhau) -> nếu giữ timestamp thật, các đường sẽ nằm
    # rải rác không chồng lên nhau được. reset_index đưa tất cả về
    # chung một mốc "sample number" để so sánh pattern công bằng.
)

fig, ax = plt.subplots()
participant_df.groupby("participant")["acc_y"].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("Số thứ tự sample")
ax.set_title("Bench Press - So sánh giữa các Participant")
ax.legend()
plt.show()

# Ý nghĩa business: đây chính là câu hỏi "generalization" - sản
# phẩm thực tế sẽ được dùng bởi người dùng MỚI, chưa từng xuất hiện
# trong dữ liệu training. Nếu pattern giữa 5 participant (A-E) khác
# nhau quá nhiều (ví dụ do chiều cao, cách đeo thiết bị, kỹ thuật tập
# khác nhau), model train trên 5 người này có thể KHÔNG hoạt động tốt
# với khách hàng thứ 6. Đây là rủi ro sản phẩm cần đánh giá SỚM, trước
# khi quyết định đầu tư vào việc thu thập thêm data hoặc đổi feature.


# --------------------------------------------------------------
# 10 & 11. Vẽ cả 3 trục X/Y/Z của một exercise + participant
# --------------------------------------------------------------
# Accelerometer có 3 trục đo chuyển động theo 3 hướng không gian:
#   X -> trái/phải      Y -> lên/xuống      Z -> trước/sau
# Muốn thấy TOÀN BỘ chuyển động (không chỉ riêng trục Y), cần vẽ
# cùng lúc cả acc_x, acc_y, acc_z.
label = "squat"
participant = "A"
# f-string cho phép chèn biến trực tiếp vào chuỗi query, giúp dễ đổi
# label/participant mà không phải viết lại chuỗi thủ công mỗi lần.
all_axes_df = df.query(
    f"label == '{label}' and participant == '{participant}'"
).reset_index(drop=True)

fig, ax = plt.subplots()
# QUAN TRỌNG: [["acc_x","acc_y","acc_z"]] dùng NGOẶC KÉP (2 cặp []) để
# lấy ra một DataFrame gồm 3 cột, khác với df["acc_x"] (1 cặp []) chỉ
# lấy ra 1 Series (1 cột). Khi .plot() nhận DataFrame nhiều cột, nó
# tự vẽ MỖI CỘT thành 1 đường riêng trên cùng 1 axes:
#
#   acc_x ─────╮╭──────
#   acc_y ─────╯╰──╮╭──  ← 3 đường, 3 màu khác nhau, auto vẽ bởi .plot()
#   acc_z ─────────╯╰──
all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("Accelerometer")
ax.set_xlabel("Số thứ tự sample")
ax.set_title(f"{label} ({participant}) - Accelerometer X/Y/Z".title())
plt.show()

# Ý nghĩa business: nhìn riêng từng trục giúp trả lời câu hỏi
# "trục nào MANG THÔNG TIN QUAN TRỌNG NHẤT để phân biệt exercise?".
# Nếu chỉ 1-2 trục (ví dụ chỉ acc_y) đã đủ phân biệt rõ các bài tập,
# đây là cơ sở để về sau đơn giản hóa feature engineering (dùng ít
# trục hơn vẫn đủ chính xác), hoặc ngược lại là lý do cần giữ đủ cả
# 3 trục + cả gyroscope nếu 1 trục không đủ thông tin.


# --------------------------------------------------------------
# 12. Lặp qua TẤT CẢ tổ hợp exercise x participant (Accelerometer)
# --------------------------------------------------------------
# Thay vì đổi tay label/participant rồi chạy lại từng lần (6 exercise
# x 5 participant = 30 lần làm thủ công), ta dùng NESTED LOOP (lặp
# lồng nhau) để tự động tạo hết mọi figure cần thiết trong 1 lần chạy:
#
#   for mỗi exercise (bench, squat, dead, ohp, row, rest):
#       for mỗi participant (A, B, C, D, E):
#           -> vẽ 1 figure cho tổ hợp (exercise, participant) này
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axes_df = df.query(
            f"label == '{label}' and participant == '{participant}'"
        ).reset_index(drop=True)

        # Không phải participant nào cũng tập đủ mọi exercise (ví dụ
        # participant B có thể không có bài "dead"). Nếu tổ hợp này
        # không có dữ liệu (len == 0) thì bỏ qua, tránh tạo ra một
        # figure trống rỗng vô nghĩa.
        if len(all_axes_df) > 0:
            fig, ax = plt.subplots()
            all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("Accelerometer")
            ax.set_xlabel("Số thứ tự sample")
            ax.set_title(f"{label} ({participant})".title())
            plt.show()

# Ý nghĩa business: đây là bước "data QA" (kiểm tra chất lượng dữ
# liệu) có hệ thống trên TOÀN BỘ dataset, trước khi đưa vào train
# model. Việc nhìn qua từng combination giúp phát hiện sớm các vấn
# đề như: file bị lỗi, sensor bị nhiễu bất thường, hoặc một
# participant/exercise có quá ít dữ liệu để model học tốt — những rủi
# ro này nếu không phát hiện ở đây sẽ chỉ lộ ra muộn hơn (khi model
# cho kết quả kém), gây tốn thời gian debug ngược.


# --------------------------------------------------------------
# 13. Accelerometer -> Gyroscope (dùng lại y nguyên logic ở trên)
# --------------------------------------------------------------
# Gyroscope đo chuyển động XOAY (độ/giây) thay vì GIA TỐC (G-force)
# như accelerometer, nhưng cách vẽ hoàn toàn giống nhau -> chỉ cần
# đổi 3 cột acc_x/y/z thành gyr_x/y/z, mọi logic còn lại y nguyên.
for label in labels:
    for participant in participants:
        all_axes_df = df.query(
            f"label == '{label}' and participant == '{participant}'"
        ).reset_index(drop=True)

        if len(all_axes_df) > 0:
            fig, ax = plt.subplots()
            all_axes_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("Gyroscope")
            ax.set_xlabel("Số thứ tự sample")
            ax.set_title(f"{label} ({participant})".title())
            plt.show()


# --------------------------------------------------------------
# 14, 15, 16. Gộp Accelerometer + Gyroscope và export ra file ảnh
# --------------------------------------------------------------
# Mục tiêu cuối: với MỖI tổ hợp (exercise, participant), tạo MỘT
# figure duy nhất gồm 2 subplot xếp DỌC (accelerometer ở trên,
# gyroscope ở dưới) để xem đồng thời cả 2 loại chuyển động, rồi lưu
# thành file .png trong reports/figures để dùng cho báo cáo sau này.
#
#   ┌───────────────────────────────┐
#   │   Accelerometer X / Y / Z     │  ← ax[0]
#   ├───────────────────────────────┤
#   │   Gyroscope X / Y / Z         │  ← ax[1]
#   └───────────────────────────────┘
#              Số thứ tự sample (trục X dùng chung)
os.makedirs(FIGURES_DIR, exist_ok=True)

for label in labels:
    for participant in participants:
        combined_df = df.query(
            f"label == '{label}' and participant == '{participant}'"
        ).reset_index(drop=True)

        if len(combined_df) == 0:
            continue

        # nrows=2    -> tạo 2 subplot xếp DỌC, truy cập bằng ax[0], ax[1]
        #               (Python đánh số từ 0: ax[0] là ô TRÊN, ax[1] là ô DƯỚI).
        # sharex=True-> 2 subplot dùng CHUNG 1 trục X -> không cần vẽ
        #               lại label trục X ở cả 2 ô, đồng thời giúp so sánh
        #               trực quan: cùng 1 sample number thì nhìn thẳng lên
        #               xuống là thấy acc và gyr tại đúng thời điểm đó.
        # figsize=(20,10) -> figure to hơn mặc định để nhìn rõ 2 subplot.
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

        combined_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
        combined_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

        # Đặt legend (chú giải màu X/Y/Z) nằm NGANG, PHÍA TRÊN mỗi
        # subplot, đẩy ra ngoài vùng vẽ (bbox_to_anchor) để không che
        # mất đường dữ liệu bên dưới. ncol=3 -> X, Y, Z nằm trên 1 dòng
        # thay vì xếp dọc choán nhiều diện tích.
        legend_style = dict(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=True,  # góc legend bo tròn cho đẹp mắt
            shadow=True,  # thêm đổ bóng nhẹ, dễ phân biệt với nền
        )
        ax[0].legend(**legend_style)
        ax[1].legend(**legend_style)

        ax[0].set_ylabel("Accelerometer")
        ax[1].set_ylabel("Gyroscope")
        ax[1].set_xlabel("Số thứ tự sample")
        # Vì sharex=True nên chỉ cần set_xlabel ở subplot DƯỚI CÙNG,
        # subplot trên không cần nhãn trục X trùng lặp.

        fig.suptitle(f"{label} ({participant})".title())

        # Tên file động theo exercise + participant, ví dụ:
        # "Squat (A).png", "Bench (B).png" -> tên file tự mô tả nội
        # dung, dễ tra cứu lại khi cần mà không phải mở từng ảnh.
        filename = f"{label.title()} ({participant}).png"
        filepath = os.path.join(FIGURES_DIR, filename)
        # dpi=100: đủ độ nét để đưa vào report/slide thuyết trình,
        # trong khi dung lượng file vẫn nhẹ (không cần dpi quá cao).
        fig.savefig(filepath, dpi=100)
        # Đóng figure ngay sau khi lưu (plt.close) để giải phóng
        # memory - nếu không, sau 30 lần loop sẽ có 30 figure cùng
        # tồn tại trong RAM dù đã lưu xong, gây tốn bộ nhớ không cần thiết.
        plt.close(fig)

print(f"Đã export toàn bộ figures vào: {FIGURES_DIR}")

# Ý nghĩa business: các file .png này là "tài sản chia sẻ" - người
# không biết code (product owner, personal trainer, khách hàng demo)
# vẫn xem được trực tiếp mà không cần mở Python/notebook. Đây cũng là
# tài liệu chứng minh cho quyết định kỹ thuật "kết hợp accelerometer +
# gyroscope" (thay vì chỉ dùng 1 sensor) là có cơ sở quan sát thực tế,
# phục vụ việc trình bày/báo cáo trước khi bước vào giai đoạn modeling.

import pandas as pd

# 输入文件
input_file = r"C:\Users\Administrator\Desktop\learn by yrh\2026.3\Tianrun_code\分组检测报告\src_new620260316110715.xlsx"

# 输出文件
output_file = r"C:\Users\Administrator\Desktop\learn by yrh\2026.3\Tianrun_code\分组检测报告\merged_sorted.xlsx"

# 读取Excel
xls = pd.ExcelFile(input_file)

dfs = []

for sheet in xls.sheet_names:
    # 只处理包含“详情”且不包含“汇总”的sheet
    if "详情" in sheet and "汇总" not in sheet:
        df = pd.read_excel(input_file, sheet_name=sheet)

        # 添加来源sheet
        df["来源Sheet"] = sheet

        dfs.append(df)

# 合并
merged_df = pd.concat(dfs, ignore_index=True)

# 按文件路径排序
if "文件路径" in merged_df.columns:
    merged_df = merged_df.sort_values(by="文件路径")

# 保存
merged_df.to_excel(output_file, index=False, sheet_name="合并结果")

print("处理完成，输出文件：", output_file)

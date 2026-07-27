# Contributing

感谢你改进这个工具集。提交代码前请遵循以下约定：

1. 文件名、函数名和变量名使用小写 `snake_case`。
2. 新工具放入 `tools/` 下最接近的领域目录，并在 `README.md` 的工具目录中补充用途、依赖和运行方式。
3. 文件路径必须通过命令行参数或函数参数传入，不要提交本机绝对路径、数据集、模型或运行结果。
4. 会覆盖或删除文件的操作必须提供明确提示，建议默认使用预览模式。
5. 提交前运行 `python scripts/check_repository.py`。

GDAL/OGR 和 ArcPy 的安装方式依赖操作系统与 GIS 环境，不应写入普通 `pip` 安装流程。需要 GDAL 的工具建议使用 Conda：

```bash
conda install -c conda-forge gdal
pip install -r requirements.txt
```

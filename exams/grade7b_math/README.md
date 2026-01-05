# 七年级下册数学试卷（Word + PDF）

本目录会生成一份**原创**七年级下册数学试卷，包含示意图、答案与解析，并同时输出：

- `exam.docx`
- `exam.pdf`

## 运行方式（Windows / PowerShell）

在仓库根目录执行：

```powershell
Set-Location D:\Project_dev
.\.venv\Scripts\python -m pip install -r .\vla-interview-handbook\exams\grade7b_math\requirements.txt -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
.\.venv\Scripts\python .\vla-interview-handbook\exams\grade7b_math\generate_exam.py
```

生成物在：

- `vla-interview-handbook/exams/grade7b_math/exam*.docx`
- `vla-interview-handbook/exams/grade7b_math/exam*.pdf`

说明：如果你正在用 Word/WPS 打开 `exam.docx`，Windows 会锁定该文件，脚本将自动改为输出 `exam_v2.docx/.pdf`、`exam_v3.docx/.pdf` 等版本号文件，避免覆盖失败。

## 文件说明

- `questions.json`：题库（题干、分值、答案、解析、配图引用）——“单一真源”。
- `assets/`：脚本自动生成的配图 PNG。
- `assets_builder.py`：配图生成脚本。
- `generate_exam.py`：读取题库并输出 Word/PDF。



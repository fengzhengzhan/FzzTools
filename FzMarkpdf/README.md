# FzMarkpdf

Here is an example of using the FzMarkpdf tool.

```
FzMarkpdf.exe -n "MVP Detecting Vulnerabilities using Patch-Enhanced Vulnerability Signatures.pdf"
```

A file named "Mark_MVP Detect Vulnerabilities Using Patch Enhanced Vulnerability Signatures.pdf" will be generated.

 

**简体中文 (^_^)#**

下面是一个使用FzMarkpdf工具的例子：

```
FzMarkpdf.exe -n "MVP Detecting Vulnerabilities using Patch-Enhanced Vulnerability Signatures.pdf"
```

然后将产生一个名为 "Mark_MVP 使用补丁增强型漏洞签名检测漏洞.pdf "的文件。

```
optional arguments:
  -h, --help            show this help message and exit
  --pdfname PDFNAME, -n PDFNAME
                        pdf文件名称 || The name of pdf.
  --pdfpath PDFPATH, -pp PDFPATH
                        pdf文件路径 || The path of pdf.
  --imagespath IMAGESPATH, -p IMAGESPATH
                        笔迹拍摄图片路径 || The path of images that are marked.
  --scale SCALE, -s SCALE
                        打印页面的缩放比例(100缩放 等于 1.0) || The scale of the printed page(100zoom equals 1.0).
  --correction CORRECTION, -c CORRECTION
                        页面缩放偏移修正([scale_x,y,offset_x,y]) || Page zoom offset corrected([scale_x,y,offset_x,y]).
  --minbox MINBOX, -m MINBOX
                        将小于此数值的点视为噪点 || Consider points smaller than this value as noise.
```


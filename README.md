# Zhenggang Tang personal website

双击 `index.html` 即可在浏览器打开，图片和脚本均使用相对路径，无需联网加载字体或安装依赖。

本地预览：在本目录运行 `python -m http.server 8765 --bind 127.0.0.1`，打开 http://localhost:8765/ 。

## 文件
- `index.html`：主页内容，已移除 About 导航和页面。
- `styles.css`：响应式布局和颜色。
- `app.js`：论文筛选、图片放大和键盘交互。
- `assets/`：个人照片、论文 teaser 与现有公开 CV。
- `papers.json`：论文条目结构化参考；直接修改它不会自动修改 HTML。
- `assets/teaser-sources.json`：论文原图页码与裁切坐标，坐标单位为 PDF point。

## 内容来源
个人介绍与 Luma AI 状态来自 https://recordmp3.github.io/ 。
论文、经历与教育信息综合原主页和 `tangzhenggang_CV_26.9.docx`，日期冲突处采用简历。
ICML 2026 和 CVPR 2025 Workshop 状态来自简历。
每幅 teaser 来自对应论文，标题、作者和论文链接见 `papers.json`。
CV 下载暂时沿用主页现有公开 PDF；本地修改版 Word 位于父目录。

尚未对线上仓库进行任何修改。`.nojekyll` 使这组静态文件可以在确认后作为 GitHub Pages 页面使用。

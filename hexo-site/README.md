# Hexo 站点（Notes）

源码在 `hexo-site/`，根目录下的 `*.md` 为原始笔记副本已同步到 `source/_posts/`。

## 本地预览

```bash
cd hexo-site
npm install
npx hexo clean && npx hexo generate
npx hexo server
```

浏览器打开提示的地址（默认 `http://localhost:4000/notes/`）。

## GitHub Pages 与 Actions

推送 **`master`** 分支后，[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) 会执行 `npm ci`、`hexo generate`，并将 `public/` 部署到 **`gh-pages`** 分支。

### 仓库里需要设置的一次项

1. **Settings → Actions → General → Workflow permissions**  
   选择 **Read and write permissions**，保存。（否则 `peaceiris/actions-gh-pages` 无法推送 `gh-pages`。）

2. **Settings → Pages**  
   - Source：**Deploy from a branch**  
   - Branch：**gh-pages** / **/(root)**  
   保存后站点地址：**https://llx-08.github.io/notes/**

若 GitHub 默认分支是 `main` 而不是 `master`，请把 workflow 里 `on.push.branches` 改为 `main`。

## 图片

`KL 散度` 等文章中的图片路径为 `/notes/imgs/...`，请将对应文件放到 `source/imgs/` 后再构建。

# Hexo 站点（Notes）

使用主题 **[hexo-theme-icarus](https://github.com/ppoffice/hexo-theme-icarus)**（npm 包 `hexo-theme-icarus`）。主题配置在 **`_config.icarus.yml`**；升级主题后若配置版本过旧，可按终端提示迁移或对照[官方文档](https://ppoffice.github.io/hexo-theme-icarus/)。

评论（Disqus 等）、分享（ShareThis 等）、打赏（PayPal 等）均为可选：未在 `_config.icarus.yml` 中配置时，默认已关闭对应区块，**不必**向任何人提供 shortname、install_url 等；若以后需要，再按官方文档填写即可。

数学公式仍由根目录 **`_config.yml`** 中的 **`hexo-filter-mathjax`**（服务端）渲染；Icarus 自带 `plugins.mathjax` / `katex` 保持关闭，避免重复加载。

源码在 `hexo-site/`，仓库根目录下的 `*.md` 为原始笔记副本，已同步到 `source/_posts/`；根目录 **`imgs/`** 会通过脚本同步到 **`source/imgs/`**（`npm run sync-imgs` 或 `npm run build` 会自动执行）。

**提交前自动同步**：在仓库根目录执行一次 **`./scripts/install-git-hooks.sh`**，之后每次 **`git commit`** 会先运行上述两个同步脚本，并把 `source/_posts/`、`source/imgs/` 的变更自动 **git add** 进本次提交。若某次不想跑钩子：**`git commit --no-verify`**。

**首页摘要**：未写 `<!-- more -->` 时，脚本会按纯文本约 **420 字**截断（`_config.yml` 里 `auto_excerpt_length`）。单篇若要自定截断位置，在正文插入一行 `<!-- more -->` 即可。

**侧栏头像**：在 `_config.icarus.yml` 的 `widgets` → `profile` → `avatar` 填图片 **URL**，或把图放到 `source/imgs/` 后写 **`/imgs/文件名`**（与文章图片路径规则相同）。

## 本地预览

```bash
cd hexo-site
npm install
npx hexo clean && npx hexo generate
npx hexo server
```

浏览器打开提示的地址（默认 `http://localhost:4000/notes/`）。

## GitHub Pages 与 Actions

向 **`master`** 或 **`main`** 推送后，[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) 会执行 `npm ci`、`hexo generate`，并将 `public/` 部署到 **`gh-pages`** 分支。

也可在 GitHub 上 **手动跑一次**：**Actions** → 左侧 **Deploy Hexo to GitHub Pages** → **Run workflow**（选分支后运行）。

### 在 GitHub 上需要配置的一次项

1. **打开 Actions 写权限**（必须）  
   进入仓库 **Settings** → **Actions** → **General**，滚到 **Workflow permissions**，勾选 **Read and write permissions**，点 **Save**。  
   否则部署步骤无法推送 `gh-pages`，站点会 404。

2. **指定 Pages 发布分支**（必须）  
   **Settings** → **Pages** → **Build and deployment**：  
   - **Source**：**Deploy from a branch**  
   - **Branch**：选 **`gh-pages`**，文件夹 **`/(root)`** → **Save**  
   站点地址：**https://llx-08.github.io/notes/**

3. **把含 workflow 的代码推到 GitHub**  
   将本仓库推送到 `git@github.com:llx-08/notes.git` 后，打开 **Actions** 确认 **Deploy Hexo to GitHub Pages** 有成功运行（绿色勾）。首次成功后才会出现 **`gh-pages`** 分支。

### 日常更新

本地改文章或配置后：`git push` 到 **`master` 或 `main`**（与你远程默认分支一致即可），**Actions 会自动部署**，一般不必再点 **Run workflow**。只有想「不推送代码、强制再发一版静态站」时，才用手动运行。

## 图片

文章里图片请写 **`![](/imgs/xxx.png)`**（不要用 `/notes/imgs/`，否则生成结果会变成 `/notes/notes/imgs/` 导致 404）。把 **`xxx.png` 放在 `source/imgs/`** 并提交。线上实际地址仍是 `https://llx-08.github.io/notes/imgs/xxx.png`。

若只在仓库根目录的 `imgs/` 里有图、没有拷到 `hexo-site/source/imgs/`，站点上会 **404**。根目录 Markdown 里可继续用相对路径 `imgs/xxx.png` 方便本地/GitHub 预览；同步到 `_posts` 时请改成上面的 `/imgs/` 写法。

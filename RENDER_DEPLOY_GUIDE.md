# 🚀 Render部署指南 - CKD风险计算器（Flask版本）

## 📋 准备工作

您的文件夹已经准备好，包含：
- ✅ app.py (Flask应用)
- ✅ requirements.txt (依赖包)
- ✅ Procfile (启动配置)
- ✅ templates/index.html (您的自定义HTML界面)
- ✅ models/ (模型文件)
- ✅ scaler.pkl, shap_background_*.csv

---

## 🌐 Render部署步骤（免费）

### 1️⃣ 注册Render账号
1. 访问 https://render.com
2. 点击 **Sign Up** 注册（可用GitHub账号登录）
3. 验证邮箱

### 2️⃣ 上传代码到GitHub

**选项A：使用命令行（推荐）**

```bash
# 1. 在GitHub创建新仓库（ckd-risk-calculator）
# 2. 返回终端运行：

cd /Users/gubaichuan/Desktop/心肾综合症1019/0-3期/重新算1110/重重新1115/1118重新0-3期-机器学习/【修正图像后】回顾文章方法学/deployment_package

# 初始化git（如果还没有）
git init
git add .
git commit -m "Initial commit - CKD Risk Calculator"

# 连接到GitHub仓库（替换YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/ckd-risk-calculator.git
git branch -M main
git push -u origin main
```

**选项B：使用GitHub网页**
1. 访问 https://github.com/new 创建新仓库
2. 上传文件夹中的所有文件

### 3️⃣ 在Render创建Web Service

1. 登录Render后，点击 **New +** → **Web Service**

2. **Connect Repository**：
   - 点击 **Connect GitHub**
   - 授权Render访问
   - 选择 `ckd-risk-calculator` 仓库

3. **配置Service**：
   ```
   Name: ckd-risk-calculator
   Region: Oregon (US West) 或离您最近的
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

4. **选择免费套餐**：
   - Instance Type: **Free**
   - 滚动到底部点击 **Create Web Service**

### 4️⃣ 等待部署

- 首次部署需要5-10分钟
- 可以查看实时日志了解进度
- 成功后会显示绿色 ✓ 和URL

---

## 🔗 访问您的应用

部署成功后，您会获得一个URL：
```
https://ckd-risk-calculator.onrender.com
```

**分享给任何人使用！**

---

## ⚙️ 高级配置（可选）

### 自定义域名
1. 在Render Dashboard → Settings → Custom Domain
2. 添加您的域名（需要在域名提供商处配置DNS）

### 环境变量
如果需要配置环境变量：
- Settings → Environment
- 添加键值对

### 升级到付费版（如果免费版不够用）
- Starter Plan: $7/月
- 不会休眠，内存更大，性能更好

---

## 🐛 常见问题

### Q: 部署失败，显示内存不足
**A:** 230MB的模型文件比较大，可能需要：
- 等待几分钟重试
- 或者升级到Starter套餐

### Q: 首次访问很慢
**A:** 免费版会休眠，首次访问需要10-30秒唤醒，这是正常的

### Q: 想修改代码
**A:** 修改本地文件后：
```bash
git add .
git commit -m "Update"
git push
```
Render会自动重新部署

---

## 📝 Git命令速查

```bash
# 查看状态
git status

# 添加所有更改
git add .

# 提交更改
git commit -m "描述您的更改"

# 推送到GitHub
git push

# 如果需要从GitHub拉取最新代码
git pull
```

---

## 💡 提示

1. **保持仓库私有**（如果包含敏感数据）
2. **定期访问应用**可以避免休眠（或升级到付费版）
3. **监控日志**：Dashboard → Logs 查看运行情况

---

需要帮助？随时联系！🙌

# Windows 部署指南

> 写给你 Windows 电脑上的 Claude Code。按下面步骤一步步来。

---

## 1. 环境准备

### 安装 Python 3.12+

```powershell
# PowerShell（管理员）: 先装 Python，勾选 "Add to PATH"
# 验证
python --version   # 应输出 3.12.x 或 3.13.x
```

### 安装 PostgreSQL

下载安装 [PostgreSQL 16+](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)，记住你设的 `postgres` 用户密码。

安装后确保 PostgreSQL 服务在运行：
```powershell
Get-Service postgresql*   # 状态应为 Running
```

### 安装 Git

```powershell
git --version
```

---

## 2. 克隆项目 & 创建虚拟环境

```powershell
cd D:\   # 或你放项目的地方
git clone https://github.com/Neownne/Quant.git
cd Quant

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows 激活命令（注意是反斜杠）

# 安装依赖
pip install -r requirements.txt
```

---

## 3. 配置 .env

```powershell
copy .env.example .env
```

编辑 `.env`，填入你的实际配置：

```ini
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=你装PG时设的密码
DB_NAME=quant

# 邮件推送（QQ邮箱示例）
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USER=你的QQ邮箱@qq.com
SMTP_PASS=QQ邮箱的授权码（不是密码！）
EMAIL_FROM=你的QQ邮箱@qq.com
EMAIL_TO=接收报告的邮箱@qq.com
```

> QQ 邮箱授权码获取：QQ邮箱 → 设置 → 账户 → POP3/SMTP服务 → 开启 → 生成授权码

---

## 4. 初始化数据库

```powershell
.venv\Scripts\python -c "from data.db import init_db; init_db()"
```

这会创建所有需要的表。如果报数据库连接错，检查 `.env` 里的 `DB_PASSWORD`。

---

## 5. 同步历史数据（一次性）

```powershell
# 同步股票列表 + 历史日线（首次需要较长时间，1-2小时）
.venv\Scripts\python data\sync.py --start 20200101
```

也可以分步跑（如果一步跑太久容易断）：
```powershell
.venv\Scripts\python data\sync.py --mode stock           # 先拉股票列表
.venv\Scripts\python data\sync.py --mode stock-daily --start 20200101  # 再拉日线
.venv\Scripts\python data\sync.py --mode index           # 最后拉指数
```

---

## 6. 试运行验证

```powershell
# 跳过同步，直接扫描信号（如果数据库已有数据）
.venv\Scripts\python scripts\run_daily_signals.py --dry-run --now --no-sync
```

如果正常输出三池信号（涨停池/妖股池/牛股池），说明一切 OK。

---

## 7. 日常使用

### 收盘后自动跑

```powershell
# 完整流程：同步 → 信号 → 邮件
.venv\Scripts\python scripts\run_daily_signals.py --send-email
```

脚本会自动等到 15:00 后才执行。如果想立即跑（盘中调试）：
```powershell
.venv\Scripts\python scripts\run_daily_signals.py --now --send-email
```

### 其他常用命令

```powershell
# 只看信号不发邮件
.venv\Scripts\python scripts\run_daily_signals.py --now

# 指定日期回测
.venv\Scripts\python scripts\run_daily_signals.py --date 2026-06-13 --now

# 排除创业板+科创板
.venv\Scripts\python scripts\run_daily_signals.py --now --exclude-gem-star

# 武器库面板（实时行情热力图）
.venv\Scripts\python scripts\run_arsenal.py

# 妖股回测
.venv\Scripts\python scripts\bt_yaogu.py --start 2020-01-01 --top-n 5

# 小市值反转回测
.venv\Scripts\python scripts\bt_small_cap.py --start 2020-01-01 --top-n 10

# 因子验证
.venv\Scripts\python scripts\validate_factors.py --start 2025-01-01

# Web 面板（浏览器看回测结果）
.venv\Scripts\python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → 打开 http://localhost:8899/backtest
```

### Windows 定时任务（替代 Mac crontab）

打开 PowerShell 管理员：

```powershell
# 每个交易日 15:30 自动跑（周一~周五）
$action = New-ScheduledTaskAction -Execute "D:\Quant\.venv\Scripts\python.exe" `
    -Argument "D:\Quant\scripts\run_daily_signals.py --send-email" `
    -WorkingDirectory "D:\Quant"

$trigger = New-ScheduledTaskTrigger -Daily -At 15:30

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Limited

Register-ScheduledTask -TaskName "QuantDailySignal" `
    -Action $action -Trigger $trigger -Principal $principal `
    -Description "量化每日信号扫描"
```

> 把路径里的 `D:\Quant` 替换成你的实际项目路径。

---

## 8. Claude Code 快捷指令

在 Claude Code 里你可以这样说：

> "跑一下今日信号"  
> "昨天数据质量怎么样"  
> "帮我回测妖股规则 2020 年到今天"  
> "看下最近的因子 IC 变化"  
> "数据同步出错了，帮我排查"

Claude Code 会读取 CLAUDE.md 中的铁律和速查命令，自动找到正确的脚本来跑。

---

## 9. 常见问题

| 问题 | 解决 |
|------|------|
| `psycopg2` 装不上 | 改用 `pip install psycopg2-binary` |
| 数据库连不上 | 检查 PG 服务是否运行、`.env` 密码对不对 |
| akshare 拉数据报错 | 网络问题，换个时间段试 |
| `ModuleNotFoundError` | 确认 `.venv` 已激活，或检查 `sys.path` |
| 邮件发不出去 | QQ 邮箱需用**授权码**而非密码 |

---

## 10. 项目结构速查

```
Quant/
├── scripts/
│   ├── run_daily_signals.py    ★ 每日信号（主入口）
│   ├── bt_yaogu.py             妖股回测
│   ├── bt_small_cap.py         小市值反转回测
│   ├── run_arsenal.py          武器库面板
│   ├── validate_factors.py     因子IC验证
│   └── screen_bull.py          牛股筛选器
├── factors/                    因子库（123个因子）
├── strategies/limit_up/        涨停策略（执行引擎+净值计算）
├── data/                       数据库+数据同步
├── config/settings.py          全局配置
├── lab/                        实验室（变体回测框架）
├── web/                        FastAPI Web面板
├── .env                        你的配置（不提交git）
├── CLAUDE.md                   给Claude Code的铁律和速查
└── requirements.txt            Python依赖
```

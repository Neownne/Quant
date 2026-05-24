# PostgreSQL 使用指南

> 适用环境：macOS + Homebrew 安装的 PostgreSQL

---

## 服务管理

```bash
# 启动 / 停止 / 重启
brew services start postgresql@18
brew services stop postgresql@18
brew services restart postgresql@18

# 查看服务状态
brew services list | grep postgres
```

---

## 连接数据库

```bash
# 最简连接（用当前系统用户名连本地默认数据库）
psql

# 指定用户 + 数据库
psql -U chenwan -d quant

# 指定主机 + 端口
psql -h localhost -p 5432 -U chenwan -d quant
```

---

## psql 内部命令（连进去后敲的）

### 查看信息

```
\l                列出所有数据库
\c quant          切换到 quant 数据库
\dt               列出当前库的所有表
\d stock_daily    查看 stock_daily 表结构
\du               列出所有用户/角色
\dn               列出所有 schema
```

### 查询

```sql
-- 查前10行
SELECT * FROM stock_basic LIMIT 10;

-- 计数
SELECT COUNT(*) FROM stock_daily;

-- 最近交易日
SELECT MAX(trade_date) FROM stock_daily;

-- 某只股票最近100天
SELECT * FROM stock_daily
WHERE code = '000001'
ORDER BY trade_date DESC
LIMIT 100;
```

### 退出

```
\q
```

---

## 常用 DDL（表结构操作）

```sql
-- 删表（危险！）
DROP TABLE IF EXISTS stock_daily;

-- 重建表（清空数据）
TRUNCATE TABLE stock_daily;

-- 加列
ALTER TABLE stock_basic ADD COLUMN sector VARCHAR(50);

-- 删列
ALTER TABLE stock_basic DROP COLUMN sector;

-- 加索引（加速查询）
CREATE INDEX idx_stock_daily_code ON stock_daily(code);
CREATE INDEX idx_stock_daily_date ON stock_daily(trade_date);
```

---

## 常用 DML（数据操作）

```sql
-- 删（带条件）
DELETE FROM stock_daily WHERE trade_date < '2020-01-01';

-- 改
UPDATE stock_basic SET industry = '银行' WHERE code = '000001';

-- 查 —— 按日期范围
SELECT * FROM stock_daily
WHERE code = '000001'
  AND trade_date BETWEEN '2024-01-01' AND '2024-12-31';
```

---

## 数据库管理

```sql
-- 创建数据库
CREATE DATABASE mydb;

-- 删除数据库
DROP DATABASE mydb;

-- 创建用户 + 密码
CREATE USER myuser WITH PASSWORD 'mypass';

-- 授权
GRANT ALL PRIVILEGES ON DATABASE quant TO myuser;
```

---

## 备份 & 恢复

```bash
# 导出（备份）
pg_dump -U chenwan quant > quant_backup.sql

# 只导出表结构，不要数据
pg_dump -U chenwan --schema-only quant > quant_schema.sql

# 恢复
psql -U chenwan -d quant < quant_backup.sql
```

---

## 实用查询片段

```sql
-- 查看表大小
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(tablename) DESC;

-- 查看当前连接数
SELECT COUNT(*) FROM pg_stat_activity;

-- 杀掉空闲连接
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle' AND pid <> pg_backend_pid();
```

---

## 与 Python 项目的关系

| 操作 | 在哪做 |
|---|---|
| 建库 `CREATE DATABASE` | 命令行 psql（只需一次） |
| 建表 `CREATE TABLE` | `data/db.py` → `init_db()` 自动完成 |
| 写入数据 | `data/sync.py` 自动完成 |
| 临时查询分析 | 命令行 psql 或 Jupyter Notebook |
| 备份 | 命令行 pg_dump |

---

## Python 操作 PostgreSQL

项目中通过 SQLAlchemy + psycopg2 操作数据库，核心代码在 [data/db.py](../data/db.py)。

### 1. 创建引擎

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://user:password@localhost:5432/quant",
    pool_size=5,           # 连接池大小
    max_overflow=10,       # 超出 pool_size 后最多再创建的连接数
    pool_pre_ping=True,    # 每次使用前检测连接是否存活（防止 stale connection）
    connect_args={"options": "-c timezone=Asia/Shanghai"},
)
```

### 2. 执行原始 SQL

```python
from sqlalchemy import text

# 写操作 —— 用 begin() 自动提交/回滚
with engine.begin() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS foo (id INT PRIMARY KEY)"))

# 读操作 —— 用 connect() 不自动提交
with engine.connect() as conn:
    rows = conn.execute(
        text("SELECT * FROM stock_daily WHERE code = :code"),
        {"code": "000001"},
    ).fetchall()
```

**为什么用 `text()`？** SQLAlchemy 要求显式包装原始 SQL 字符串，防止 SQL 注入时误用。`:code` 是命名参数占位符，实际值通过字典传入。

### 3. DataFrame 读写

```python
import pandas as pd

# 读 —— SQL → DataFrame
df = pd.read_sql("SELECT code, name FROM stock_basic", engine)

# 写 —— DataFrame → 表（会替代表中已有数据）
df.to_sql("stock_basic", engine, if_exists="replace", index=False)
```

### 4. Upsert 模式（项目核心）

项目中使用 **临时表 + ON CONFLICT** 实现 upsert，比逐行 INSERT/UPDATE 快几个数量级：

```python
def upsert_df(df: pd.DataFrame, table: str, engine) -> int:
    """将 DataFrame 按主键 upsert 到表中。"""
    with engine.begin() as conn:
        # Step 1: 数据写入临时表
        tmp = f"_tmp_{table}"
        df.to_sql(tmp, conn, if_exists="replace", index=False)

        # Step 2: 临时表 → 正式表（冲突时更新）
        cols = df.columns.tolist()
        col_names = ", ".join(f'"{c}"' for c in cols)
        excluded = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "code")

        sql = f"""
            INSERT INTO {table} ({col_names})
            SELECT {col_names} FROM {tmp}
            ON CONFLICT (code, trade_date) DO UPDATE SET {excluded};
        """
        result = conn.execute(text(sql))

        # Step 3: 清理临时表
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))

    return result.rowcount
```

**原理**：
- 临时表只存在于当前连接，断开自动消失
- `ON CONFLICT (pk) DO UPDATE` 是 PostgreSQL 的 upsert 语法
- `EXCLUDED` 是冲突行的别名，`EXCLUDED."close"` 表示新数据中的 close 值
- 批量操作比逐条 upsert 快 100 倍以上

### 5. 避免手动拼 SQL 参数

```python
# 错误：拼接字符串，容易 SQL 注入
conn.execute(text(f"SELECT * FROM stock_daily WHERE code = '{symbol}'"))

# 正确：使用参数绑定
conn.execute(
    text("SELECT * FROM stock_daily WHERE code = :code AND trade_date > :dt"),
    {"code": symbol, "dt": start_date},
)
```

### 6. 连接管理最佳实践

```python
# 用完释放连接（归还到连接池）
engine.dispose()

# 不要在循环里反复创建引擎 —— 全局创建一次即可
# 项目中 config/settings.py 初始化一次，其他地方复用
```

### 7. 实用的 pandas 查询片段

```python
# 查某只股票指定日期范围
df = pd.read_sql("""
    SELECT * FROM stock_daily
    WHERE code = '000001'
      AND trade_date BETWEEN '2024-01-01' AND '2024-12-31'
    ORDER BY trade_date
""", engine)

# 统计各市场股票数量
pd.read_sql("SELECT market, COUNT(*) FROM stock_basic GROUP BY market", engine)

# 找最新的交易日期
pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine)

# 时间序列直接设置索引
df = pd.read_sql("SELECT trade_date, close FROM stock_daily WHERE code = '000001'", engine)
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.set_index("trade_date").sort_index()
```

-- 「普通用户」角色（viewer）：只能看分析结果，不能改任何东西，也不能上传、重跑、追问 AI。
-- 标注员（annotator）在普通用户之上多一个标注权限。
-- SQLite 改不了 CHECK，只能把 users 表重建一遍（没有别的表用外键指向它）。
CREATE TABLE users_new (
  email TEXT PRIMARY KEY,        -- 小写
  name TEXT NOT NULL DEFAULT '',
  picture TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT 'annotator' CHECK (role IN ('admin', 'annotator', 'viewer')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'disabled')),
  created_at TEXT NOT NULL,
  last_login_at TEXT
);
INSERT INTO users_new (email, name, picture, role, status, created_at, last_login_at)
  SELECT email, name, picture, role, status, created_at, last_login_at FROM users;
DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

-- 多人标注：用 Google 登录的成员、登录会话、可以领取的标注任务，每条标注记下是谁标的。
-- 成员：status = pending（登录过但还没被邀请）| active | disabled；role = admin | annotator。
CREATE TABLE users (
  email TEXT PRIMARY KEY,        -- 小写
  name TEXT NOT NULL DEFAULT '',
  picture TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT 'annotator' CHECK (role IN ('admin', 'annotator')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'disabled')),
  created_at TEXT NOT NULL,
  last_login_at TEXT
);

-- 浏览器 cookie 里放的是随机串，这里只存它的 SHA-256
CREATE TABLE sessions (
  id_hash TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE INDEX sessions_email ON sessions (email);

-- 标注任务。kind = shuttle：一个回合的球位置；stroke：约 20 拍的击球类型。范围是视频帧号（两端都算）。
-- status：open（大厅里等人领）→ claimed（有人在标，lease_until 之前有效）→ submitted（标完提交）→ approved（管理员看过）。
-- active = 0：重新准备标注材料后，这个任务对应的回合或击球已经不在材料里了，不再显示。
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('shuttle', 'stroke')),
  idx INTEGER NOT NULL,          -- 第几个回合 / 第几组（从 0 开始，只用来显示）
  start_frame INTEGER NOT NULL,
  end_frame INTEGER NOT NULL,
  n_items INTEGER NOT NULL,      -- 多少帧 / 多少拍
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'claimed', 'submitted', 'approved')),
  assignee TEXT NOT NULL DEFAULT '',
  claimed_at TEXT,
  lease_until TEXT,
  submitted_at TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL,
  UNIQUE (video_id, kind, start_frame)
);
CREATE INDEX tasks_assignee ON tasks (assignee, status);

-- 任务是按标注材料生成的：材料比上次生成任务时新，就重新对一遍
ALTER TABLE videos ADD COLUMN annot_tasks_at TEXT;
ALTER TABLE videos ADD COLUMN strokes_tasks_at TEXT;

-- 谁标的（邮箱）。以前的标注是空的，第一个管理员绑定账号时补成管理员。
ALTER TABLE labels ADD COLUMN annotator TEXT NOT NULL DEFAULT '';
ALTER TABLE stroke_labels ADD COLUMN annotator TEXT NOT NULL DEFAULT '';

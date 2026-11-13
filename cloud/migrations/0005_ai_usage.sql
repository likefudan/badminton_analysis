-- AI 报告每次调用模型的用量（录像删了记录也留着，用来算累计），和用户的设置（默认模型、单价）。
CREATE TABLE ai_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  kind TEXT NOT NULL,            -- report | ask
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost_usd REAL,                 -- 按当时的单价估算；没有单价时是 NULL
  created_at TEXT NOT NULL
);
CREATE INDEX ai_usage_video ON ai_usage (video_id, id);

CREATE TABLE settings (
  user_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY (user_id, key)
);

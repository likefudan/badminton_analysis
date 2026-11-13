-- 处理进度（节点上报的步骤列表，JSON）和处理节点最后一次联系的时间。
ALTER TABLE videos ADD COLUMN job_progress TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN job_started_at TEXT;

CREATE TABLE nodes (
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  seen_at TEXT NOT NULL,
  PRIMARY KEY (user_id, name)
);

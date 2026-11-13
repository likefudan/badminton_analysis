-- 原始录像和分片上传进度。user_id 现在只有一个值，表结构按多用户设计。
CREATE TABLE videos (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  size INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  part_size INTEGER NOT NULL,
  r2_key TEXT NOT NULL,
  upload_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('uploading', 'uploaded')),
  meta TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX videos_user_created ON videos (user_id, created_at DESC);

CREATE TABLE video_parts (
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  part_number INTEGER NOT NULL,
  etag TEXT NOT NULL,
  size INTEGER NOT NULL,
  PRIMARY KEY (video_id, part_number)
);

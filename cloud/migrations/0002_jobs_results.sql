-- 处理任务和结果文件。录像传完后进入 queued，Mac 处理节点领取后 processing，结束是 done 或 failed。
-- archived：结果因为云端只留最近约 10 GB 被清理，Mac 上还有。
ALTER TABLE videos ADD COLUMN job_status TEXT NOT NULL DEFAULT 'none';
ALTER TABLE videos ADD COLUMN job_stage TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN job_error TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN job_node TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN job_lease_until TEXT;
ALTER TABLE videos ADD COLUMN job_updated_at TEXT;
ALTER TABLE videos ADD COLUMN options TEXT NOT NULL DEFAULT '{}';
ALTER TABLE videos ADD COLUMN original_deleted_at TEXT;
UPDATE videos SET job_status = 'queued' WHERE status = 'uploaded';

CREATE TABLE results (
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  size INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (video_id, name)
);

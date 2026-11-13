-- 手机上标注球的位置。网页点「标注球位置」后 annot_status = queued，Mac 节点领取后 preparing，
-- 切好图、传完是 ready，失败是 failed。图片和预测放在 R2 的 annot/ 下（不算在 10 GB 结果里）。
-- 标注本身存在 labels 表里，Mac 节点把它同步回录像旁边的 .labels.csv（labels_synced_at 记到哪一次改动）。
ALTER TABLE videos ADD COLUMN annot_status TEXT NOT NULL DEFAULT 'none';
ALTER TABLE videos ADD COLUMN annot_stage TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN annot_error TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN annot_options TEXT NOT NULL DEFAULT '{}';
ALTER TABLE videos ADD COLUMN annot_lease_until TEXT;
ALTER TABLE videos ADD COLUMN annot_updated_at TEXT;
ALTER TABLE videos ADD COLUMN labels_changed_at TEXT;
ALTER TABLE videos ADD COLUMN labels_synced_at TEXT;

CREATE TABLE labels (
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  frame INTEGER NOT NULL,
  visibility INTEGER NOT NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  status TEXT NOT NULL,          -- human | bulk | skip（和 badminton annotate 的含义一样）
  updated_at TEXT NOT NULL,
  PRIMARY KEY (video_id, frame)
);

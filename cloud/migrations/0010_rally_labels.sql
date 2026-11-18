-- 结果页上人给每个回合标谁赢（「回合和得失分」一栏）。节点同步回录像旁边的 .points.labels.csv，
-- 重算 points.json 和 AI 报告的摘要再传上来。
ALTER TABLE videos ADD COLUMN rally_labels_changed_at TEXT;
ALTER TABLE videos ADD COLUMN rally_labels_synced_at TEXT;

CREATE TABLE rally_labels (
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  frame INTEGER NOT NULL,        -- 这个回合最后一拍的视频帧号（页面上看到的那一版 points.json 里的）
  winner TEXT NOT NULL CHECK (winner IN ('near', 'far', 'none')),  -- none：不算回合（把球打回去、热身）
  note TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  PRIMARY KEY (video_id, frame)
);

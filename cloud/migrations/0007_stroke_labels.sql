-- 手机上标注每一拍的技术类型。和 0006 的球位置标注一样：网页请求 → Mac 节点准备（每拍一段小视频，
-- 放在 R2 的 strokes/ 下）→ 网页标 → 节点同步回录像旁边的 .strokes.labels.csv。
ALTER TABLE videos ADD COLUMN strokes_status TEXT NOT NULL DEFAULT 'none';
ALTER TABLE videos ADD COLUMN strokes_stage TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN strokes_error TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN strokes_lease_until TEXT;
ALTER TABLE videos ADD COLUMN strokes_updated_at TEXT;
ALTER TABLE videos ADD COLUMN stroke_labels_changed_at TEXT;
ALTER TABLE videos ADD COLUMN stroke_labels_synced_at TEXT;

CREATE TABLE stroke_labels (
  video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  frame INTEGER NOT NULL,        -- 击球那一帧（视频帧号）
  player_id TEXT NOT NULL,       -- 击球人编号，空字符串 = 不知道
  type TEXT NOT NULL,            -- 主类型（放网 … 发长球、未知）
  tags TEXT NOT NULL,            -- 附加标记，分号分隔（劈杀;对角）
  note TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (video_id, frame)
);

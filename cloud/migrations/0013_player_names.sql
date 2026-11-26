-- 每段录像的球员名字（结果页「球员是谁」里由超级管理员填）。
-- JSON：{"names": {"1": "小王"}, "was": {"1": ["球员 1"]}}；was 是改名前页面上显示过的名字，
-- 网页用它把旧的 AI 报告里的名字也换掉。NULL = 从没在网页上起过名字（Mac 上的 players.json 不动）。
ALTER TABLE videos ADD COLUMN player_names TEXT;

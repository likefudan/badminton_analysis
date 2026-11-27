-- 「能用 Mac 分析」：超级管理员给哪个成员开了，他就能上传自己的录像，排进家里 Mac 的队列（排在超级管理员的后面），
-- 管自己传的录像（改名、删除、重跑、起球员名字、分析区域、标胜负、AI 报告有每月次数）。和角色（只能看 / 标注员 / 管理员）叠加。
ALTER TABLE users ADD COLUMN can_analyze INTEGER NOT NULL DEFAULT 0;
-- 谁传的录像：空 = 超级管理员（以前的录像都是）；别人传的写他的邮箱（小写）。只有他和超级管理员看得到。
ALTER TABLE videos ADD COLUMN uploader TEXT NOT NULL DEFAULT '';
-- 是谁让模型写的（算每人每月的次数）；空 = 超级管理员
ALTER TABLE ai_usage ADD COLUMN actor TEXT NOT NULL DEFAULT '';
CREATE INDEX ai_usage_actor ON ai_usage (actor, kind, created_at);

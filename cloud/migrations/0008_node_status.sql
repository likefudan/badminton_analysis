-- Mac 节点每半分钟上报一次自己的状态（当前作业、后台任务、最近日志、机器负载），原样存成 JSON，
-- 网页的「Mac 状态」页读它。seen_at 仍然是节点最后一次联系的时间。
ALTER TABLE nodes ADD COLUMN status TEXT NOT NULL DEFAULT '';
ALTER TABLE nodes ADD COLUMN status_at TEXT;

-- 手机上远程启停 Mac 上的处理程序。
-- 网页把指令记在这里；Mac 上常驻的面板服务（badminton dashboard）每 10 秒上报一次自己看到的状态，
-- 顺便把等着的指令取走执行。处理程序停着的时候面板服务还在，所以「启动」也能送到。
ALTER TABLE nodes ADD COLUMN agent TEXT NOT NULL DEFAULT '';   -- 面板服务最后一次上报的内容（JSON）
ALTER TABLE nodes ADD COLUMN agent_at TEXT;

CREATE TABLE node_commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  node TEXT NOT NULL,
  action TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending 等 Mac 来取，taken 已取走，done 已执行，failed 没成功或没人来取
  message TEXT NOT NULL DEFAULT '',
  created_by TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX node_commands_node ON node_commands (user_id, node, id);

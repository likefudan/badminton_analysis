-- 录像的标题（用户自己起的名字，比如「周六双打第 2 局」）；空着就显示文件名。
ALTER TABLE videos ADD COLUMN title TEXT NOT NULL DEFAULT '';

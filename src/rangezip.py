"""Чтение отдельных файлов из большого удалённого ZIP через HTTP Range."""
import io
import zipfile

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos = url, block, 0
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.size = int(r.headers["Content-Length"])
        self._cache = (0, b"")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start, end):
        h = {"Range": f"bytes={start}-{end - 1}"}
        for _ in range(4):
            try:
                r = requests.get(self.url, headers=h, timeout=120)
                if r.status_code == 206:
                    return r.content
            except requests.RequestException:
                pass
        raise IOError(f"range {start}-{end} failed")

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        cs, cd = self._cache
        if not (cs <= self.pos and end <= cs + len(cd)):
            stop = min(max(end, self.pos + self.block), self.size)
            cd, cs = self._fetch(self.pos, stop), self.pos
            self._cache = (cs, cd)
        out = cd[self.pos - cs:end - cs]
        self.pos = end
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip(url):
    return zipfile.ZipFile(HttpRangeFile(url))

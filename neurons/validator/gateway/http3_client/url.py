from urllib.parse import urlparse


class URL:
    def __init__(self, url: str) -> None:
        parsed = urlparse(url)

        self.authority = parsed.netloc
        parts = parsed.netloc.split(":")
        self.host = parts[0]
        if len(parts) == 2:
            self.port = int(parts[1])
        else:
            self.port = 443
        self.full_path = parsed.path or "/"
        if parsed.query:
            self.full_path += "?" + parsed.query
        self.scheme = parsed.scheme

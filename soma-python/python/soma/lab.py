"""Lab connection API for remote Soma execution."""

import json
try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


class Lab:
    """Connection to a remote Soma lab.

    Example::

        lab = Lab.connect("http://localhost:3000")
        print(lab.health())
        print(lab.info())
        # lab.run(pipeline, data) — future
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._check_connection()

    @classmethod
    def connect(cls, url: str) -> "Lab":
        """Connect to a Soma lab at the given URL."""
        return cls(url)

    def _check_connection(self):
        """Verify the lab is reachable."""
        try:
            self._get("/health")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Soma lab at {self.url}: {e}")

    def health(self) -> str:
        """Check lab health."""
        return self._get("/health")

    def info(self) -> dict:
        """Get worker capabilities."""
        resp = self._get("/info")
        return json.loads(resp)

    def workers(self) -> list:
        """List available workers."""
        info = self.info()
        return [info]  # Single worker for now

    def _get(self, path: str) -> str:
        """Make a GET request to the lab."""
        url = f"{self.url}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8")

    def __repr__(self):
        return f"Lab({self.url!r})"

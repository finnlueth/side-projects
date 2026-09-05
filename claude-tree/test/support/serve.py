"""Serves the harness at /chat/<uuid> so the extension sees a real conversation URL."""
import http.server, os
from http.server import ThreadingHTTPServer


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.split('?')[0].startswith('/chat/'):
            return os.path.join(os.getcwd(), '_check.html')
        return super().translate_path(path)

    def log_message(self, *a):
        pass


# Threaded on purpose. A single-threaded server takes one connection at a time, and a suite
# that drives several pages at once then queues behind itself: requests stall, and a page
# times out on navigation or reloads part way through, which reads as a failing test rather
# than a slow server.
ThreadingHTTPServer.allow_reuse_address = True
ThreadingHTTPServer.daemon_threads = True
with ThreadingHTTPServer(('127.0.0.1', 8765), Handler) as httpd:
    httpd.serve_forever()

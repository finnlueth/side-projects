"""Serves the harness at /chat/<uuid> so the extension sees a real conversation URL."""
import http.server, socketserver, os

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.split('?')[0].startswith('/chat/'):
            return os.path.join(os.getcwd(), '_check.html')
        return super().translate_path(path)
    def log_message(self, *a):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', 8765), Handler) as httpd:
    httpd.serve_forever()

import cv2
import numpy as np
import urllib.request
import base64
import threading
import time

class MJPEGStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = True
        self.failed = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _make_request(self):
        if '@' in self.url:
            scheme = self.url.split('://')[0]
            user_pass = self.url.split('@')[0].split('://')[1]
            host_path = self.url.split('@')[1]
            clean_url = f'{scheme}://{host_path}'
            encoded = base64.b64encode(user_pass.encode()).decode()
            req = urllib.request.Request(clean_url)
            req.add_header('Authorization', f'Basic {encoded}')
        else:
            req = urllib.request.Request(self.url)
        return req

    def _reader(self):
        for attempt in range(5):
            try:
                req = self._make_request()
                stream = urllib.request.urlopen(req, timeout=15)
                self.failed = False
                buffer = b""
                while self.running:
                    buffer += stream.read(8192)
                    a = buffer.find(b'\xff\xd8')
                    b = buffer.find(b'\xff\xd9')
                    while a != -1 and b != -1 and b > a:
                        jpg = buffer[a:b+2]
                        buffer = buffer[b+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.frame = frame
                        a = buffer.find(b'\xff\xd8')
                        b = buffer.find(b'\xff\xd9')
                return
            except urllib.request.HTTPError as e:
                if e.code == 401:
                    print(f"  Auth required for stream. Use: http://user:pass@{self.url.split('://')[1]}")
                    self.failed = True
                    self.running = False
                    return
                print(f"  Stream attempt {attempt+1}/5 failed: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"  Stream attempt {attempt+1}/5 failed: {e}")
                time.sleep(2)
        print("  Could not connect to stream after 5 attempts")
        self.failed = True
        self.running = False

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def isOpened(self):
        return self.running and self.frame is not None and not self.failed

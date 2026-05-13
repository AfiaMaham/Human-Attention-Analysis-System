import threading
import cv2


class VideoStream:
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            if not self.ret:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        self._thread.join()
        self.cap.release()

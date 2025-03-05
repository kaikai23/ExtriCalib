import os
import cv2
import uuid
import numpy as np
import queue
import threading
from typing import Optional, Any, List, Union, Tuple


class FakeVideoCapture():
    def __init__(self, source: str) -> None:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        self.frame = cv2.imread(source)

    def isOpened(self):
        return True

    def read(self):
        if self.frame is not None:
            return True, self.frame.copy()
        return False, None

    def release(self):
        self.frame = None
        return True


class OpenCVWindow:
    def __init__(
        self, name: str, width: Optional[int] = None, height: Optional[int] = None
    ):
        self.opencv_winname = name if name is not None else str(uuid.uuid4())

        cv2.namedWindow(winname=self.opencv_winname, flags=cv2.WINDOW_NORMAL)
        if width is None:
            width = 640
        if height is None:
            height = 480

        # cv2.resizeWindow(winname=self.opencv_winname, width=width, height=height)
        self.resize(width, height)

    def show(self, mat: np.asarray, wait_time: int = 1) -> Any:
        assert wait_time >= 0, "Wait time must be a positive integer or zero"
        cv2.imshow(self.opencv_winname, mat)
        q = cv2.waitKey(wait_time)
        return q

    def close(self) -> None:
        cv2.destroyWindow(winname=self.opencv_winname)

    def resize(self, width: int, height: int) -> None:
        cv2.resizeWindow(winname=self.opencv_winname,
                         width=width, height=height)


class MousePointsClick:
    def __init__(
        self, window: Union[OpenCVWindow, str], num_pts: Optional[int] = None
    ) -> None:
        if isinstance(window, str):
            self.window_name = window
        elif isinstance(window, OpenCVWindow):
            self.window_name = window.opencv_winname
        else:
            raise NotImplementedError()

        if num_pts is not None:
            assert num_pts > 0, "Not valid number of points"

        self.max_pts = num_pts
        self._is_finished = False
        self.points: List[np.ndarray] = []

        self.set()

    def set(self) -> None:
        cv2.setMouseCallback(self.window_name, self.callback)

    def callback(self, event: Any, x: Any, y: Any, flags: Any, param: Any) -> None:
        # if event == cv2.EVENT_LBUTTONDBLCLK:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(np.asarray([x, y]))

        if self.max_pts is not None:
            if len(self.points) >= self.max_pts:
                self._is_finished = True
                self.stop()

    def get_points(
        self,
        window: OpenCVWindow,
        image: np.ndarray,
        pts_size: int = 5,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        img = image.copy()  # we are drawing on it, keep a copy
        while True:
            # Draw current collected points
            points_util_now = self.points
            p: np.ndarray
            for p in points_util_now:
                pt = (int(p[0]), int(p[1]))
                cv2.circle(img, pt, pts_size, color, -1)

            # Show image and stop if 'q' is pressed or if we have collected
            key = window.show(img)
            # 13 is [ENTER] key
            if key == ord("q") or self.done() or key == 13:
                window.close()
                break

    def done(self) -> bool:
        return self._is_finished

    def stop(self) -> None:
        cv2.setMouseCallback(self.window_name, lambda *args: None)


class AsyncVideoCapture:
    def __init__(self, name: str) -> None:
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> Tuple[bool, np.ndarray]:
        return True, self.q.get()

import threading
import cv2
import numpy as np
from queue import Queue, Empty
from typing import Tuple


# TODO:
# currently, the following error is raised due to running cv2 not in the main thread:
#   QObject::killTimer: Timers cannot be stopped from another thread
#   QObject::~QObject: Timers cannot be stopped from another thread
class ImageViewer:
    """
    Example usage:
    viewer = ImageViewer(window_name="Test Viewer", refresh_rate=0.02)
    while viewer.running:
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        viewer.update_image(image)
    viewer.stop()
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (512, 512),
        window_name: str = "Simulator Viewer",
        refresh_rate: float = 0.02,
    ):
        self.window_name = window_name
        self.refresh_rate = refresh_rate
        self.image_queue = Queue()
        self.running = True
        self.current_image = None
        self.window_size = window_size

        # Start the thread
        self.viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
        self.viewer_thread.start()

    def _run_viewer(self):
        cv2.namedWindow(self.window_name)
        while self.running:
            try:
                # Get the latest image from the queue, with a small timeout to avoid blocking forever
                self.current_image = self.image_queue.get(timeout=self.refresh_rate)
            except Empty:
                # If no new image is available, continue displaying the current image
                pass

            if self.current_image is not None:
                # Resize image to fit window
                self.current_image = cv2.resize(self.current_image, self.window_size)
                cv2.imshow(self.window_name, self.current_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(int(self.refresh_rate * 1000)) & 0xFF == ord("q"):
                self.running = False

        # Close window when done
        cv2.destroyWindow(self.window_name)

    def update_image(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array.")
        self.image_queue.put(image)

    def stop(self):
        self.running = False
        self.viewer_thread.join()

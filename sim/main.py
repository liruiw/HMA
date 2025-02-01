import numpy as np
import imageio
from sim.simulator import Simulator
from sim.policy import Policy
from sim.viewer import ImageViewer
from typing import List, Tuple

step_time = []
psnr = []
delta_psnr = []


class InteractiveDigitalWorld:
    def __init__(
        self,
        simulator: Simulator,
        policy: Policy,
        offscreen: bool = True,  # if False, show live window
        window_size: Tuple[int, int] = (512, 512),
    ):
        self.simulator = simulator
        self.policy = policy
        self.offscreen = offscreen
        self.video_frames: List[np.ndarray] = []
        self.dt = simulator.dt

        self.obs = self.simulator.reset()  # input to policy
        self.video_frames.append(self.obs)

        if not offscreen:
            self.viewer = ImageViewer(
                window_name=(f"Simulator: {simulator.__class__.__name__} | " f"Policy: {policy.__class__.__name__}"),
                refresh_rate=self.dt,
                window_size=window_size,
            )
            self.viewer.update_image(self.obs)

    def step(self) -> None:
        action = self.policy.generate_action(self.obs)
        result = self.simulator.step(action)
        next_frame = result["pred_next_frame"]
        if "gt_next_frame" in result:
            gt_next_frame = result["gt_next_frame"]
            next_frame = np.concatenate([next_frame, gt_next_frame], axis=1)
        if "psnr" in result:
            psnr.append(result["psnr"])
        if "delta_psnr" in result:
            delta_psnr.append(result["delta_psnr"])
        if "step_time" in result:
            step_time.append(result["step_time"])
        self.obs = next_frame
        if not self.offscreen:
            self.viewer.update_image(next_frame)
        self.video_frames.append(next_frame)

    def save_video(self, save_path: str, as_gif: bool = False) -> None:
        if as_gif:
            imageio.mimsave(save_path, self.video_frames, format="GIF", fps=1 / self.dt)
        else:
            imageio.mimsave(save_path, self.video_frames, format="mp4", fps=1 / self.dt)
        print(f"{'GIF' if as_gif else 'MP4'} saved to {save_path}")

    def reset(self) -> None:
        self.obs = self.simulator.reset()
        self.video_frames = []

    def close(self) -> None:
        self.simulator.close()
        if not self.offscreen:
            self.viewer.stop()

        def analyze_scalar_sequence(data: List[float]):
            q1 = np.percentile(data, 25, method="nearest")
            median = np.median(data)
            q3 = np.percentile(data, 75, method="nearest")
            mean = np.mean([t for t in data if q1 <= t <= q3])
            return mean, median

        # report stats
        if len(step_time) > 0:
            # take mean over data between q1 and q3
            mean, median = analyze_scalar_sequence(step_time)
            print(f"=========== Timing ===========\n" f"Mean: {mean}\n" f"Meadian: {median}\n")

        if len(psnr) > 0:
            mean, median = analyze_scalar_sequence(psnr)
            print(f"=========== PSNR ===========\n" f"Mean: {mean}\n" f"Meadian: {median}\n")

        if len(delta_psnr) > 0:
            mean, median = analyze_scalar_sequence(delta_psnr)
            print(f"=========== Delta PSNR ===========\n" f"Mean: {mean}\n" f"Meadian: {median}\n")

"""Replay recorded teleop episodes to visually inspect data quality."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zarr

from hw3.sim_env import SO100SimEnv
from hw3.teleop_utils import CAMERA_NAMES, compose_camera_views
from so101_gym.constants import ASSETS_DIR

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=None,
                        help="Replay a specific episode (1-indexed). If omitted, replays all.")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    z = zarr.open_group(str(args.zarr), mode="r")
    ep_ends = z["meta"]["episode_ends"][:]
    starts = np.concatenate(([0], ep_ends[:-1]))

    joints = np.asarray(z["data"]["state_joints"][:])
    gripper = np.asarray(z["data"]["state_gripper"][:])

    env = SO100SimEnv(str(XML_PATH), render_w=640, render_h=480)

    episodes = list(range(len(ep_ends)))
    if args.episode is not None:
        episodes = [args.episode - 1]

    cv2.namedWindow("Replay", cv2.WINDOW_AUTOSIZE)

    for ep_idx in episodes:
        s, e = int(starts[ep_idx]), int(ep_ends[ep_idx])
        print(f"Episode {ep_idx+1}/{len(ep_ends)}: {e-s} steps  [press 'q' to quit, 'n' to skip]")

        for t in range(s, e):
            env.data.qpos[:5] = joints[t, :5]
            env.data.qpos[5] = gripper[t, 0]
            import mujoco
            mujoco.mj_forward(env.model, env.data)

            images = {cam: env.render(cam) for cam in CAMERA_NAMES}
            img = compose_camera_views(images, CAMERA_NAMES)

            cv2.putText(img, f"Ep {ep_idx+1}/{len(ep_ends)}  step {t-s}/{e-s}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Replay", img)

            key = cv2.waitKey(max(1, int(10 / args.speed))) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            if key == ord('n'):
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

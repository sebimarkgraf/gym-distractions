import os
import random
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

import distractor_dmc2gym as dmc2gym

domain_name = "cheetah"
task_name = "run"
distract_type = "dots_constant"
difficulty = "hard"
ground = "background"
background_dataset_path = Path("./background-datasets-test")
seed = 1
image_size = 256
action_repeat = 10
intensity = 1
save_video = False
np.random.seed(seed)
random.seed(seed)

env = dmc2gym.make(
    domain_name=domain_name,
    task_name=task_name,
    distraction_source=distract_type,
    distraction_location=ground,
    difficulty=difficulty,
    background_dataset_path=background_dataset_path,
    visualize_reward=False,
    from_pixels=True,
    height=image_size,
    width=image_size,
    frame_skip=action_repeat,
)

save_dir = "./test_env/"
frames = []


def video_record(env):
    frame = env.render(mode="rgb_array", height=image_size, width=image_size)
    frames.append(frame)


def video_save(file_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, file_name)
    imageio.mimsave(path, frames, fps=10)


def main():
    cv2.namedWindow("env", 0)
    cv2.resizeWindow("env", 1000, 1000)
    i = 0
    start = time.time()
    for _ in tqdm(range(5)):
        env.reset()
        done = False
        while not done:
            img = env.render(mode="rgb_array")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            cv2.imshow("env", img)
            # cv2.imwrite(f"test_env/env_{distract_type}_{ground}_{i}.png", img)
            # cv2.imwrite(
            #    f"test_env/env_{distract_type}_{ground}_mask_{i}.png",
            #    info["mask"].astype(float) * 255.0,
            # )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
            i += 1

    stop = time.time()
    print(f"Took {stop - start}")


if __name__ == "__main__":
    main()
    print("done")

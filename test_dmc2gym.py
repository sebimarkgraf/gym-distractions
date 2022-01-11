import os
import random
from pathlib import Path

import cv2
import imageio
import numpy as np

import distractor_dmc2gym as dmc2gym

domain_name = 'cartpole'
task_name = 'swingup'
distract_type = 'dots'
difficulty = 'hard'
ground = 'foreground'
background_dataset_path = Path('./davis')
seed = 1
image_size = 84
action_repeat = 2
intensity = 1
save_video = False
np.random.seed(seed)
random.seed(seed)

env = dmc2gym.make(domain_name=domain_name,
                   task_name=task_name,
                   distract_type=distract_type,
                   ground=ground,
                   difficulty=difficulty,
                   intensity=intensity,
                   background_dataset_path=background_dataset_path,
                   seed=seed,
                   visualize_reward=False,
                   from_pixels=True,
                   height=image_size,
                   width=image_size,
                   frame_skip=action_repeat,
                   )

save_dir = './test_env/'
frames = []


def video_record(env):
    frame = env.render(mode='rgb_array', height=image_size, width=image_size)
    frames.append(frame)


def video_save(file_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, file_name)
    imageio.mimsave(path, frames, fps=10)


def main():
    env.save_distractors_info(save_dir)
    cv2.namedWindow("env", 0)
    cv2.resizeWindow("env", 1000, 1000)
    i = 0
    while True:
        env.reset()
        done = False
        while not done:
            img = env.render(mode='rgb_array')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            video_record(env)
            cv2.imshow('env', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            i += 1
            if save_video:
                if i == 200:
                    video_save('RandomDots_for.mp4')
                    return


if __name__ == '__main__':
    main()
    print('done')

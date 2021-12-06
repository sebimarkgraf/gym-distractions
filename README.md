# occluded_dmc

## Instructions
to use local_dmc2gym just import local_dmc2gym, and use make() method.
```
import local_dmc2gym as dmc2gym

env = dmc2gym.make(domain_name,
                    task_name,
                    distract_type,
                    ground, 
                    difficulty, 
                    background_dataset_path,
                    train_or_val, 
                    seed,
                    visualize_reward,
                    from_pixels,
                    height,
                    width,
                    camera_id,
                    frame_skip,
                    episode_length,
                    environment_kwargs,
                    time_limit,
                    channels_first
                    )
```
The following is the explanation of the parameters which used to control the background/foreground,  
and the rest of the parameters are the same as the original dmc2gym code.
```
distract_type             : choice{'color', 'noise', 'dots', 'videos'}, default: None
ground                    : choice{'background', 'forground', 'both'}, default: None, 
                            'both' only used for 'videos' distractor, 'color' and 'noise' only have 'background'
difficulty                : choice{'easy', 'medium', 'hard'}, default: 'hard' 
                            only useful for 'dots' and 'videos' distractor
                            when use 'dots' distractor, set num_dots to: 'hard'=16, 'medium'=8, 'easy'=5; 
                            when use 'videos' distract type, set num_video to: 'hard'=all, others the same as 'dots'
intensity                 : 0-1, default: 1
                            distracting intensity(non-transparency?): 1 is all distrated, 0 is same as original env
background_dataset_path   : where you put your video/image dataset, only useful for 'videos'
train_or_val              : choice{'train', 'val'}, default: None
                            when use DAVIS Dataset, can divided it to train-set and validation-set
```
when you use 'dots', by default, distractors are repeated, and movements of dots obey the dynamics of 
an ideal gas with no collison. If you want to change those default settings, or you want to modify
sizes/velocitys/positions/quantity/or some others of dots, 
you can modify them in file 'background_source.py', class 'RandomDotsSource' by yourself.

you can modify 'test_dmc2gym.py' and test.
```
python test_dmc2gym.py
```

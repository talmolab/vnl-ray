"""Utility functions."""

from typing import Sequence

from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict
import numpy as np
from PIL import Image


def rollout_and_render(env, policy, n_steps=100, run_until_termination=False, camera_ids=[-1], **render_kwargs):
    """Rollout policy for n_steps or until termination, and render video.
    Rendering is possible from multiple cameras; in that case, each element in
    returned `frames` is a list of cameras."""
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    while (i < n_steps and not run_until_termination) or (timestep.step_type != 2 and run_until_termination):
        i += 1
        frame = []
        for camera_id in camera_ids:
            frame.append(env.physics.render(camera_id=camera_id, **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int)
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def parse_mujoco_camera(s: str):
    """Parse `Copy camera` XML string from MuJoCo viewer to pos and xyaxes.

    Example input string:
    <camera pos="-4.552 0.024 3.400" xyaxes="0.010 -1.000 0.000 0.382 0.004 0.924"/>
    """
    split = s.split('"')
    pos = split[1]
    xyaxes = split[3]
    pos = [float(s) for s in pos.split()]
    xyaxes = [float(s) for s in xyaxes.split()]
    return pos, xyaxes


# Helper Function
def blow(x, repeats=2):
    """Repeat columns and rows requested number of times."""
    return np.repeat(np.repeat(x, repeats, axis=0), repeats, axis=1)


def vision_rollout_and_render(env, policy, camera_id=1, eye_blow_factor=5, **render_kwargs):
    """Run vision-guided flight episode and render frames, including eyes."""
    frames = []
    timestep = env.reset()
    # Run full episode until it ends.
    i = 0
    while timestep.step_type != 2 and i < 500:
        i += 1
        # Render eyes and scene.
        pixels = env.physics.render(camera_id=camera_id, **render_kwargs)
        # raw eye pixel is accessible in the timestep
        eyes = eye_pixels_from_observation(timestep, blow_factor=eye_blow_factor)
        # Add eye pixels to scene.
        pixels[0 : eyes.shape[0], 0 : eyes.shape[1], :] = eyes
        frames.append(pixels)
        # Step environment.
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def eye_pixels_from_observation(timestep, blow_factor=4):
    """Get current eye view from timestep.observation."""
    # In the actual task, the averaging over axis=-1 is done by the visual
    # network as a pre-processing step, so effectively the visual observations
    # are gray-scale.
    egocentric_camera = timestep.observation["walker/egocentric_camera"].mean(axis=-1)

    pixels = egocentric_camera
    pixels = np.tile(pixels[:, :, None], reps=(1, 1, 3))
    pixels = blow(pixels, blow_factor)
    pixels = pixels.astype("uint8")
    return pixels


def eye_pixels_from_cameras(physics, **render_kwargs):
    """Render two-eye view, assuming eye cameras have particular names."""
    for i in range(physics.model.ncam):
        name = physics.model.id2name(i, "camera")
        if "eye_left" in name:
            left_eye = physics.render(camera_id=i, **render_kwargs)
        if "eye_right" in name:
            right_eye = physics.render(camera_id=i, **render_kwargs)
    pixels = np.hstack((left_eye, right_eye))
    return pixels


# Frame width and height for rendering.
render_kwargs = {"width": 640, "height": 480}


def render_with_rewards_info(env, policy, rollout_length=600, render_vision_if_available=True):
    """
    Generate rollout with the reward related information.
    """
    reward_channels = []
    frames = []
    reset_idx = []
    timestep = env.reset()

    render_kwargs = {"width": 640, "height": 480}
    for i in range(rollout_length):
        pixels = env.physics.render(camera_id=1, **render_kwargs)
        # optionally also render the vision.
        if render_vision_if_available and "walker/egocentric_camera" in timestep.observation:
            eyes = eye_pixels_from_observation(timestep, blow_factor=3)
            # Add eye pixels to scene.
            pixels[0 : eyes.shape[0], 0 : eyes.shape[1], :] = eyes
        frames.append(pixels)
        action = policy(timestep.observation)
        timestep = env.step(action)
        reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)
    return frames, reset_idx, reward_channels


def agg_backend_context(func):
    def wrapper(*args, **kwargs):
        orig_backend = matplotlib.get_backend()
        matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
        # Code to execute BEFORE the original function
        result = func(*args, **kwargs)
        # Code to execute AFTER the original function
        plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
        matplotlib.use(orig_backend)
        return result

    return wrapper


@agg_backend_context
def plot_reward(idx, episode_start, rewards: dict, ylim=(-0.05, 1.1), terminated=False):
    """
    visualization technics
    returns the rgb array of the reward composition.
    """
    ylim = list(ylim)  # to make it dynamic
    window_size = 250
    idx_in_this_episode = idx - episode_start
    plt.figure(figsize=(6.4, 4.8))
    for key, val in rewards.items():
        plt.plot(val[episode_start:idx], label=key)
        plt.scatter(idx - episode_start, val[idx])
    if terminated:
        plt.axvline(x=idx - episode_start, color="r", linestyle="-")
        # Add the text label
        plt.text(
            idx - episode_start - 8,  # Adjust the x-offset as needed
            sum(ylim) / 2,  # Adjust the y-position as needed
            "Episode Terminated",
            color="r",
            rotation=90,
        )  # Rotate the text vertically
    if idx_in_this_episode <= window_size:
        plt.xlim(0, window_size)
    else:
        plt.xlim(idx_in_this_episode - window_size, idx_in_this_episode)  # dynamically move xlim as time progress
    max_reward = np.max(list(rewards.values()))
    if max_reward > ylim[1]:
        ylim[1] = max_reward + 0.1
    plt.ylim(*ylim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title("Reward Composition")
    # Get the current figure
    fig = plt.gcf()
    # Create a canvas for rendering
    canvas = FigureCanvasAgg(fig)
    # Render the canvas to a buffer
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    # Convert the buffer to a PIL Image
    image = Image.frombytes("RGBA", (width, height), s)
    rgb_array = np.array(image.convert("RGB"))
    return rgb_array


def render_with_rewards(env, policy, rollout_length=600):
    """
    render with the rewards progression graph concat alongside with the rendering
    """
    frames, reset_idx, reward_channels = render_with_rewards_info(env, policy, rollout_length=rollout_length)
    rewards = defaultdict(list)
    reward_keys = env.task._reward_keys
    for key in reward_keys:
        rewards[key] += [rcs[key] for rcs in reward_channels]
    concat_frames = []
    episode_start = 0
    # implement reset logics of the reward graph too.
    for idx, frame in enumerate(frames):
        if len(reset_idx) != 0 and idx == reset_idx[0]:
            reward_plot = plot_reward(idx, episode_start, rewards, terminated=True)
            for _ in range(50):
                concat_frames.append(np.hstack([frame, reward_plot]))  # create stoppage when episode terminates
            reset_idx.pop(0)
            episode_start = idx
            continue
        concat_frames.append(np.hstack([frame, plot_reward(idx, episode_start, rewards)]))
    return concat_frames

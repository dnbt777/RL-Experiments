
import argparse
import argparse
import pygame
import torch
from snake_game import SnakeGame
from dqn import DQNAgent
from config import *
from renderer import Renderer
import time
import os
import subprocess

def render_model_runs(model_path, num_episodes):
    pygame.init()
    display = (WIDTH, HEIGHT)
    window = pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    
    renderer = Renderer()
    game = SnakeGame()
    state_shape = game.get_state().shape
    input_channels = state_shape[0]
    action_dim = len(BASIS_DIRECTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(input_channels=input_channels, n_actions=action_dim, epsilon=0, epsilon_min=0)
    agent.load_models(model_path)
    agent.q_network.eval()

    timestamp = int(time.time())
    render_dir = f"./render_frames/{timestamp}"
    os.makedirs(render_dir, exist_ok=True)

    frame_count = 0
    movement_counter = 0

    for episode in range(num_episodes):
        state = game.reset()
        episode_done = False

        while not episode_done:
            if movement_counter == 0:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.choose_action(state_tensor)
                next_state, reward, done = game.step(action)
                print(next_state, reward, done)
                
                if done:
                    episode_done = True
                
                state = next_state

            snake, food = game.get_render_data()
            renderer.render(snake, food, frame_output_path=f"{render_dir}")
            
            movement_counter = (movement_counter + 1) % FRAMES_PER_SNAKE_MOVEMENT

    pygame.quit()

    # Stitch frames into video
    output_video = f"./render_frames/output_{timestamp}.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", "30",
        "-i", f"{render_dir}/frame_%05d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Add audio to the video
    final_output = f"./render_frames/final_output_{timestamp}.mp4"
    ffmpeg_add_audio_cmd = [
        "ffmpeg",
        "-i", output_video,
        "-i", song_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_output
    ]
    subprocess.run(ffmpeg_add_audio_cmd, check=True)

    print(f"Rendering complete. Final video saved as {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Snake Game Model Runs")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to render")
    args = parser.parse_args()

    render_model_runs(args.model_path, args.episodes)

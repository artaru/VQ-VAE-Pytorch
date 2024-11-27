import gymnasium as gym
from PIL import Image

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    observation, info = env.reset()

    for i in range(10000):
        action = env.action_space.sample()  # Agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        # Render and save the frame
        frame = env.render()  # Render in "rgb_array" mode for an image array
        img = Image.fromarray(frame)
        img.save(f"data/images/frame_{i:03}.png")  # Save to data folder

        # End the loop if the episode is over
        if terminated or truncated:
            observation, info = env.reset()  # Reset environment if the episode ends

    for i in range(1000):
        action = env.action_space.sample()  # Agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        # Render and save the frame
        frame = env.render()  # Render in "rgb_array" mode for an image array
        img = Image.fromarray(frame)
        img.save(f"data/validation_images/frame_{i:03}.png")  # Save to data folder

        # End the loop if the episode is over
        if terminated or truncated:
            observation, info = env.reset()  # Reset environment if the episode ends

    env.close()
    print("Images saved in the 'data' folder.")


from moviepy import VideoFileClip
input_file = r"D:\datasets\SSV2\s2s_test\Moving something across a surface until it falls down\3913.webm"
output_file = r"D:\datasets\SSV2\mp4\temp.mp4"


def save_mp4_tmp_file(input_file, output_file):
    video = VideoFileClip(input_file)
    width, height = video.size

    # Ensure dimensions are even (divisible by 2)
    new_width = width if width % 2 == 0 else width - 1
    new_height = height if height % 2 == 0 else height - 1

    # Resize if needed
    if (new_width, new_height) != (width, height):
        print(f"Resizing from {width}x{height} to {new_width}x{new_height}")
        video = video.resized(new_size=(new_width, new_height)) 

    # Write the video with the additional parameters
    video.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
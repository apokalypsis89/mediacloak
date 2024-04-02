from src.subtitle_generator import generate_subtitles
from src.face_detector import face_extraction
from src.stylegan2 import run_projection
import os

def main():
    # Folder structure variables
    svideo = r'\videos'
    fpath = r'\faces'

    # Folder paths
    input_video = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\input_videos\test_video01.mp4'
    output_put = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\output'

    #________________Generating Subttitles
    generate_subtitles(input_video, output_put + svideo)
    #________________Extracting Faces
    face_extraction(input_video, output_put + fpath)
    face_directory = output_put + fpath
    #________________Deepfake Generation
    network_pkl = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\module weight\stylegan2-ffhq-1024x1024.pkl'
    outdir = r'C:\Dev\Data Science and Machine Learning\MediaCloak\data\output\deepfake'
    target_fnames = target_fnames = [os.path.join(face_directory, f'face_{i}_.png') for i in range(23, 3619)]
    num_steps = 100
    seed = 303
    run_projection(
        network_pkl,
        target_fnames,
        outdir,
        seed,
        num_steps
    )

if __name__ == "__main__":
    main()

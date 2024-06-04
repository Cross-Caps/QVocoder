# FAD scores : conda activate FAD 



from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)



# background_embds_path = "/path/to/saved/background/embeddings.npy"
# eval_embds_path = "/path/to/saved/eval/embeddings.npy"

# # Compute FAD score while reusing the saved embeddings (or saving new ones if paths are provided and embeddings don't exist yet)
# fad_score = frechet.score(
#     "/path/to/background/set",
#     "/path/to/eval/set",
#     background_embds_path=background_embds_path,
#     eval_embds_path=eval_embds_path,
#     dtype="float32"
# )

source_real = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/generated_dir"

source_qnn = "/home/hiddenleaf/ARYAN_MT22019/hifi-ganQ/generated_dir"

target = "/home/hiddenleaf/ARYAN_MT22019/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/test_dir"

fad_score_real = frechet.score(source_real, target, dtype="float32")

print(fad_score_real)



fad_score_QNN = frechet.score(source_qnn, target, dtype="float32")

print(fad_score_QNN)







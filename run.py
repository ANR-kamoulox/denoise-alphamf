from modules import denoise, vad
import soundfile as sf
import argparse


def process(mix_path):
    mixture, fs = sf.read(mix_path)

    window = slice(0, mixture.shape[0])
    # vad_est = vad.vad(mixture[window, :], fs)

    real_speech, speech_filtered, noise_filtered = \
        denoise.alpha_denoise(
            mixture[window, :],
            L=50, alpha=1.9,
            nmh=1, burnin=False,
            niter=30,
            # vad=vad_est
        )

    return mixture


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )
    parser.add_argument(
        'input_file',
        help='audio file'
    )

    args = parser.parse_args()
    out = process(args.input_file)
    print(out)

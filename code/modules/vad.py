import webrtcvad
import numpy as np

def vad(input_audio, rate, sensitivity=1):
    # convert to 16bit signed int
    (T,I) = input_audio.shape
    audio = np.int16(input_audio * 32767)
    vad = webrtcvad.Vad()

    # mode 3 (max=3) means, very sensitive regarding to non-speech
    vad.set_mode(sensitivity)

    # window size 10ms
    window = int(rate * 0.01)

    voiced = np.zeros((T,))
    pos = 0
    while pos + window < T:
        for chan in range(I):
            chunk = audio[pos:pos + window,chan]
            voiced[pos:pos+window] += vad.is_speech(chunk.tobytes(), rate)
        pos += window
    return voiced/float(I)

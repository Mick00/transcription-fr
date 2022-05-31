
import os
from speechbrain.pretrained import EncoderASR

RECORDING_FOLDER = "./recordings/"
TRANSCRIPTION_FOLDER = "./transcription/"


def transcribe_audio():
    if not os.path.exists(TRANSCRIPTION_FOLDER):
        os.mkdir(TRANSCRIPTION_FOLDER)
    for file_name in os.listdir(RECORDING_FOLDER):
        print("--- Processing", file_name, "---")
        if file_name.endswith(".wav"):
            asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-fr",
                                                savedir="pretrainedModels/asrWav2vec2CommonvoiceFR")
            transcription = asr_model.transcribe_file(RECORDING_FOLDER + file_name)
            with open(TRANSCRIPTION_FOLDER + file_name + ".txt", "w+") as file:
                file.write(transcription)
        else:
            print("ERROR: format not supported")


if __name__ == '__main__':
    transcribe_audio()


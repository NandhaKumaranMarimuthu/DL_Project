import os 
os.chdir('D:\\GTECH\\CS-7643\\Project\\Comparison')

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from jiwer import wer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return []
    torchaudio.list_audio_backends = list_audio_backends
# ----------------------------------------------------------------

from speechbrain.pretrained import EncoderClassifier

########################################
# 1. CONFIG
########################################

ORIG_TRANS_PATH = "61-70968.trans.txt"
ORIG_UTT_ID = "61-70968-0003"
ORIG_AUDIO_PATH = "61-70968-0003.flac"

TTS_TRANS_PATH = "61-70968.trans_v1.txt"
TTS_UTT_ID = "61-70968-0005"
TTS_AUDIO_PATH = "61-70968-0005.flac"


########################################
# AUDIO LOADING
########################################

def load_audio_librosa(path, target_sr=None):

    y, sr = librosa.load(path, sr=target_sr)  # mono
    wav = torch.from_numpy(y).unsqueeze(0)    # (1, T)
    return wav, sr


########################################
# LOAD TEXT
########################################

def load_text_from_transcript(transcript_path, utt_id):

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            line_id, text = parts
            if line_id == utt_id:
                return text.strip()
    raise ValueError(f"Utterance ID {utt_id} not found in {transcript_path}")


########################################
# SPEAKER SIMILARITY WITH ECAPA-TDNN
########################################

def compute_speaker_similarity_ecapa(path_a, path_b):

    classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)

    # ECAPA model expects 16kHz mono
    wav_a, _ = load_audio_librosa(path_a, target_sr=16000)
    wav_b, _ = load_audio_librosa(path_b, target_sr=16000)

    wav_a = wav_a.to(DEVICE)
    wav_b = wav_b.to(DEVICE)

    with torch.no_grad():
        emb_a = classifier.encode_batch(wav_a).squeeze(0)
        emb_b = classifier.encode_batch(wav_b).squeeze(0)

    if emb_a.ndim == 2:
        emb_a = emb_a.mean(dim=0)
    if emb_b.ndim == 2:
        emb_b = emb_b.mean(dim=0)

    emb_a = emb_a / emb_a.norm(p=2)
    emb_b = emb_b / emb_b.norm(p=2)

    cos_sim = F.cosine_similarity(
        emb_a.unsqueeze(0), emb_b.unsqueeze(0)
    ).item()

    return cos_sim


########################################
# ASR + WER (unchanged)
########################################

def load_asr_model():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(DEVICE)
    labels = bundle.get_labels()
    return model, labels, bundle.sample_rate


def greedy_decode(emissions, labels):

    indices = torch.argmax(emissions, dim=-1)  # (batch, time)
    transcripts = []

    for inds in indices:
        prev = None
        tokens = []
        for i in inds:
            i = i.item()
            if i != 0 and i != prev:  # 0 = blank
                tokens.append(labels[i])
            prev = i
        transcripts.append("".join(tokens).replace("|", " ").strip())
    return transcripts


def transcribe_audio(path, model, labels, asr_sr):
    wav, _ = load_audio_librosa(path, target_sr=asr_sr)
    wav = wav.to(DEVICE)

    with torch.no_grad():
        emissions, _ = model(wav)
        emissions = torch.log_softmax(emissions, dim=-1)

    text = greedy_decode(emissions.cpu(), labels)[0]
    return text


def compute_wer(reference_text, hyp_text):
    ref = reference_text.lower().strip()
    hyp = hyp_text.lower().strip()
    return wer(ref, hyp)


########################################
# MAIN
########################################

def main():
    print("Loading reference transcripts...")
    orig_text = load_text_from_transcript(ORIG_TRANS_PATH, ORIG_UTT_ID)
    tts_text = load_text_from_transcript(TTS_TRANS_PATH, TTS_UTT_ID)

    print(f"Original text ({ORIG_UTT_ID}): {orig_text}")
    print(f"TTS text      ({TTS_UTT_ID}): {tts_text}")
    print()

    # ECAPA-TDNN speaker similarity
    print("Computing speaker similarity with ECAPA-TDNN...")
    spk_sim = compute_speaker_similarity_ecapa(ORIG_AUDIO_PATH, TTS_AUDIO_PATH)
    print(f"ECAPA speaker similarity (cosine, -1 to 1, higher = more similar): {spk_sim:.4f}")
    print()

    # ASR + WER
    print("Loading ASR model (Wav2Vec2)...")
    asr_model, asr_labels, asr_sr = load_asr_model()

    print("Transcribing original audio...")
    orig_asr = transcribe_audio(ORIG_AUDIO_PATH, asr_model, asr_labels, asr_sr)
    print(f"ASR (original): {orig_asr}")

    print("Transcribing TTS/test audio...")
    tts_asr = transcribe_audio(TTS_AUDIO_PATH, asr_model, asr_labels, asr_sr)
    print(f"ASR (TTS):      {tts_asr}")
    print()

    orig_wer_val = compute_wer(orig_text, orig_asr)
    tts_wer_val = compute_wer(tts_text, tts_asr)

    print(f"WER (original audio vs ref text): {orig_wer_val:.4f}")
    print(f"WER (TTS/test audio vs TTS text): {tts_wer_val:.4f}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()

import os 
os.chdir('D:\\GTECH\\CS-7643\\Project\\Comparison')

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from jiwer import wer


import matplotlib.pyplot as plt

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

ORIG_FOLDER = r"D:\GTECH\CS-7643\Project\Comparison\original"
GEN_FOLDER  = r"D:\GTECH\CS-7643\Project\Comparison\generated"

ORIG_TRANS_PATH = os.path.join(ORIG_FOLDER, "trans.txt")  # change name if needed


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



def load_all_transcripts(transcript_path):
    utt2text = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            line_id, text = parts
            utt2text[line_id] = text.strip()
    return utt2text


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
# PLOTTING 
########################################

def plot_results(utt_ids, orig_wers, gen_wers, spk_sims):
    x = np.arange(len(utt_ids))

    # WER plot
    plt.figure(figsize=(10, 5))
    width = 0.35
    plt.bar(x - width / 2, orig_wers, width, label="Original WER")
    plt.bar(x + width / 2, gen_wers,  width, label="Generated WER")
    plt.xticks(x, utt_ids, rotation=45, ha="right")
    plt.ylabel("WER")
    plt.title("WER per utterance (original vs generated)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("wer_per_utterance.png")
    plt.close()

    # ECAPA similarity plot
    plt.figure(figsize=(10, 5))
    plt.bar(x, spk_sims)
    avg = np.mean(spk_sims)
    plt.axhline(avg, color='red', linestyle='--', linewidth=2,
                label=f"Average = {avg:.4f}")
    
    plt.xticks(x, utt_ids, rotation=45, ha="right")
    plt.ylabel("Cosine similarity")
    plt.title("ECAPA speaker similarity (original vs generated)")
    
    # >>> SMART ADAPTIVE Y-LIMITS <<<
    
    min_val = min(spk_sims)
    max_val = max(spk_sims)
    
    if max_val - min_val < 0.05:
        # Values are extremely similar (almost all 1.0)
        padding = 0.01
        plt.ylim(min_val - padding, max_val + padding)
    else:
        # Values have normal spread
        padding = 0.05 * (max_val - min_val)
        plt.ylim(min_val - padding, max_val + padding)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("ecapa_similarity_per_utterance.png")
    plt.close()



########################################
# MAIN 
########################################

def main():
    print("Loading reference transcripts...")
    utt2text = load_all_transcripts(ORIG_TRANS_PATH)
    print(f"Found {len(utt2text)} utterances in transcript file.")

    print("Loading ASR model (Wav2Vec2)...")
    asr_model, asr_labels, asr_sr = load_asr_model()

    all_ids = []
    all_orig_wers = []
    all_gen_wers = []
    all_spk_sims = []

    # Loop over all IDs in the transcript file
    for utt_id, text in utt2text.items():
        # Assume audio filenames are <utt_id>.flac in each folder.
        orig_audio_path = os.path.join(ORIG_FOLDER, f"{utt_id}.flac")
        gen_audio_path  = os.path.join(GEN_FOLDER,  f"{utt_id}.flac")

        if not os.path.exists(orig_audio_path):
            print(f"[WARN] Original audio missing for {utt_id}: {orig_audio_path}")
            continue
        if not os.path.exists(gen_audio_path):
            print(f"[WARN] Generated audio missing for {utt_id}: {gen_audio_path}")
            continue

        print("\n========================================")
        print(f"Processing {utt_id}")
        print(f"Reference text: {text}")
        print("========================================")

        # ECAPA speaker similarity
        print("Computing speaker similarity with ECAPA-TDNN...")
        spk_sim = compute_speaker_similarity_ecapa(orig_audio_path, gen_audio_path)
        print(f"ECAPA speaker similarity (cosine, -1 to 1, higher = more similar): {spk_sim:.4f}")

        # ASR + WER for original audio
        print("Transcribing original audio...")
        orig_asr = transcribe_audio(orig_audio_path, asr_model, asr_labels, asr_sr)
        print(f"ASR (original):  {orig_asr}")

        # ASR + WER for generated audio
        print("Transcribing generated audio...")
        gen_asr = transcribe_audio(gen_audio_path, asr_model, asr_labels, asr_sr)
        print(f"ASR (generated): {gen_asr}")

        orig_wer_val = compute_wer(text, orig_asr)
        gen_wer_val  = compute_wer(text, gen_asr)

        print(f"WER (original vs ref):  {orig_wer_val:.4f}")
        print(f"WER (generated vs ref): {gen_wer_val:.4f}")

        all_ids.append(utt_id)
        all_orig_wers.append(orig_wer_val)
        all_gen_wers.append(gen_wer_val)
        all_spk_sims.append(spk_sim)

    # After all samples
    if all_ids:
        print("\n========================================")
        print("Overall statistics")
        print("========================================")
        print(f"Number of evaluated utterances: {len(all_ids)}")
        print(f"Average WER (original):  {np.mean(all_orig_wers):.4f}")
        print(f"Average WER (generated): {np.mean(all_gen_wers):.4f}")
        print(f"Average ECAPA similarity: {np.mean(all_spk_sims):.4f}")

        print("\nGenerating plots...")
        plot_results(all_ids, all_orig_wers, all_gen_wers, all_spk_sims)
        print("Saved 'wer_per_utterance.png' and 'ecapa_similarity_per_utterance.png'")
    else:
        print("No utterances were processed. Check that your audio files exist and IDs match the transcript.")


if __name__ == "__main__":
    main()

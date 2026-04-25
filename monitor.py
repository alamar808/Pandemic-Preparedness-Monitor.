# -*- coding: utf-8 -*-
import torch
import argparse
import sys
import numpy as np
import os
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

class PandemicPreparednessMonitor:
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D"):
        print("[*] Initialisiere Pandemic Preparedness Monitor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = EsmModel.from_pretrained(model_name).to(self.device).half()
            self.model.eval()
        except Exception as e:
            print("[!] Fehler: {}".format(e))
            sys.exit(1)
        
        print("[*] System bereit auf Gerät: {}".format(self.device.type.upper()))

    def analyze_batch(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()

def main():
    parser = argparse.ArgumentParser(description="AIxBio Hackathon: RAM-Optimized Monitor")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=500, help="Speichere alle X Batches")
    args = parser.parse_args()

    monitor = PandemicPreparednessMonitor()

    print("[*] Lade und dedupliziere Sequenzen...")
    with open(args.file, "r") as f:
        unique_sequences = list(set(line.strip().upper() for line in f if line.strip()))
    
    total_seqs = len(unique_sequences)
    print("[*] Einzigartige Sequenzen: {}".format(total_seqs))

    current_embeddings = []
    chunk_count = 0
    
    # Ordner für Zwischenergebnisse erstellen
    os.makedirs("chunks", exist_ok=True)

    print("\n--- Scanning with RAM-Protection ---")
    for i in tqdm(range(0, total_seqs, args.batch_size)):
        batch = unique_sequences[i : i + args.batch_size]
        emb = monitor.analyze_batch(batch)
        current_embeddings.append(emb)

        # Wenn X Batches erreicht sind -> Auf Festplatte schreiben
        if len(current_embeddings) >= args.save_every:
            chunk_file = "chunks/emb_chunk_{:03d}.npy".format(chunk_count)
            np.save(chunk_file, np.vstack(current_embeddings))
            
            # Index für diesen Chunk speichern
            idx_start = i - (len(current_embeddings) - 1) * args.batch_size
            idx_end = i + len(batch)
            with open("chunks/index_chunk_{:03d}.txt".format(chunk_count), "w") as f:
                for s in unique_sequences[idx_start:idx_end]:
                    f.write(s + "\n")
            
            current_embeddings = [] # RAM leeren
            chunk_count += 1

    # Restliche Daten am Ende speichern
    if current_embeddings:
        np.save("chunks/emb_chunk_final.npy", np.vstack(current_embeddings))

    print("\n[*] Fertig! Alle Chunks liegen im Ordner 'chunks/'.")

if __name__ == "__main__":
    main()

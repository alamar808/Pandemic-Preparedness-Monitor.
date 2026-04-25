from Bio import SeqIO
import sys

def translate_6_frames(dna_seq):
    frames = []
    # Vorwärts: Frames 0, 1, 2
    for i in range(3):
        frames.append(dna_seq[i:].translate(to_stop=True))
    # Rückwärts-Komplement: Frames 0, 1, 2
    rc_seq = dna_seq.reverse_complement()
    for i in range(3):
        frames.append(rc_seq[i:].translate(to_stop=True))
    return frames

# --- KONFIGURATION ---
input_file = "FASTA/SRR37006657_chunk_00.fasta"
output_file = "FASTA/SRR37006657_proteins_00.txt"
LIMIT = 10000000  # <--- Hier einstellen, wie viele FASTA-Einträge verarbeitet werden sollen
# ----------------------

seen_proteins = set()
processed_count = 0

print(f"[*] Starte Uebersetzung von max. {LIMIT} Datensaetzen...")

try:
    with open(output_file, "w") as out_f:
        # SeqIO.parse liest die Datei Sequenz für Sequenz (speicherschonend)
        for record in SeqIO.parse(input_file, "fasta"):
            
            # 6-Frame Translation durchführen
            protein_frames = translate_6_frames(record.seq)
            
            for protein in protein_frames:
                prot_str = str(protein)
                # Filter: Nur Proteine > 30 Aminosäuren und keine Duplikate
                if len(prot_str) > 30 and prot_str not in seen_proteins:
                    out_f.write(prot_str + "\n")
                    seen_proteins.add(prot_str)
            
            processed_count += 1
            
            # Fortschritt alle 1000 Records anzeigen
            if processed_count % 1000 == 0:
                print(f"[>] {processed_count} DNA-Sequenzen verarbeitet...")

            # Stopp-Bedingung
            if processed_count >= LIMIT:
                print(f"[!] Limit von {LIMIT} erreicht. Beende Uebersetzung.")
                break

except Exception as e:
    print(f"[!] Abbruch durch Fehler: {e}")

print(f"[*] Fertig! {len(seen_proteins)} einzigartige Proteine extrahiert.")

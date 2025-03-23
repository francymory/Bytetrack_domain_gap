import json

print('start')

# Percorso al tuo file di annotazione JSON
annotations_file = '/homes/fmorandi/stage_bytetrack/ByteTrack/datasets/motsynth/annotations/MOTSynth_annotations_10_train.json'

# Carica il file JSON
with open(annotations_file, 'r') as f:
    annotations = json.load(f)
    print("loaded")

# Funzione per stampare ricorsivamente la struttura del JSON con un limite di profondità
def print_json_structure(data, indent=0, max_depth=3):
    """
    Funzione che stampa la struttura del JSON, mostrandone le chiavi e i valori
    solo fino a un certo livello di profondità.
    """
    if max_depth == 0:
        return
    spacing = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}:")
            print_json_structure(value, indent + 2, max_depth - 1)
    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{spacing}List of {len(data)} items:")
            print_json_structure(data[0], indent + 2, max_depth - 1)
        else:
            print(f"{spacing}Empty List")
    else:
        print(f"{spacing}{data}")

# Stampa la struttura del JSON (limita la profondità a 3 livelli)
print("Struttura del JSON:")
print_json_structure(annotations, max_depth=3)

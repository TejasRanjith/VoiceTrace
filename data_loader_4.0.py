import os
import glob
import string
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from ml2en import ml2en 

# Import your model
from my_model import LipFormer

# --- 1. Configuration ---
CONFIG = {
    "data": {
        "landmarks": r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Extracted_landmarks_model_ready\\",
        "lip_rois": r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Extracted_lip_crosssection\\",
        "transcripts": r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Transcripts\\",
        "vocab_cache": "malayalam_vocab.json" # New: Cache file
    },
    "checkpoint_dir": "checkpoints",
    "epochs": 100,
    "batch_size": 1, 
    "learning_rate": 1e-5,
    "teacher_forcing_ratio": 0.5,
    "lambda_val": 0.7,
    "image_size": (80, 160),
    "validation_split": 0.1, 
}

# --- 2. Vocabulary Definitions ---

# --- Vocabulary for Manglish ---
MANGLISH_PAD_TOKEN = 0
MANGLISH_SOS_TOKEN = 1
MANGLISH_EOS_TOKEN = 2
MANGLISH_UNK_TOKEN = 3
MANGLISH_CHARS = string.ascii_lowercase + " .'-"
manglish_to_int = {char: i + 4 for i, char in enumerate(MANGLISH_CHARS)}
manglish_to_int["<pad>"] = MANGLISH_PAD_TOKEN
manglish_to_int["<sos>"] = MANGLISH_SOS_TOKEN
manglish_to_int["<eos>"] = MANGLISH_EOS_TOKEN
manglish_to_int["<unk>"] = MANGLISH_UNK_TOKEN
int_to_manglish = {i: char for char, i in manglish_to_int.items()}
MANGLISH_VOCAB_SIZE = len(manglish_to_int)

# --- Vocabulary for Malayalam ---
MALAYALAM_PAD_TOKEN = 0
MALAYALAM_SOS_TOKEN = 1
MALAYALAM_EOS_TOKEN = 2
MALAYALAM_UNK_TOKEN = 3

malayalam_to_int_map= {'<pad>': 0,
 '<sos>': 1,
 '<eos>': 2,
 '<unk>': 3,
 ' ': 4,
 '"': 5,
 '-': 6,
 '.': 7,
 'ം': 8,
 'ഃ': 9,
 'അ': 10,
 'ആ': 11,
 'ഇ': 12,
 'ഈ': 13,
 'ഉ': 14,
 'ഊ': 15,
 'ഋ': 16,
 'എ': 17,
 'ഏ': 18,
 'ഐ': 19,
 'ഒ': 20,
 'ഓ': 21,
 'ഔ': 22,
 'ക': 23,
 'ഖ': 24,
 'ഗ': 25,
 'ഘ': 26,
 'ങ': 27,
 'ച': 28,
 'ഛ': 29,
 'ജ': 30,
 'ഞ': 31,
 'ട': 32,
 'ഠ': 33,
 'ഡ': 34,
 'ണ': 35,
 'ത': 36,
 'ഥ': 37,
 'ദ': 38,
 'ധ': 39,
 'ന': 40,
 'പ': 41,
 'ഫ': 42,
 'ബ': 43,
 'ഭ': 44,
 'മ': 45,
 'യ': 46,
 'ര': 47,
 'റ': 48,
 'ല': 49,
 'ള': 50,
 'ഴ': 51,
 'വ': 52,
 'ശ': 53,
 'ഷ': 54,
 'സ': 55,
 'ഹ': 56,
 'ാ': 57,
 'ി': 58,
 'ീ': 59,
 'ു': 60,
 'ൂ': 61,
 'ൃ': 62,
 'െ': 63,
 'േ': 64,
 'ൈ': 65,
 'ൊ': 66,
 'ോ': 67,
 'ൌ': 68,
 '്': 69,
 'ൗ': 70,
 'ൺ': 71,
 'ൻ': 72,
 'ർ': 73,
 'ൽ': 74,
 'ൾ': 75,
 '瑞': 76,
 '阿': 77}

int_to_malayalam_map = {i: char for char, i in malayalam_to_int_map.items()}
MALAYALAM_VOCAB_SIZE = len(malayalam_to_int_map)

# def build_malayalam_vocab(transcript_dir, cache_file):
#     """
#     Builds vocab or loads from cache to save time.
#     """
#     # 1. Try to load from cache
#     if os.path.exists(cache_file):
#         print(f"Loading Malayalam vocabulary from cache: {cache_file}")
#         with open(cache_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             return data['vocab_size'], data['char_to_int'], {int(k): v for k, v in data['int_to_char'].items()}

#     # 2. If no cache, build from scratch
#     print("Building Malayalam vocabulary from scratch (this may take a while)...")
#     malayalam_to_int = {
#         "<pad>": MALAYALAM_PAD_TOKEN,
#         "<sos>": MALAYALAM_SOS_TOKEN,
#         "<eos>": MALAYALAM_EOS_TOKEN,
#         "<unk>": MALAYALAM_UNK_TOKEN,
#     }

#     vocab = set()
#     transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
    
#     for file_path in tqdm(transcript_files, desc="Scanning Transcripts"):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = [line.strip().split() for line in f.readlines()]
#         # Extract text robustly
#         full_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
#         vocab.update(list(full_text))
    
#     for i, char in enumerate(sorted(list(vocab))):
#         malayalam_to_int[char] = i + 4
        
#     int_to_malayalam = {i: char for char, i in malayalam_to_int.items()}
#     vocab_size = len(malayalam_to_int)

#     # 3. Save to cache
#     with open(cache_file, 'w', encoding='utf-8') as f:
#         json.dump({
#             'vocab_size': vocab_size,
#             'char_to_int': malayalam_to_int,
#             'int_to_char': int_to_malayalam
#         }, f, ensure_ascii=False, indent=4)
    
#     print(f"Vocabulary saved to {cache_file}")
#     return vocab_size, malayalam_to_int, int_to_malayalam

# --- 3. Custom PyTorch Dataset ---
class LipReadingDataset(Dataset):
    def __init__(self, landmark_dir, lip_roi_dir, transcript_dir, img_size, mal_to_int_map, man_to_int_map):
        self.img_size = img_size
        self.samples = []
        self.mal_to_int = mal_to_int_map
        self.man_to_int = man_to_int_map

        print("Searching for data samples...")
        landmark_files = sorted(glob.glob(os.path.join(landmark_dir, "*.npy")))
        
        for landmark_path in tqdm(landmark_files, desc="Matching data files"):
            base_name = os.path.basename(landmark_path).replace("_landmarks.npy", "")
            roi_dir = os.path.join(lip_roi_dir, base_name)
            transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")

            if os.path.isdir(roi_dir) and os.path.exists(transcript_path):
                self.samples.append({
                    "landmarks": landmark_path,
                    "rois": roi_dir,
                    "transcript": transcript_path,
                })
        
        print(f"Found {len(self.samples)} complete data samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            landmarks_np = np.load(sample["landmarks"])
            if landmarks_np.shape[0] == 0: return None
            landmarks = torch.from_numpy(landmarks_np).float()
        except Exception:
            return None

        roi_paths = sorted(glob.glob(os.path.join(sample["rois"], "*.png")))
        
        if not roi_paths or len(roi_paths) != landmarks.shape[0]:
            return None

        frames = []
        for frame_path in roi_paths:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                frames.append(frame / 255.0)
        video_tensor = torch.from_numpy(np.array(frames)).float().unsqueeze(0)

        with open(sample["transcript"], 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        malayalam_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
        
        manglish_text = ml2en.transliterate(malayalam_text).lower()

        # Create FULL sequences [SOS, ... , EOS]
        # We will slice them inside the training loop
        mal_tokens = [MALAYALAM_SOS_TOKEN] + [self.mal_to_int.get(c, MALAYALAM_UNK_TOKEN) for c in malayalam_text] + [MALAYALAM_EOS_TOKEN]
        mal_label = torch.tensor(mal_tokens, dtype=torch.long)

        man_tokens = [MANGLISH_SOS_TOKEN] + [self.man_to_int.get(c, MANGLISH_UNK_TOKEN) for c in manglish_text] + [MANGLISH_EOS_TOKEN]
        man_label = torch.tensor(man_tokens, dtype=torch.long)

        return {"video": video_tensor, "landmarks": landmarks, "malayalam_label": mal_label, "manglish_label": man_label}

def collate_fn(batch):
    """
    Custom collate function to handle variable length videos and landmarks.
    Implements 'Replication Padding' for landmarks to prevent gradient spikes.
    """
    # 1. Filter out None values from failed loads in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 2. Extract components
    videos = [item['video'] for item in batch]         # List of [1, T, H, W]
    landmarks = [item['landmarks'] for item in batch]  # List of [T, 37, 2]
    mal_labels = [item['malayalam_label'] for item in batch]
    man_labels = [item['manglish_label'] for item in batch]

    # 3. Pad Videos (Standard Zero Padding)
    # Input video shape is [C, T, H, W]. pad_sequence expects [T, ...]
    # We permute to [T, C, H, W], pad, then permute back to [B, C, T, H, W]
    padded_videos = pad_sequence(
        [v.permute(1, 0, 2, 3) for v in videos], 
        batch_first=True, 
        padding_value=0
    ).permute(0, 2, 1, 3, 4)

    # 4. Pad Landmarks (Replication Padding - THE MODIFICATION)
    # We repeat the last valid frame to fill the gap.
    # This ensures the "difference" between frames becomes 0, meaning "no movement".
    max_len = max([l.size(0) for l in landmarks])
    padded_landmarks_list = []
    
    for l in landmarks:
        n_frames = l.size(0)
        diff = max_len - n_frames
        
        if diff > 0:
            # Take the last frame: Shape [37, 2] -> Unsqueeze to [1, 37, 2]
            last_frame = l[-1].unsqueeze(0) 
            # Repeat it 'diff' times: Shape [diff, 37, 2]
            padding = last_frame.repeat(diff, 1, 1) 
            # Concatenate original + padding
            padded_l = torch.cat([l, padding], dim=0)
        else:
            padded_l = l
            
        padded_landmarks_list.append(padded_l)

    padded_landmarks = torch.stack(padded_landmarks_list)

    # 5. Pad Labels (Standard Pad Token)
    padded_mal_labels = pad_sequence(mal_labels, batch_first=True, padding_value=MALAYALAM_PAD_TOKEN)
    padded_man_labels = pad_sequence(man_labels, batch_first=True, padding_value=MANGLISH_PAD_TOKEN)
    
    return {
        "video": padded_videos, 
        "landmarks": padded_landmarks, 
        "malayalam_label": padded_mal_labels, 
        "manglish_label": padded_man_labels
    }

def tokens_to_text(token_indices, int_to_vocab_map):
    text = ""
    for token_idx in token_indices:
        idx = token_idx.item()
        char = int_to_vocab_map.get(idx)
        if char in ["<eos>", "<pad>"]: break
        if char == "<sos>": continue
        if char is not None: text += char
    return text

# --- 4. Main Training & Validation Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Load/Build Vocab
    # MALAYALAM_VOCAB_SIZE, malayalam_to_int_map, int_to_malayalam_map = build_malayalam_vocab(
    #     CONFIG["data"]["transcripts"], 
    #     CONFIG["data"]["vocab_cache"]
    # )
    
    dataset = LipReadingDataset(
        CONFIG["data"]["landmarks"], 
        CONFIG["data"]["lip_rois"], 
        CONFIG["data"]["transcripts"], 
        CONFIG["image_size"],
        malayalam_to_int_map,
        manglish_to_int 
    )

    if len(dataset) == 0:
        print("No data found.")
        return

    val_size = int(CONFIG["validation_split"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = LipFormer(
        num_manglishs=MANGLISH_VOCAB_SIZE,
        num_chars=MALAYALAM_VOCAB_SIZE,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    manglish_loss_fn = nn.CrossEntropyLoss(ignore_index=MANGLISH_PAD_TOKEN)
    char_loss_fn = nn.CrossEntropyLoss(ignore_index=MALAYALAM_PAD_TOKEN)
    scaler = torch.amp.GradScaler('cuda')

    # Load Checkpoint
    best_checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "lipformer_epoch_17.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(best_checkpoint_path):
        print(f"Resuming from {best_checkpoint_path}...")
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0)
        except Exception as e:
            print(f"Checkpoint error: {e}")

    # --- EPOCH LOOP ---
    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        
        for batch in train_bar:
            if batch is None: continue
            
            videos, landmarks, mal_labels, man_labels = [d.to(device) for d in batch.values()]
            
            # --- FIX: INPUT vs TARGET SLICING ---
            # Input: [SOS, A, B] (Slice: 0 to -1)
            # Target: [A, B, EOS] (Slice: 1 to End)
            
            man_input = man_labels[:, :-1]
            man_target = man_labels[:, 1:]
            
            mal_input = mal_labels[:, :-1]
            mal_target = mal_labels[:, 1:]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Pass INPUTS to the model (for teacher forcing)
                manglish_preds, char_preds = model(
                    videos, 
                    landmarks, 
                    manglish_targets=man_input, 
                    char_targets=mal_input, 
                    teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"]
                )
                
                # Calculate Loss against TARGETS
                loss_manglish = manglish_loss_fn(manglish_preds.reshape(-1, MANGLISH_VOCAB_SIZE), man_target.reshape(-1))
                loss_char = char_loss_fn(char_preds.reshape(-1, MALAYALAM_VOCAB_SIZE), mal_target.reshape(-1))
                total_loss = CONFIG["lambda_val"] * loss_manglish + (1 - CONFIG["lambda_val"]) * loss_char
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            
            # Cleanup
            del videos, landmarks, man_input, mal_input, manglish_preds, char_preds

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        outputs_shown = 0 
        
        print(f"\n--- Epoch {epoch+1} Validation ---")
        val_bar = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for batch in val_bar:
                if batch is None: continue

                videos, landmarks, mal_labels, man_labels = [d.to(device) for d in batch.values()]
                
                # Validation Target (remove SOS)
                man_target_val = man_labels[:, 1:]
                mal_target_val = mal_labels[:, 1:]

                # Prediction (No teacher forcing)
                manglish_preds, char_preds = model(videos, landmarks, manglish_targets=None, char_targets=None, teacher_forcing_ratio=0.0)

                # --- FIX: ALIGN LENGTHS for LOSS ---
                def align_for_loss(preds, targets, pad_idx):
                    B, T_pred, C = preds.shape
                    _, T_tgt = targets.shape
                    
                    if T_pred > T_tgt:
                        # Pad target if prediction is longer
                        padding = torch.full((B, T_pred - T_tgt), pad_idx, device=targets.device, dtype=torch.long)
                        targets = torch.cat([targets, padding], dim=1)
                    elif T_pred < T_tgt:
                        # Pad prediction? No, slice target (model stopped early, we penalize based on what it missed)
                        # Actually, correct is to pad prediction with LOW confidence or just slice target to match 
                        # To keep it simple and robust: we slice the target to match prediction length
                        # (Ideally we should penalize early stopping, but this prevents crash)
                        targets = targets[:, :T_pred]
                    return preds, targets

                p_preds, p_targs = align_for_loss(manglish_preds, man_target_val, MANGLISH_PAD_TOKEN)
                c_preds, c_targs = align_for_loss(char_preds, mal_target_val, MALAYALAM_PAD_TOKEN)

                loss_manglish = manglish_loss_fn(p_preds.reshape(-1, MANGLISH_VOCAB_SIZE), p_targs.reshape(-1))
                loss_char = char_loss_fn(c_preds.reshape(-1, MALAYALAM_VOCAB_SIZE), c_targs.reshape(-1))
                total_loss = CONFIG["lambda_val"] * loss_manglish + (1 - CONFIG["lambda_val"]) * loss_char
                val_loss += total_loss.item()

                if outputs_shown < 3:
                    pred_man = tokens_to_text(torch.argmax(manglish_preds, dim=2)[0], int_to_manglish)
                    pred_mal = tokens_to_text(torch.argmax(char_preds, dim=2)[0], int_to_malayalam_map)
                    true_man = tokens_to_text(man_labels[0], int_to_manglish) # Use original label for printing
                    true_mal = tokens_to_text(mal_labels[0], int_to_malayalam_map)
                    
                    print(f"\nExample {outputs_shown+1}:")
                    print(f"Manglish: {true_man} -> {pred_man}")
                    print(f"Malayalam: {true_mal} -> {pred_mal}")
                    outputs_shown += 1

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_checkpoint_path)
            print("Saved Best Model.")

        if (epoch + 1) % 2 != 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()
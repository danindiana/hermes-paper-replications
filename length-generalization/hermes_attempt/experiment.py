#!/usr/bin/env python3

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Set random seeds for reproducibility
SEEDS = [42, 123]

class PositionalEncoding(nn.Module):
    """Learned absolute positional encoding"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self.seq_len = torch.arange(max_len, dtype=torch.float)
    
    def forward(self, x):
        seq_len = x.size(1)
        freqs = torch.einsum("i,j->ij", self.seq_len[:seq_len], self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = torch.repeat_interleave(emb[:, None, :], x.shape[0], dim=0)
        return self.apply_rope(x, emb)
    
    def apply_rope(self, x, emb):
        cos_pos = emb.cos()[..., None]
        sin_pos = emb.sin()[..., None]
        x2 = torch.stack([x[..., 0::2], x[..., 1::2]], dim=-1)
        roped = torch.cat([
            x2[..., 0] * cos_pos - x2[..., 1] * sin_pos,
            x2[..., 1] * cos_pos + x2[..., 0] * sin_pos
        ], dim=-1)
        return torch.flatten(roped, -2, -1)

class TransformerModel(nn.Module):
    """Small causal decoder-only Transformer"""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, pos_encoding_type='learned'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if pos_encoding_type == 'learned':
            self.pos_encoding = PositionalEncoding(d_model)
        elif pos_encoding_type == 'rope':
            self.pos_encoding = RotaryPositionalEncoding(d_model)
        else:
            self.pos_encoding = None
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True,
                                                         norm_first=True, enable_nested_tensor=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # Embed and add positional encoding
        src_embed = self.embedding(src)
        if self.pos_encoding:
            src_embed = self.pos_encoding(src_embed)
        
        # Teacher forcing
        if tgt is None:
            tgt = src
            output = torch.zeros_like(src)
        else:
            output = torch.zeros_like(tgt)
        
        for t in range(1, src.size(1)):
            if tgt is None:
                tgt_input = output[:, :t]
            else:
                tgt_input = tgt[:, :t]
            
            tgt_embed = self.embedding(tgt_input)
            if self.pos_encoding:
                tgt_embed = self.pos_encoding(tgt_embed)
            
            # Transformer forward pass
            outputs = self.transformer(src_embed, tgt_embed)
            pred = self.fc_out(outputs[:, -1, :])
            
            if random.random() < teacher_forcing_ratio:
                output[:, t] = tgt[:, t]
            else:
                output[:, t] = torch.argmax(pred, dim=1)
        
        return self.fc_out(outputs) if tgt is not None else self.fc_out(outputs)

def create_task(name, length):
    """Generate training data for a task"""
    if name == 'copy':
        # Unique tokens for copy
        tokens = random.choices(range(10), k=length)
        input_seq = tokens + [11]  # 11 is separator
        target_seq = tokens + [11]
    elif name == 'reverse':
        tokens = random.choices(range(10), k=length)
        input_seq = tokens + [11]
        target_seq = list(reversed(tokens)) + [11]
    elif name == 'sort':
        tokens = random.sample(range(10), length)
        input_seq = tokens + [11]
        target_seq = sorted(tokens) + [11]
    else:
        raise ValueError(f"Unknown task: {name}")
    
    return input_seq, target_seq

class TaskDataset(Dataset):
    """Dataset for a single task"""
    def __init__(self, task_name, min_len=1, max_len=20, num_samples=1000):
        self.task_name = task_name
        self.min_len = min_len
        self.max_len = max_len
        self.num_samples = num_samples
        self.data = []
        self._generate_data()
    
    def _generate_data(self):
        for _ in range(self.num_samples):
            length = random.randint(self.min_len, self.max_len)
            input_seq, target_seq = create_task(self.task_name, length)
            self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TokenPadCollate:
    """Collate function that pads sequences to max length in batch"""
    def __call__(self, batch):
        # Separate inputs and targets
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Find max length
        max_len = max(max(len(inp), len(tgt)) for inp, tgt in zip(inputs, targets))
        
        # Pad inputs and targets
        pad_val = 12  # Use separator token as padding
        padded_inputs = [inp + [pad_val] * (max_len - len(inp)) for inp in inputs]
        padded_targets = [tgt + [pad_val] * (max_len - len(tgt)) for tgt in targets]
        
        return torch.LongTensor(padded_inputs), torch.LongTensor(padded_targets)

def evaluate_model(model, task_name, eval_lengths, vocab_size, num_samples=100, device='cpu'):
    """Evaluate model on specific lengths"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for L in eval_lengths:
            correct = 0
            for _ in range(num_samples):
                input_seq, target_seq = create_task(task_name, L)
                input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(device)
                target_tensor = torch.LongTensor(target_seq).unsqueeze(0).to(device)
                
                output = model(input_tensor, target_tensor, teacher_forcing_ratio=0)
                _, preds = torch.max(output, dim=2)
                
                # Calculate exact match, ignoring padding
                true_seq = target_tensor[0, 1:L+1]  # Skip first token, take actual length
                pred_seq = preds[0, 1:L+1]
                matches = true_seq.eq(pred_seq)
                correct += int(matches.all().item())
            
            results[L] = correct / num_samples
    
    return results

def train_model(task_name, pos_encoding_type, seed, device='cpu'):
    """Train a model on a task"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create dataset
    dataset = TaskDataset(task_name, min_len=1, max_len=20, num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=TokenPadCollate())
    
    vocab_size = 13  # 0-9 digits + separator + pad
    model = TransformerModel(vocab_size, d_model=128, nhead=4, num_layers=4, 
                             pos_encoding_type=pos_encoding_type).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(20):
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs, targets, teacher_forcing_ratio=0.5 if epoch < 10 else 0.0)
            
            # Calculate loss only on non-padded tokens
            loss = criterion(output.view(-1, vocab_size), targets[:, 1:].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

def run_experiment():
    """Run full experiment"""
    results = {}
    
    # Train and evaluate for each task
    tasks = ['copy', 'reverse', 'sort']
    pos_encodings = ['learned', 'rope']
    
    for task in tasks:
        for pos_type in pos_encodings:
            for seed in SEEDS:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                print(f"\n=== Task: {task}, Pos: {pos_type}, Seed: {seed} ===")
                model = train_model(task, pos_type, seed, device)
                
                # Evaluate on train lengths
                train_lengths = list(range(1, 21))
                eval_lengths = list(range(1, 61))
                
                train_acc = evaluate_model(model, task, train_lengths, 12, num_samples=50, device=device)
                test_acc = evaluate_model(model, task, eval_lengths, 12, num_samples=50, device=device)
                
                results[f"{task}_{pos_type}_seed{seed}"] = {
                    'train_lengths': train_lengths,
                    'test_lengths': eval_lengths,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc
                }
    
    # Save results
    with open('/workspace/lengthgen/result.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Experiment complete! ===")
    print(f"Results saved to result.json")
    return results

if __name__ == '__main__':
    results = run_experiment()
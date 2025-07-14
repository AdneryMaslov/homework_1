import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional
from tokenizers import Tokenizer

from transformer_basics.layers import PositionalEncoding, FeedForward, MultiheadAttention


class DecoderOnlyLayer(nn.Module):
    """
    Слой декодера без кросс-внимания (cross-attention).
    Это ключевое отличие от `DecoderLayer` из твоего урока.
    """
    def __init__(self, d_model: int, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attention = deepcopy(mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Первый блок: self-attention с остаточным соединением и нормализацией
        x_norm = self.layernorm1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)

        # Второй блок: feed-forward с остаточным соединением и нормализацией
        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x

class GeneratorTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        self.d_model = config['d_model']
        self.max_length = config['max_length']
        self.device = config['device']
        
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')
        self.eos_token_id = self.tokenizer.token_to_id('</s>')

        # Слои модели
        self.embedding = nn.Embedding(tokenizer.get_vocab_size(), self.d_model, padding_idx=self.pad_token_id)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_length)
        
        mha = MultiheadAttention(self.d_model, config['nhead'])
        ffn = FeedForward(self.d_model, config['d_ff'])
        decoder_layer = DecoderOnlyLayer(self.d_model, mha, ffn, config['dropout'])
        
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(config['num_decoder_layers'])])
        
        self.output_layer = nn.Linear(self.d_model, tokenizer.get_vocab_size())

    def _get_causal_mask(self, seq_len):
        """Создает каузальную маску правильной формы."""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.unsqueeze(0).to(self.device)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size()
        
        causal_mask = self._get_causal_mask(seq_len)
        
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        logits = self.output_layer(x)
        return logits

    def generate(self, prompt: str, temperature=1.0, max_new_tokens=100):
        self.eval()
        with torch.no_grad():
            # Токенизация входного промпта
            input_ids = self.tokenizer.encode(prompt).ids
            # Сразу создаем тензор правильного типа torch.long
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            generated_ids = input_tensor

            for _ in range(max_new_tokens):
                # Обрезаем контекст
                # Явно приводим тип к torch.long на всякий случай перед подачей в модель
                context = generated_ids[:, -self.max_length+1:].long()
                
                outputs = self.forward(context)
                
                next_token_logits = outputs[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated_ids[0].tolist())

    def save_checkpoint(self, path):
        """Сохраняет модель и ее конфигурацию."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'tokenizer_path': 'transformer_basics/mistral_tokenizer.json' # Сохраним путь к токенизатору
        }
        torch.save(checkpoint, path)
        print(f"Модель сохранена в {path}")

    @classmethod
    def load_from_checkpoint(cls, path, device='cpu'):
        """Загружает модель из чекпоинта."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        config['device'] = device
        
        tokenizer = Tokenizer.from_file(checkpoint['tokenizer_path'])
        tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
        
        model = cls(config, tokenizer)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
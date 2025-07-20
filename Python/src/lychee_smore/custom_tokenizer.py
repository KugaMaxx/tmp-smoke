#!/usr/bin/env python
"""
Custom Tokenizer for special format data
Author: GitHub Copilot
"""

import re
import json
from typing import List, Optional, Union, Dict, Any
from transformers import CLIPTokenizer
import torch


class CustomTokenizer(CLIPTokenizer):
    """
    Custom tokenizer that handles special format data like "[HRR] 100; [T] 7; [HD] 20, 124, 20, 20, 81, 20"
    
    Tokenization rules:
    - Special tokens: "[HRR]", "[T]", "[HD]" with fixed IDs
    - Numbers are tokenized individually and encoded as themselves
    - Semicolons and commas are ignored during tokenization
    - Other rules remain consistent with CLIPTokenizer
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define special tokens for our custom format with fixed IDs starting from 2000
        self.special_format_tokens = {
            "[HRR]": 2000,
            "[T]": 2001,
            "[HD]": 2002
        }
        
        # Reverse mapping from ID to token
        self.special_id_to_token = {v: k for k, v in self.special_format_tokens.items()}
        
        # Create a mapping for numbers (they encode to themselves)
        self.number_tokens = {}
        
        # Add special format tokens to vocabulary if not present
        # Note: These tokens will use fixed IDs starting from 2000
        self._add_custom_tokens()
    
    def _add_custom_tokens(self):
        """Add custom tokens to the tokenizer vocabulary with fixed IDs"""
        # We don't need to add these to the actual vocabulary since we handle them
        # with fixed IDs in our conversion methods
        pass
    
    def _custom_tokenize(self, text: str) -> List[str]:
        """
        Custom tokenization for the special format
        
        Args:
            text: Input text like "[HRR] 100; [T] 7; [HD] 20, 124, 20, 20, 81, 20"
            
        Returns:
            List of tokens
        """
        # Pattern to match the special format
        # This pattern captures: [HRR], [T], [HD], numbers, ignoring semicolons and commas
        pattern = r'(\[HRR\]|\[T\]|\[HD\]|\d+)'
        
        tokens = []
        matches = re.findall(pattern, text)
        
        for match in matches:
            if match.isdigit():
                # Numbers encode to themselves
                tokens.append(match)
                # Store the number mapping
                self.number_tokens[match] = int(match)
            elif match in self.special_format_tokens:
                # Use the original token, not the mapped value
                tokens.append(match)
            else:
                # For any other content, use the parent tokenizer
                parent_tokens = super()._tokenize(match)
                tokens.extend(parent_tokens)
        
        return tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Override the tokenize method to handle our custom format
        """
        # Check if the text matches our special format pattern
        special_pattern = r'^\[HRR\]\s*\d+;\s*\[T\]\s*\d+;\s*\[HD\][\d,\s]+$'
        
        if re.match(special_pattern, text.strip()):
            return self._custom_tokenize(text)
        else:
            # For regular text, use the parent tokenizer
            return super()._tokenize(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert token to ID, handling numbers and special tokens specially
        """
        # If it's a number token, return the number itself as ID
        if token.isdigit():
            return int(token)
        
        # If it's one of our special format tokens, return the fixed ID
        if token in self.special_format_tokens:
            return self.special_format_tokens[token]
        
        # For other tokens, use the parent method
        return super()._convert_token_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert ID to token, handling numbers and special tokens specially
        """
        # If the index is one of our special format token IDs, return the token
        if index in self.special_id_to_token:
            return self.special_id_to_token[index]
            
        # If the index is a number that we've seen before, return it as string
        if str(index) in self.number_tokens:
            return str(index)
        
        # For other IDs, use the parent method
        return super()._convert_id_to_token(index)
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode text to token IDs
        """
        tokens = self._tokenize(text)
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for token in tokens:
            token_id = self._convert_token_to_id(token)
            ids.append(token_id)
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """
        Decode token IDs back to text
        """
        tokens = []
        
        for i, token_id in enumerate(token_ids):
            if skip_special_tokens and token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            
            token = self._convert_id_to_token(token_id)
            tokens.append(token)
            
            # Add appropriate separators based on position
            if i < len(token_ids) - 1:
                next_token_id = token_ids[i + 1]
                if not (skip_special_tokens and next_token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]):
                    next_token = self._convert_id_to_token(next_token_id)
                    
                    # Add space after [HRR], [T] and [HD]
                    if token in ["[HRR]", "[T]", "[HD]"]:
                        tokens.append(" ")
                    # Add semicolon and space after numbers that follow [HRR] or [T]
                    elif token.isdigit() and i > 0:
                        prev_token_id = token_ids[i - 1]
                        if not (skip_special_tokens and prev_token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]):
                            prev_token = self._convert_id_to_token(prev_token_id)
                            if prev_token in ["[HRR]", "[T]"]:
                                tokens.append("; ")
                            elif next_token.isdigit():  # Add comma between numbers in HD section
                                tokens.append(", ")
        
        # Join tokens back to text
        text = "".join(tokens)
        return text
    
    def tokenize_batch(self, texts: List[str], **kwargs) -> List[List[str]]:
        """
        Tokenize a batch of texts
        """
        return [self._tokenize(text) for text in texts]
    
    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """
        Encode a batch of texts
        """
        return [self.encode(text, **kwargs) for text in texts]


def test_custom_tokenizer():
    """Test function for the custom tokenizer"""
    print("Testing Custom Tokenizer...")
    
    # Initialize the tokenizer (using CLIP tokenizer as base)
    tokenizer = CustomTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test data
    test_texts = [
        "[HRR] 100; [T] 7; [HD] 20, 124, 20, 20, 81, 20",
        "[HRR] 200; [T] 15; [HD] 30, 256, 40, 35, 127, 50",
        "[HRR] 150; [T] 3; [HD] 10, 64, 15, 12, 45, 25"
    ]
    
    print("\nTesting tokenization:")
    for text in test_texts:
        print(f"\nOriginal text: {text}")
        
        # Tokenize
        tokens = tokenizer._tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Encode
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"Token IDs: {token_ids}")
        
        # Decode
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")
        
        # Verify numbers are preserved
        numbers_in_original = re.findall(r'\d+', text)
        numbers_in_ids = [str(id) for id in token_ids if str(id).isdigit()]
        print(f"Numbers preserved: {numbers_in_original == numbers_in_ids}")


if __name__ == "__main__":
    test_custom_tokenizer()

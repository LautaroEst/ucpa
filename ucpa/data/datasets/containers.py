
from torch.utils.data import Dataset
from ...models.utils import SUPPORTED_MODELS
import numpy as np

class BasicContainer(Dataset):

    def __init__(self, prompts, labels, sentences_ids):
        self._data = {
            "sentences": prompts,
            "labels": labels,
            "sentences_ids": sentences_ids
        }
        
    def __len__(self):
        return len(self._data["sentences"])
    
    def __getitem__(self, idx):
        return {
            "original_id": self._data["original_ids"][idx],
            "sentence": self._data["sentences"][idx], 
            "label": self._data["labels"][idx]
        }

class LanguageModelDataset(Dataset):

    def __init__(self, prompts, prompts_ids, model_name, random_state=None):
        if model_name in SUPPORTED_MODELS["decoder_only"]:
            self.mode = "decoder_only"
        elif model_name in SUPPORTED_MODELS["encoder_decoder"]:
            self.mode = "encoder_decoder"
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        self.prompts = prompts
        self.prompts_ids = prompts_ids
        self.rs = np.random.RandomState(random_state)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):

        if self.mode == "decoder_only":
            sample = {
                "original_text": self.prompts[idx],
                "src_text": self.prompts[idx],
                "tgt_text": self.prompts[idx],
                "text_id": self.prompts_ids[idx]
            }
        elif self.mode == "encoder_decoder":
            original_prompt = self.prompts[idx]
            tokenized_prompt = original_prompt.split(" ")
            sampled_index = self.rs.choice(len(tokenized_prompt), len(tokenized_prompt) * 0.15, replace=False)
            noisy_tokenized_prompt = []
            tgt_text = []
            counter = 0
            for i in range(len(tokenized_prompt)):
                if i not in sampled_index:
                    noisy_tokenized_prompt.append(tokenized_prompt[i])
                else:
                    if i-1 in sampled_index:
                        tgt_text.append(tokenized_prompt[i])
                    else:
                        noisy_tokenized_prompt.append(f"<extra_id_{counter}")
                        tgt_text.extend([f"<extra_id_{counter}", tokenized_prompt[i]])
                        counter += 1

            src_text = " ".join(noisy_tokenized_prompt)
            tgt_text.append(f"<extra_id_{counter}")
            tgt_text = " ".join(tgt_text)
            
            sample = {
                "original_text": self.prompts[idx],
                "src_text": src_text,
                "tgt_text": tgt_text,
                "text_id": self.prompts_ids[idx]
            }

        return sample
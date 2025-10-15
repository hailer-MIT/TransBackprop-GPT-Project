import torch

def pad_collate_fn(batch, pad_token_id):
    # print(f"[pad_collate_fn] Using pad_token_id: {pad_token_id}")
    # Pad input_ids and labels to the maximum length in the batch
    max_len_input = max(len(item['input_ids']) for item in batch)
    max_len_label = max(len(item['labels']) for item in batch)

    padded_input_ids = []
    padded_labels = []

    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']

        # Pad input_ids
        pad_length_input = max_len_input - len(input_ids)
        padded_input_ids.append(torch.cat([
            input_ids, 
            torch.full((pad_length_input,), pad_token_id, dtype=torch.long)
        ]))

        # Pad labels (use -1 for ignore_index in CrossEntropyLoss)
        pad_length_label = max_len_label - len(labels)
        padded_labels.append(torch.cat([
            labels, 
            torch.full((pad_length_label,), -1, dtype=torch.long)
        ]))

    return {
        'input_ids': torch.stack(padded_input_ids, dim=0),
        'labels': torch.stack(padded_labels, dim=0)
    }

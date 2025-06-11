from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import torch.nn.functional as F

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def answer_question(context, question, top_k=2, max_length=384, stride=128):
    """
    Perform question answering on the context using BERT with sliding window.
    Args:
        context (str): The context paragraph text.
        question (str): The question string.
        top_k (int): Number of top answers and top token probabilities to consider.
        max_length (int): Max token length for BERT input.
        stride (int): Sliding window stride for long context.

    Prints:
        Best answers with their probability scores, token spans and character spans.
        Top start and end token positions with their probabilities.
    """
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",            # Only truncate the context
        stride=stride,                       # Overlap for sliding window
        return_overflowing_tokens=True,     # Return multiple windows if needed
        return_offsets_mapping=True,         # Return token char offsets for mapping
        padding="max_length",                # Pad to max_length
        return_tensors="pt"                  # Return PyTorch tensors
    )
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    offset_mapping = inputs["offset_mapping"]

    all_candidates = []
    # Dictionaries to keep max probabilities for each token position across all windows
    start_prob_stats = {}
    end_prob_stats = {}

    # Process each sliding window
    for i in range(len(input_ids)):
        ids = input_ids[i].unsqueeze(0)              # Shape: [1, seq_len]
        mask = attention_mask[i].unsqueeze(0)
        token_type = token_type_ids[i].unsqueeze(0)
        offsets = offset_mapping[i]

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type)

        start_logits = outputs.start_logits[0]       # [seq_len]
        end_logits = outputs.end_logits[0]           # [seq_len]

        # Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        # Get top_k indices and probabilities for start and end tokens
        start_topk = torch.topk(start_probs, k=top_k)
        end_topk = torch.topk(end_probs, k=top_k)

        # Update max probability for each token position (aggregate over windows)
        for rank in range(top_k):
            sidx = start_topk.indices[rank].item()
            sprob = start_topk.values[rank].item()
            start_prob_stats[sidx] = max(start_prob_stats.get(sidx, 0), sprob)

            eidx = end_topk.indices[rank].item()
            eprob = end_topk.values[rank].item()
            end_prob_stats[eidx] = max(end_prob_stats.get(eidx, 0), eprob)

        candidates = []
        # Construct candidate answers by combining top_k start and end tokens
        for start_idx, start_prob in zip(start_topk.indices, start_topk.values):
            for end_idx, end_prob in zip(end_topk.indices, end_topk.values):
                s = start_idx.item()
                e = end_idx.item()
                # Valid answer span conditions
                if s <= e and (e - s + 1) <= 15:
                    score = (start_prob * end_prob).item()

                    token_span_ids = ids[0, s: e + 1]
                    # Decode token span to string
                    answer = tokenizer.decode(token_span_ids, skip_special_tokens=True).strip()

                    # Map token indices to original context character positions
                    char_start = offsets[s][0].item()
                    char_end = offsets[e][1].item()
                    answer_text_from_context = context[char_start:char_end]

                    candidates.append({
                        "score": score,
                        "answer": answer,
                        "start_token": s,
                        "end_token": e,
                        "char_start": char_start,
                        "char_end": char_end,
                        "answer_text_from_context": answer_text_from_context,
                    })
        all_candidates.extend(candidates)

    # Remove duplicates answers and keep top_k by score
    unique_answers = {}
    for cand in sorted(all_candidates, key=lambda x: x["score"], reverse=True):
        ans_text = cand["answer"]
        if ans_text not in unique_answers:
            unique_answers[ans_text] = cand
        if len(unique_answers) == top_k:
            break

    # Print final top answers
    print(f"\nQuestion: {question}\n")
    print("Final Top Answer Candidates:\n")
    for rank, cand in enumerate(unique_answers.values(), 1):
        print(f"{rank}. Answer: {cand['answer']}")
        print(f"   Score: {cand['score']:.6f}")
        print(f"   Start token index: {cand['start_token']}, End token index: {cand['end_token']}")
        print(f"   Character span in context: ({cand['char_start']}, {cand['char_end']})")
        print(f"   Text span from original context: \"{cand['answer_text_from_context']}\"")
        print()

    # Display summary of top start token positions and probabilities
    print("Summary of start token probabilities (top positions):")
    for pos, prob in sorted(start_prob_stats.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        token_str = tokenizer.convert_ids_to_tokens(input_ids[0][pos].item())
        print(f"  Position {pos}: Token '{token_str}' with max probability {prob:.4f}")

    # Display summary of top end token positions and probabilities
    print("Summary of end token probabilities (top positions):")
    for pos, prob in sorted(end_prob_stats.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        token_str = tokenizer.convert_ids_to_tokens(input_ids[0][pos].item())
        print(f"  Position {pos}: Token '{token_str}' with max probability {prob:.4f}")

# Sample context paragraph
context = """
Climate change refers to significant, long-term changes in the global climate system,
including temperature, precipitation, and wind patterns. While natural factors like volcanic eruptions
and solar radiation variations can cause climate changes, human activities have been the primary driver since the Industrial Revolution.
The burning of fossil fuels releases large amounts of greenhouse gases such as carbon dioxide into the atmosphere.
These gases accumulate and enhance the greenhouse effect - the process where certain atmospheric gases trap heat from the sun,
preventing it from escaping back into space. This excessive greenhouse effect leads to global warming,
causing sea level rise, extreme weather events, and ecosystem disruptions.
"""

print("Context loaded. You can ask questions about this text. (Type 'exit' to quit)")

while True:
    question = input("\nYour question: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    if not question:
        print("Please input a question.")
        continue

    answer_question(context, question, top_k=2)

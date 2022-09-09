import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class Inference:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

    def __call__(self, questions, contexts, impossible, top_k=1, max_ans_length=15):
        answers = []
        # infer for each question independently and collect answers
        for question, context in zip(questions, contexts):
            answer = self._infer(question, context, impossible, top_k, max_ans_length)
            answers.append(answer)
        return answers

    def _infer(self, question, context, impossible, top_k, max_ans_length):
        # tokenize question and context
        # might split up the context over multiple spans if max_length is exceeded
        # parameters are taken over from the model training
        enc = self.tokenizer(text=question, text_pair=context, max_length=386, stride=128, padding=True,
                             return_overflowing_tokens=True,
                             return_offsets_mapping=True,
                             return_special_tokens_mask=True,
                             truncation="only_second", return_tensors='pt')

        # get predicted start and end logits from the model
        with torch.no_grad():
            out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)

        # create mask for possible answers
        # exclude padding and other special tokens
        masks = enc.attention_mask & ~enc.special_tokens_mask
        i = 1
        while masks[0, i] == 1:
            masks[:, i] = 0
            i += 1

        # if question might be impossible to answer unmask the CLS token
        if impossible:
            masks[:, 0] = 1

        # calculate probabilities for all valid answers
        answers = []
        min_prob_impossible = 1.0
        for starts, ends, encoding, mask in zip(out.start_logits, out.end_logits, enc.encodings, masks):
            # set all invalid logits to negative infinity, so they have no effect on the softmax
            starts = torch.softmax(starts + torch.log(mask), dim=0)
            ends = torch.softmax(ends + torch.log(mask), dim=0)

            # get minimal probability for question impossible to answer over all spans
            min_prob_impossible = min(min_prob_impossible, starts[0] * ends[0])
            # mask CLS if not already done to search for actual answer
            starts[0], ends[0] = 0, 0

            # iterate through tokens to find possible start of the answer
            for s, start in enumerate(starts):
                # skip if start token is masked
                if start == 0:
                    continue

                # iterate through tokens from start token to find possible end of the answer
                for e in range(s, min(s + max_ans_length, len(ends))):
                    end = ends[e]
                    # skip if end token is masked
                    if end == 0:
                        continue

                    # calculate score and create new possible answer
                    score = (start * end).item()
                    pos_s = encoding.offsets[s][0]
                    pos_e = encoding.offsets[e][1]
                    answer = context[pos_s:pos_e]
                    answers.append({'score': score, 'start': pos_s, 'end': pos_e, 'answer': answer})

        # add 'impossible' as an answer if a valid answer is not enforced
        if impossible:
            answers.append({'score': min_prob_impossible.item(), 'start': 0, 'end': 0, 'answer': ''})

        # sort answers by score and return top_k candidates
        answers.sort(key=lambda a: a['score'], reverse=True)
        return answers[:top_k]

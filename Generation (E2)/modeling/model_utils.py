import torch
import numpy as np
from modeling.models import RNN, LSTM, GRU
from tqdm import tqdm

def get_fresh_initial_hidden_states(model, batch_size):
    device = next(model.parameters()).device
    if(isinstance(model, LSTM)):
        hidden_state = (
            torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device),
            torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
        )
    elif(isinstance(model, RNN) or isinstance(model, GRU)):
        hidden_state = torch.zeros(model.num_layers, batch_size, model.hidden_dim).to(device)
    else:
        raise Exception("Unknown model type")
    
    return hidden_state

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def perplexity(
        model, sentence_list,
        tokenizer = None,  # if None, assume sentence_list is already tokenized
        batch_size = 32,
        consider_length = 32,   # if the length is too large the perplexity will be infinite
        context_state = None,   # if None, start fresh for each sentence
    ):
    model.eval()
    device = next(model.parameters()).device
    if(tokenizer is not None):
        tokenized = tokenizer.tokenize([sentence_list], get_token_ids=True)
        token_ids = tokenized["token_ids"][0]
    else:
        token_ids = sentence_list

    perplexity_track = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(token_ids), batch_size)):
            batch_end = min(len(token_ids), batch_start + batch_size)
            cur_batch_size = batch_end - batch_start
            min_length = min(consider_length, np.array([len(t) for t in token_ids[batch_start : batch_end]]).min())

            batch = torch.stack(
                [torch.tensor(t[:min_length]) for t in token_ids[batch_start : batch_end]]
            ).to(device)

            if(context_state is None):
                hidden_state = get_fresh_initial_hidden_states(model, cur_batch_size)
            else:
                if(isinstance(model, LSTM)):
                    hidden_state = (
                        context_state[0].repeat(1, cur_batch_size, 1),
                        context_state[1].repeat(1, cur_batch_size, 1),
                    )
                else:
                    hidden_state = context_state.repeat(1, cur_batch_size, 1)

            log_true_proba = 0
            for i in range(min_length - 1):
                x = batch[:, i][None].T
                y_true = batch[:, i+1][None].T
                y_pred, hidden_state = model(x, hidden_state)
                cur_log_proba = 0
                for pred, true in zip(y_pred, y_true):
                    softmax = torch.nn.functional.softmax(pred.squeeze(), dim = 0)
                    # print(true[0].item(), softmax[true[0]].item(), torch.log(softmax[true[0]]).item())
                    cur_log_proba += torch.log(softmax[true[0]])
                cur_log_proba /= cur_batch_size
                # print(cur_log_proba.item())
                log_true_proba += cur_log_proba

                # hidden_state = hidden_state.detach()

            proba = torch.exp(log_true_proba.to(torch.float64))
            # print(f"log proba = {log_true_proba.item()}", f"p = {proba.item()}")
            perplexity_track.append(torch.pow(proba, -1/min_length))
    
    return torch.tensor(perplexity_track).mean().item()


def generate(
        model, tokenizer, 
        prompt="", max_new_tokens = 10, top_k = 5,
        debug = False,
        context_state = None,  # if None, start fresh for each sentence
    ):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        tokenized = tokenizer.tokenize([prompt], get_token_ids=True)
        token_ids = tokenized["token_ids"][0][:-1]  # remove the </s> token
        end_of_sent = tokenizer.word_index[tokenizer.SENT_END]

        if(context_state is None):
            # start fresh for each sentence
            hidden_state = get_fresh_initial_hidden_states(model, 1)
        else:
            hidden_state = context_state
        
        for p in token_ids:
            x = torch.tensor([[p]]).to(device)
            y_pred, hidden_state = model(x, hidden_state)
        
        last_token = token_ids[-1]
        generated_tokens = []
        for i in range(max_new_tokens):
            # print(hidden_state.norm().item())
            y_pred, hidden_state = model(
                torch.tensor([[last_token]]).to(device), hidden_state
            )
            y_pred = y_pred.squeeze()
            y_pred = torch.nn.functional.softmax(y_pred, dim = 0)
            y_pred = y_pred.cpu().detach().numpy()
            top_k_idx = y_pred.argsort()[-top_k:][::-1]
            top_k_values = y_pred[top_k_idx]

            if(debug):
                print([(tokenizer.index_word[t], np.round(v, 4)) for t, v in zip(top_k_idx, top_k_values)])
            
            top_k_probs = top_k_values / np.sum(top_k_values)
            sampled_idx = np.random.choice(top_k_idx, p = top_k_probs)
            token_ids.append(sampled_idx)
            last_token = sampled_idx
            generated_tokens.append(sampled_idx)
            if(sampled_idx == end_of_sent):
                break
    
    return {
        "text": tokenizer.decode([token_ids + generated_tokens])[0],
        "generated_tokens": generated_tokens,
    }
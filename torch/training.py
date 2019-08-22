import torch
import torch.nn as nn
import os
import time
import math
import sys


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    model = model.double()
    for i, batch in enumerate(iterator):
        src = batch['input']
        trg = batch['output']
        expert = batch['expert']

        src = src.view(src.size(1), src.size(0), src.size(2))
        trg = trg.view(trg.size(1), trg.size(0), trg.size(2))
        expert = expert.view(expert.size(1), expert.size(0), expert.size(2))
        #print('encoder', src.double().size())

        optimizer.zero_grad()

        output = model(src.double(), trg.double(), expert.double())

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output.view(-1, output.shape[-1]).double()
        trg = trg.view(-1).double()

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['input']
            trg = batch['output']
            expert = batch['expert']

            src = src.view(src.size(1), src.size(0), src.size(2))
            trg = trg.view(trg.size(1), trg.size(0), trg.size(2))
            expert = expert.view(expert.size(1), expert.size(0), expert.size(2))

            output = model(src.double(), trg.double(), expert.double())  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output.view(-1, output.shape[-1]).double()
            trg = trg.view(-1).double()

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_result(model, iterator, criterion, transform_flag, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['input']
            trg = batch['output']
            expert = batch['expert']
            mu = batch['mean']
            std = batch['std']
            mu = mu.to(device)
            std = std.to(device)

            src = src.view(src.size(1), src.size(0), src.size(2))
            trg = trg.view(trg.size(1), trg.size(0), trg.size(2))
            expert = expert.view(expert.size(1), expert.size(0), expert.size(2))

            output = model(src.double(), trg.double(), expert.double())  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output.view(-1, output.shape[-1]).double()
            trg = trg.view(-1).double()

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            if transform_flag == 'standard':
                output = output*std+mu
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), output, trg


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def start_training(SAVE_DIR, MODEL_SAVE_PATH, N_EPOCHS, model, train_iterator, valid_iterator, optimizer, criterion,
                   CLIP):
    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print('Validation loss improved saving model')

        print(
            f'| Epoch: {epoch + 1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Val.Loss: {valid_loss: .3f} |')

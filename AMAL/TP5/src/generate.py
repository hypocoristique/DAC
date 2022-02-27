import enum
from textloader import  string2code, id2lettre, code2string
import math
import torch
import  torch.nn.functional as F
from icecream import ic

#  TODO:  Ce fichier contient les différentes fonction de génération

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Only for RNN
def generate_greedy(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération greedy uniquement pour RNN simple (l'embedding et le decodeur peuvent être des fonctions du rnn).
         Initialise le réseau avec start (ou à 0 si start est vide)
            et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits
    # (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    coded_sentence = string2code(start).tolist()
    with torch.no_grad():
        x_emb = emb(torch.tensor(coded_sentence)).unsqueeze(1)
        H = rnn.forward(x_emb)
        h = H[-1] #On génère la séquence à partir du dernier état caché de la séquence fournie
        char = []
        char = torch.argmax(decoder(h))
        coded_sentence.append(char)
        while char != eos:
            x_emb = emb(torch.tensor(coded_sentence[-1]).unsqueeze(0))
            h = rnn.one_step(x_emb, h)
            char = torch.argmax(rnn.decode(h))
            coded_sentence.append(char)
            if len(coded_sentence) == maxlen:
                coded_sentence.append(eos)
                return code2string(torch.FloatTensor(coded_sentence))
    return code2string(torch.FloatTensor(coded_sentence))

    # seq = [emb(rnn(start))]
    # i = 0
    # while len(seq)<maxlen and eos not in seq:
    #     seq.append(torch.multinomial(seq[i]))
    #     i += 1

    # decoder(rnn(emb))
    # return torch.multinomial(logits,1)

def generate_multinomial(model, emb, decoder, eos, start="", maxlen=200, LSTM=False, GRU=False):
    """  Fonction de génération aléatoire (l'embedding et le decodeur peuvent être des fonctions du rnn).
         Initialise le réseau avec start (ou à 0 si start est vide)
            et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits
    # (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    coded_sentence = string2code(start).tolist()
    with torch.no_grad():
        x_emb = emb(torch.tensor(coded_sentence)).unsqueeze(1)
        if LSTM:
            H, C, _, _, _ = model.forward(x_emb).to(device)
            c = C[-1]
        elif GRU:
            H, _, _ = model.forward(x_emb).to(device)
        else:
            H = model.forward(x_emb).to(device)
        h = H[-1] #On génère la séquence à partir du dernier état caché de la séquence fournie
        char = []
        probas = F.softmax(decoder(h))
        char = torch.multinomial(input=probas, num_samples=1).item()
        coded_sentence.append(char)
        while char != eos:
            x_emb = emb(torch.tensor(coded_sentence[-1]).unsqueeze(0))
            if LSTM:
                h, c, _, _, _ = model.one_step(x_emb, h, c)
            elif GRU:
                h, _, _ = model.one_step(x_emb, h)
            else:
                h = model.one_step(x_emb, h)
            probas = F.softmax(decoder(h))
            char = torch.multinomial(input=probas, num_samples=1).item()
            coded_sentence.append(char)
            if len(coded_sentence) == maxlen:
                coded_sentence.append(eos)
                return code2string(torch.FloatTensor(coded_sentence))
    return code2string(torch.FloatTensor(coded_sentence))

def generate_beam(model, emb, decoder, eos, k, start="", maxlen=200, LSTM=False, GRU=False, nucleus=False):
    """
        Génère une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    coded_sentence = string2code(start).tolist()
    k_strings = [coded_sentence[:]] * k
    k_probas = []
    with torch.no_grad():
        x_emb = emb(torch.tensor(coded_sentence)).unsqueeze(1)
        if LSTM:
            H, C, _, _, _ = model.forward(x_emb)
            c = C[-1]
            k_c = [c] * k
        elif GRU:
            H, _, _ = model.forward(x_emb)
        else:
            H = model.forward(x_emb)
        h = H[-1] #On génère la séquence à partir du dernier état caché de la séquence fournie
        k_h = [h] * k
        nb_char = 0
        if nucleus:
            probas = torch.log(p_nucleus(decoder, k=k)(h))
        else:
            probas = torch.log(F.softmax(decoder(h), dim=1))
        probas, indices = torch.topk(probas.squeeze(), k)
        for i, string in enumerate(k_strings):
            string.append(indices[i].item())
            k_probas.append(probas[i].item())

        while True:
            nb_char += 1
            P = []
            Char = []
            new_k_strings = []
            new_k_probas = []
            for idx_string, string in enumerate(k_strings):
                x_emb = emb(torch.tensor(string[-1])).unsqueeze(0)
                if LSTM:
                    k_h[idx_string], k_c[idx_string], _, _, _ = model.one_step(x_emb, k_h[idx_string], k_c[idx_string])
                elif GRU:
                    k_h[idx_string], _, _ = model.one_step(x_emb, k_h[idx_string])
                else:
                    k_h[idx_string] = model.one_step(x_emb, k_h[idx_string])
                if nucleus:
                    probas = torch.log(p_nucleus(decoder, k=k)(k_h[idx_string]))
                else:
                    probas = torch.log(F.softmax(decoder(k_h[idx_string]), dim=1))
                probas, indices = torch.topk(probas.squeeze(), k)
                P.append(k_probas[idx_string] + probas)
                Char.append(indices)
            P = torch.stack(P, dim=0)
            Char = torch.stack(Char, dim=0)

            best_probas, best_indices = torch.topk(P.flatten(), k)
            rows = (best_indices // k).int()
            cols = best_indices % k

            for i in range(k):
                new_k_strings.append(k_strings[rows[i]] + [Char[rows[i], cols[i]].item()])
                new_k_probas.append(best_probas[i].item())
            k_strings = new_k_strings
            k_probas = new_k_probas

            best_idx = torch.argmax(torch.tensor(k_probas))
            best_string = k_strings[best_idx]

            if nb_char + len(start) >= maxlen:
                return code2string(best_string+[eos])
            
            if eos in best_string:
                eos_idx = [idx for idx, value in enumerate(best_string) if value==eos]
                best_string = best_string[:eos_idx[0]+1]
                k_strings[best_idx] = best_string
                return code2string(best_string)
            
            

# p_nucleus
def p_nucleus(decoder, k, alpha=0.95):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        y = decoder(h)
        nb_char = y.shape[1]
        probas = F.softmax(y, dim=1)
        probas, indices = torch.topk(probas.squeeze(), k)
        sum = probas.sum().item()
        nucleus_probas = torch.zeros(nb_char)
        for i, char in enumerate(indices):
            nucleus_probas[char.item()] = probas[i].item() / sum
            ic(nucleus_probas.shape, nucleus_probas)
            return nucleus_probas
    return compute
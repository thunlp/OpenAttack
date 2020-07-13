import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np

class ParseNet(nn.Module):
    def __init__(self, d_nt, d_hid, len_voc):
        super(ParseNet, self).__init__()
        self.d_nt = d_nt
        self.d_hid = d_hid
        self.len_voc = len_voc

        self.trans_embs = nn.Embedding(len_voc, d_nt)

        self.encoder = nn.LSTM(d_nt, d_hid, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(d_nt + d_hid, d_hid, num_layers=1, batch_first=True)

        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.out_nonlin = nn.LogSoftmax(dim=1)

        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid))

        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_nt + d_hid, 1))

    def compute_mask(self, lengths):
        device = lengths.device
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len, device=device).unsqueeze(0).expand(lengths.size(0), max_len)
        mask = lengths.unsqueeze(1).expand_as(range_row)
        mask = (range_row < mask).float()
        return mask

    def masked_softmax(self, vector, mask):
        result = nn.functional.softmax(vector, dim=1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13) # avoid dividing zero
        return result

    # compute masked attention over enc hiddens with bilinear product
    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous[0].mm(self.att_W)
        scores = b_hn.unsqueeze(1) * enc_hids
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # compute masked attention over parse sequence with bilinear product
    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):
        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous[0].mm(self.att_parse_W)
        scores = b_hn.unsqueeze(1) * trans_embs
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # return encoding for an input batch
    def encode_batch(self, inputs, lengths):
        device = inputs.device
        bsz, max_len = inputs.size()
        in_embs = self.trans_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)

        e_hid_init = torch.zeros(1, bsz, self.d_hid, device=device)
        e_cell_init = torch.zeros(1, bsz, self.d_hid, device=device)
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0]

        return all_hids[_indices], enc_last_hid.squeeze(0)[_indices]

    # decode one timestep
    def decode_step(self, idx, prev_words, prev_hid, prev_cell, 
                    enc_hids, trans_embs, in_sent_lens, trans_lens, bsz, max_len):
        device = self.trans_embs.parameters().__next__().device
        # initialize with zeros
        if idx == 0:
            word_input = torch.zeros(bsz, 1, self.d_nt, device=device)
        else:
            word_input = self.trans_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_nt)
        
        # concatenate w/ transformation embeddings
        trans_weights = self.compute_transformation_attention(prev_hid, trans_embs, trans_lens)
        trans_ctx = torch.sum(trans_weights.unsqueeze(2) * trans_embs, dim=1)
        decoder_input = torch.cat([word_input, trans_ctx.unsqueeze(1)], dim=2)

        # feed to decoder lstm
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))

        # compute attention for next time step and att weighted ave of encoder hiddens
        attn_weights = self.compute_decoder_attention(hn, enc_hids, in_sent_lens)
        attn_ctx = torch.sum(attn_weights.unsqueeze(2) * enc_hids, dim=1)

        # compute copy prob as function of lotsa shit
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)
        p_copy += attn_ctx.mm(self.copy_att_v)
        p_copy += hn.squeeze(0).mm(self.copy_hid_v)
        p_copy = torch.sigmoid(p_copy).squeeze(1)

        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self):
        raise NotImplemented

    # beam search given a single input parse and a batch of template transformations
    def batch_beam_search(self, inputs, out_trimmed, in_trans_lens,
            out_trimmed_lens, eos_idx, beam_size=5, max_steps=250):

        device = inputs.device
        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_trans_lens[0]]

        # encode inputs and trimmed outputs
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_lens)
        trim_hids, trim_last_hid = self.encode_batch(out_trimmed, out_trimmed_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0)
        cn = torch.zeros(1, 1, self.d_hid, device=device)

        # initialize beams (dictionary of batch_idx: beam params)
        beam_dict = {}
        for b_idx in range(trim_hids.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        # loop til max_decode, do lstm tick using previous prediction
        while True:
            # set up accumulators for predictions
            # assumption: all examples have same number of beams at each timestep
            prev_words = []
            prev_hs = []
            prev_cs = []

            for b_idx in beam_dict:
                beams = beam_dict[b_idx]
                # loop over everything in the beam
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b
                    # start with last word in sequence, if eos end the beam
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None
                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)


            # now batch decoder computations
            hs = torch.cat(prev_hs, dim=1)
            cs = torch.cat(prev_cs, dim=1)
            num_examples = hs.size(1)
            if prev_words is not None:
                prev_words = torch.LongTensor(prev_words).to(device)

            if num_examples != trim_hids.size(0):
                d1, d2, d3 = trim_hids.size()
                rep_factor = num_examples // d1
                curr_out = trim_hids.unsqueeze(1).expand(d1, rep_factor, d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens = out_trimmed_lens.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)
            else:
                curr_out = trim_hids
                curr_out_lens = out_trimmed_lens

            # expand out inputs and encoder hiddens
            _, in_len, hid_d = enc_hids.size()
            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)
            curr_enc_lens = in_trans_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_trans_lens[0])

            # concat prev word emb and prev attn input and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words, hs, cs, curr_enc_hids, curr_out, curr_enc_lens, curr_out_lens, num_examples, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = torch.zeros(num_examples, self.len_voc, device=device)
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()

            # compute prediction over vocab for a single time step
            pred_inp = torch.cat([hn.squeeze(0), attn_ctx], dim=1)
            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy.unsqueeze(1) * vocab_scores + (1 - p_copy.unsqueeze(1)) * preds

            # now loop over the examples and sort each separately
            for b_idx in beam_dict:
                beam_candidates = []
                # no words previously predicted
                if num_examples == len(beam_dict):
                    ex_hn = hn[:,b_idx,:].unsqueeze(0)
                    ex_cn = cn[:,b_idx,:].unsqueeze(0)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    # add top n candidates
                    for z in range(beam_size):
                        word_idx = top_indices[z].item()
                        beam_candidates.append((preds[word_idx].item(), ex_hn, ex_cn, [word_idx]))
                    beam_dict[b_idx] = beam_candidates
                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:,start:end,:]
                    ex_cn = cn[:,start:end,:]
                    ex_preds = final_preds[start:end]

                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob
                        # if one of the beams is already complete, add it to candidates
                        # note: this is inefficient, but whatever
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)

                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:,o_idx,:].unsqueeze(0)
                        curr_cn = ex_cn[:,o_idx,:].unsqueeze(0)
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].item()
                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().item()), curr_hn, curr_cn, seq + [word_idx]))

                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]
            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict


class SCPN(nn.Module):
    def __init__(self, d_word, d_hid, d_nt, d_trans, 
            len_voc, len_trans_voc, use_input_parse):
        super(SCPN, self).__init__()
        self.d_word = d_word
        self.d_hid = d_hid
        self.d_trans = d_trans
        self.d_nt = d_nt + 1
        self.len_voc = len_voc
        self.len_trans_voc = len_trans_voc
        self.use_input_parse = use_input_parse

        # embeddings
        self.word_embs = nn.Embedding(len_voc, d_word)
        self.trans_embs = nn.Embedding(len_trans_voc, d_nt)

        # lstms
        if use_input_parse:
            self.encoder = nn.LSTM(d_word + d_trans, d_hid, num_layers=1, bidirectional=True, batch_first=True)
        else:
            self.encoder = nn.LSTM(d_word, d_hid, num_layers=1, bidirectional=True, batch_first=True)

        self.encoder_proj = nn.Linear(d_hid * 2, d_hid)
        self.decoder = nn.LSTM(d_word + d_hid, d_hid, num_layers=2, batch_first=True)
        self.trans_encoder = nn.LSTM(d_nt, d_trans, num_layers=1, batch_first=True)
        
        # output softmax
        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.att_nonlin = nn.Softmax(dim=1)
        self.out_nonlin = nn.LogSoftmax(dim=1)

        # attention params
        self.att_parse_proj = nn.Linear(d_trans, d_hid)
        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid))

        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_word + d_hid, 1))

    # create matrix mask from length vector
    def compute_mask(self, lengths):
        device = lengths.device
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len, device=device).unsqueeze(0).expand(lengths.size(0), max_len)
        mask = lengths.unsqueeze(1).expand_as(range_row)
        mask = (range_row < mask).float()
        return mask

    # masked softmax for attention
    def masked_softmax(self, vector, mask):
        result = torch.nn.functional.softmax(vector, dim=1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
        return result

    # compute masked attention over enc hiddens with bilinear product
    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous.mm(self.att_W)
        scores = b_hn.unsqueeze(1) * enc_hids
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # compute masked attention over parse sequence with bilinear product
    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):
        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous.mm(self.att_parse_W)
        scores = b_hn.unsqueeze(1) * trans_embs
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    # return encoding for an input batch
    def encode_batch(self, inputs, trans, lengths):
        device = inputs.device
        bsz, max_len = inputs.size()
        in_embs = self.word_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)

        # concat word embs with trans hid
        if self.use_input_parse:
            in_embs = torch.cat([in_embs, trans.unsqueeze(1).expand(bsz, max_len, self.d_trans)], dim=2)
        
        e_hid_init = torch.zeros(2, bsz, self.d_hid, device=device)
        e_cell_init = torch.zeros(2, bsz, self.d_hid, device=device)
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]
        all_hids = self.encoder_proj(all_hids.view(-1, self.d_hid * 2)).view(bsz, max_len, self.d_hid)

        enc_last_hid = torch.cat([enc_last_hid[0], enc_last_hid[1]], dim=1)
        enc_last_hid = self.encoder_proj(enc_last_hid)[_indices]

        return all_hids, enc_last_hid

    # return encoding for an input batch
    def encode_transformations(self, trans, lengths, return_last=True):
        device = trans.device
        bsz, _ = trans.size()

        lens, indices = torch.sort(lengths, 0, True)
        in_embs = self.trans_embs(trans)
        t_hid_init = torch.zeros(1, bsz, self.d_trans, device=device)
        t_cell_init = torch.zeros(1, bsz, self.d_trans, device=device)
        all_hids, (enc_last_hid, _) = self.trans_encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (t_hid_init, t_cell_init))
        _, _indices = torch.sort(indices, 0)

        if return_last:
            return enc_last_hid.squeeze(0)[_indices]
        else:
            all_hids = unpack(all_hids, batch_first=True)[0]
            return all_hids[_indices]

    # decode one timestep
    def decode_step(self, idx, prev_words, prev_hid, prev_cell, 
                    enc_hids, trans_embs, in_sent_lens, trans_lens, bsz, max_len):
        device = self.word_embs.parameters().__next__().device

        # initialize with zeros
        if idx == 0:
            word_input = torch.zeros(bsz, 1, self.d_word, device=device)
        else:
            word_input = self.word_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_word)

        # concatenate w/ transformation embeddings
        trans_weights = self.compute_transformation_attention(prev_hid[1], trans_embs, trans_lens)
        trans_ctx = torch.sum(trans_weights.unsqueeze(2) * trans_embs, dim=1)
        decoder_input = torch.cat([word_input, trans_ctx.unsqueeze(1)], dim=2)

        # feed to decoder lstm
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))

        # compute attention for next time step and att weighted ave of encoder hiddens
        attn_weights = self.compute_decoder_attention(hn[1], enc_hids, in_sent_lens)
        attn_ctx = torch.sum(attn_weights.unsqueeze(2) * enc_hids, dim=1)

        # compute copy prob as function of lotsa shit
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)
        p_copy += attn_ctx.mm(self.copy_att_v)
        p_copy += hn[1].mm(self.copy_hid_v)
        p_copy = torch.sigmoid(p_copy).squeeze(1)

        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self):
        raise NotImplemented

    # beam search given a single sentence and a batch of transformations
    def batch_beam_search(self, inputs, out_trans, in_sent_lens, out_trans_lens, eos_idx, beam_size=5, max_steps=70):
        device = inputs.device
        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_sent_lens[0]]

        # encode transformations
        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, None, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = torch.zeros(2, 1, self.d_hid, device=device)

        # initialize beams (dictionary of batch_idx: beam params)
        beam_dict = {}
        for b_idx in range(out_trans.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        while True:
            # set up accumulators for predictions
            # assumption: all examples have same number of beams at each timestep
            prev_words = []
            prev_hs = []
            prev_cs = []

            for b_idx in beam_dict:
                beams = beam_dict[b_idx]

                # loop over everything in the beam
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b

                    # start with last word in sequence, if eos end the beam
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None

                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)

            # now batch decoder computations
            hs = torch.cat(prev_hs, dim=1)
            cs = torch.cat(prev_cs, dim=1)
            num_examples = hs.size(1)

            if prev_words is not None:
                prev_words = torch.LongTensor(prev_words).to(device)

            # expand out parse states if necessary
            if num_examples != out_trans_hids.size(0):
                d1, d2, d3 = out_trans_hids.size()
                rep_factor = num_examples // d1
                curr_out = out_trans_hids.unsqueeze(1).expand(d1, rep_factor, d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens = out_trans_lens.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)

            else:
                curr_out = out_trans_hids
                curr_out_lens = out_trans_lens

            # expand out inputs and encoder hiddens
            _, in_len, hid_d = enc_hids.size()
            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)
            curr_enc_lens = in_sent_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_sent_lens[0])

            # concat prev word emb and prev attn input and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words, hs, cs, curr_enc_hids, curr_out, curr_enc_lens, curr_out_lens, num_examples, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = torch.zeros(num_examples, self.len_voc, device=device)
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()

            # compute prediction over vocab for a single time step
            pred_inp = torch.cat([hn[1], attn_ctx], dim=1)

            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy.unsqueeze(1) * vocab_scores + (1 - p_copy.unsqueeze(1)) * preds

            # now loop over the examples and sort each separately
            for b_idx in beam_dict:
                beam_candidates = []

                # no words previously predicted
                if num_examples == len(beam_dict):
                    ex_hn = hn[:,b_idx,:].unsqueeze(1)
                    ex_cn = cn[:,b_idx,:].unsqueeze(1)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    # add top n candidates
                    for z in range(beam_size):
                        word_idx = top_indices[z].item()
                        beam_candidates.append((preds[word_idx].item(), ex_hn, ex_cn, [word_idx]))
                        beam_dict[b_idx] = beam_candidates
                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:,start:end,:]
                    ex_cn = cn[:,start:end,:]
                    ex_preds = final_preds[start:end]

                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob
                        # if one of the beams is already complete, add it to candidates
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)

                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:,o_idx,:]
                        curr_cn = ex_cn[:,o_idx,:]
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].item()
                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().item()),curr_hn.unsqueeze(1), curr_cn.unsqueeze(1), seq + [word_idx]))
                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict
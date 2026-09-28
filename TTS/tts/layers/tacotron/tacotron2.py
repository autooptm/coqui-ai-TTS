import logging
import os

import torch
from torch import nn
from torch.nn import functional as F

from .attentions import init_attn
from .common_layers import Linear, Prenet

logger = logging.getLogger(__name__)


_AO_MAX_STEPS = int(os.environ.get("TTS_OPT_MAX_STEPS", "10000"))
_AO_CACHE_MAX = int(os.environ.get("TTS_OPT_CACHE", "16"))
_AO_STOP_EVERY = int(os.environ.get("TTS_OPT_STOP_EVERY", "16"))


class _AoStepOpt:

    def __init__(self, dec, inputs, processed):
        self.dec = dec
        B, T, C = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        fc = dec.frame_channels
        self.r, self.fc = dec.r, fc
        self.cap = max(1, min(_AO_MAX_STEPS, int(dec.max_decoder_steps)))

        def z(*s):
            return torch.zeros(*s, device=dev, dtype=dt)

        self.inp, self.proc = z(B, T, C), z(*processed.shape)
        self.q, self.c = z(B, dec.query_dim), z(B, dec.query_dim)
        self.dh, self.dc = z(B, dec.decoder_rnn_dim), z(B, dec.decoder_rnn_dim)
        self.ctx = z(B, dec.encoder_embedding_dim)
        self.aw, self.awc = z(B, T), z(B, T)
        self.mem = z(B, fc)
        self.out, self.stop, self.align = z(self.cap, fc * dec.r), z(self.cap), z(self.cap, T)
        self.t, self.one = (torch.zeros(1, device=dev, dtype=torch.long),
                            torch.ones(1, device=dev, dtype=torch.long))
        self.stop_last = z(1)
        widths = [lin.linear_layer.out_features for lin in dec.prenet.linear_layers]
        self.ones = [torch.ones(B, w, device=dev, dtype=dt) for w in widths]
        self.mbuf = [z(B, w) for w in widths]
        self.drop_active = bool(getattr(dec.prenet, "prenet_dropout", False)
                                and (dec.training
                                     or getattr(dec.prenet, "dropout_at_inference", False)))
        if not self.drop_active:
            for buf in self.mbuf:
                buf.fill_(1.0)
        self._g = None

    def draw_masks(self):
        if not self.drop_active:
            return
        for ones, buf in zip(self.ones, self.mbuf):
            buf.copy_(torch.dropout(ones, 0.5, True))

    def _body(self):
        dec, att = self.dec, self.dec.attention
        x = self.mem
        for lin, mask in zip(dec.prenet.linear_layers, self.mbuf):
            x = torch.relu(lin(x)) * mask
        query_input = torch.cat((x, self.ctx), -1)
        q, c = dec.attention_rnn(query_input, (self.q, self.c))
        attention_cat = torch.cat((self.aw.unsqueeze(1), self.awc.unsqueeze(1)), dim=1)
        energies = att.v(torch.tanh(att.query_layer(q.unsqueeze(1))
                                    + att.location_layer(attention_cat) + self.proc))
        alignment = torch.softmax(energies.squeeze(-1), dim=-1)
        self.awc += alignment
        ctx = torch.bmm(alignment.unsqueeze(1), self.inp).squeeze(1)
        dh, dc = dec.decoder_rnn(torch.cat((q, ctx), -1), (self.dh, self.dc))
        decoder_output = dec.linear_projection(torch.cat((dh, ctx), dim=1))
        stopnet_input = torch.cat((dh, decoder_output), dim=1)
        stop_token = dec.stopnet(stopnet_input.detach() if dec.separate_stopnet else stopnet_input)
        decoder_output = decoder_output[:, : self.r * self.fc]

        self.out.index_copy_(0, self.t, decoder_output)
        self.stop_last.copy_(torch.sigmoid(stop_token).view(1))
        self.stop.index_copy_(0, self.t, self.stop_last)
        self.align.index_copy_(0, self.t, alignment)
        self.q.copy_(q); self.c.copy_(c)
        self.dh.copy_(dh); self.dc.copy_(dc)
        self.ctx.copy_(ctx)
        self.aw.copy_(alignment)
        self.mem.copy_(decoder_output[:, self.fc * (self.r - 1):])
        self.t += self.one

    def _reset(self, inputs, processed):
        self.inp.copy_(inputs); self.proc.copy_(processed)
        for b in (self.q, self.c, self.dh, self.dc, self.ctx, self.aw, self.awc, self.mem):
            b.zero_()
        self.t.zero_()

    def build(self, inputs, processed):
        self._reset(inputs, processed)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self._reset(inputs, processed)
        self._g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._g):
            self._body()
        return self

    def run(self, inputs, processed, guard):
        self._reset(inputs, processed)
        cap, thr, every = self.cap, self.dec.stop_threshold, max(1, _AO_STOP_EVERY)
        n = 0
        while n < cap:
            snap = torch.cuda.get_rng_state() if self.drop_active else None
            chunk = min(every, cap - n)
            for _ in range(chunk):
                self.draw_masks()
                self._g.replay()
            base, n = n, n + chunk
            for i, fired in enumerate((self.stop[base:n] > thr).tolist()):
                if fired and (base + i) > guard:
                    if snap is not None:
                        torch.cuda.set_rng_state(snap)
                        for _ in range(i + 1):
                            self.draw_masks()
                    return base + i + 1
        return n


def _ao_opt_for(dec, inputs):
    if os.environ.get("TTS_NO_OPT") or not inputs.is_cuda or dec.training:
        return None
    att = dec.attention
    if not (att.location_attention and not att.forward_attn and not att.windowing
            and att.norm == "softmax" and inputs.shape[0] == 1
            and dec.prenet.prenet_type == "original"):
        return None
    cache = getattr(dec, "_ao_opts", None)
    if cache is None:
        cache = dec._ao_opts = {}
    key = int(inputs.shape[1])
    sg = cache.get(key)
    if sg is None:
        processed = att.preprocess_inputs(inputs)
        try:
            sg = _AoStepOpt(dec, inputs, processed).build(inputs, processed)
        except Exception:                       # noqa: BLE001 -- never break synthesis
            logger.warning("optimized path unavailable; using the stock decoder loop",
                           exc_info=True)
            cache[key] = False
            return None
        if len(cache) >= _AO_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = sg
    return sg or None


# pylint: disable=no-value-for-parameter
# pylint: disable=unexpected-keyword-arg
class ConvBNBlock(nn.Module):
    r"""Convolutions with Batch Normalization and non-linear activation.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): convolution kernel size.
        activation (str): 'relu', 'tanh', None (linear).

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.convolution1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_normalization = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        o = self.convolution1d(x)
        o = self.batch_normalization(o)
        o = self.activation(o)
        o = self.dropout(o)
        return o


class Postnet(nn.Module):
    r"""Tacotron2 Postnet

    Args:
        in_out_channels (int): number of output channels.

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, in_out_channels, num_convs=5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(ConvBNBlock(in_out_channels, 512, kernel_size=5, activation="tanh"))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(ConvBNBlock(512, 512, kernel_size=5, activation="tanh"))
        self.convolutions.append(ConvBNBlock(512, in_out_channels, kernel_size=5, activation=None))

    def forward(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        return o


class Encoder(nn.Module):
    r"""Tacotron2 Encoder

    Args:
        in_out_channels (int): number of input and output channels.

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, in_out_channels=512):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for _ in range(3):
            self.convolutions.append(ConvBNBlock(in_out_channels, in_out_channels, 5, "relu"))
        self.lstm = nn.LSTM(
            in_out_channels, int(in_out_channels / 2), num_layers=1, batch_first=True, bias=True, bidirectional=True
        )
        self.rnn_state = None

    def forward(self, x, input_lengths):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o

    def inference(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        # self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        return o


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    """Tacotron2 decoder. We don't use Zoneout but Dropout between RNN layers.

    Args:
        in_channels (int): number of input channels.
        frame_channels (int): number of feature frame channels.
        r (int): number of outputs per time step (reduction rate).
        memory_size (int): size of the past window. if <= 0 memory_size = r
        attn_type (string): type of attention used in decoder.
        attn_win (bool): if true, define an attention window centered to maximum
            attention response. It provides more robust attention alignment especially
            at interence time.
        attn_norm (string): attention normalization function. 'sigmoid' or 'softmax'.
        prenet_type (string): 'original' or 'bn'.
        prenet_dropout (float): prenet dropout rate.
        forward_attn (bool): if true, use forward attention method. https://arxiv.org/abs/1807.06736
        trans_agent (bool): if true, use transition agent. https://arxiv.org/abs/1807.06736
        forward_attn_mask (bool): if true, mask attention values smaller than a threshold.
        location_attn (bool): if true, use location sensitive attention.
        attn_K (int): number of attention heads for GravesAttention.
        separate_stopnet (bool): if true, detach stopnet input to prevent gradient flow.
        max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 10000.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        in_channels,
        frame_channels,
        r,
        attn_type,
        attn_win,
        attn_norm,
        prenet_type,
        prenet_dropout,
        forward_attn,
        trans_agent,
        forward_attn_mask,
        location_attn,
        attn_K,
        separate_stopnet,
        max_decoder_steps,
    ):
        super().__init__()
        self.frame_channels = frame_channels
        self.r_init = r
        self.r = r
        self.encoder_embedding_dim = in_channels
        self.separate_stopnet = separate_stopnet
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = 0.5

        # model dimensions
        self.query_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.attn_dim = 128
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # memory -> |Prenet| -> processed_memory
        prenet_dim = self.frame_channels
        self.prenet = Prenet(
            prenet_dim, prenet_type, prenet_dropout, out_features=[self.prenet_dim, self.prenet_dim], bias=False
        )

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_channels, self.query_dim, bias=True)

        self.attention = init_attn(
            attn_type=attn_type,
            query_dim=self.query_dim,
            embedding_dim=in_channels,
            attention_dim=128,
            location_attention=location_attn,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            windowing=attn_win,
            norm=attn_norm,
            forward_attn=forward_attn,
            trans_agent=trans_agent,
            forward_attn_mask=forward_attn_mask,
            attn_K=attn_K,
        )

        self.decoder_rnn = nn.LSTMCell(self.query_dim + in_channels, self.decoder_rnn_dim, bias=True)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_channels, self.frame_channels * self.r_init)

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(self.decoder_rnn_dim + self.frame_channels * self.r_init, 1, bias=True, init_gain="sigmoid"),
        )
        self.memory_truncated = None

    def set_r(self, new_r):
        self.r = new_r

    def get_go_frame(self, inputs):
        B = inputs.size(0)
        memory = torch.zeros(1, device=inputs.device).repeat(B, self.frame_channels * self.r)
        return memory

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)
        # T = inputs.size(1)
        if not keep_states:
            self.query = torch.zeros(1, device=inputs.device).repeat(B, self.query_dim)
            self.attention_rnn_cell_state = torch.zeros(1, device=inputs.device).repeat(B, self.query_dim)
            self.decoder_hidden = torch.zeros(1, device=inputs.device).repeat(B, self.decoder_rnn_dim)
            self.decoder_cell = torch.zeros(1, device=inputs.device).repeat(B, self.decoder_rnn_dim)
            self.context = torch.zeros(1, device=inputs.device).repeat(B, self.encoder_embedding_dim)
        self.inputs = inputs
        self.processed_inputs = self.attention.preprocess_inputs(inputs)
        self.mask = mask

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.frame_channels:
            memory = memory.view(memory.shape[0], memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, frame_channels)
        memory = memory.transpose(0, 1)
        return memory

    def _parse_outputs(self, outputs, stop_tokens, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.frame_channels)
        outputs = outputs.transpose(1, 2)
        return outputs, stop_tokens, alignments

    def _update_memory(self, memory):
        if len(memory.shape) == 2:
            return memory[:, self.frame_channels * (self.r - 1) :]
        return memory[:, :, self.frame_channels * (self.r - 1) :]

    def decode(self, memory):
        """
        shapes:
           - memory: B x r * self.frame_channels
        """
        # self.context: B x D_en
        # query_input: B x D_en + (r * self.frame_channels)
        query_input = torch.cat((memory, self.context), -1)
        # self.query and self.attention_rnn_cell_state : B x D_attn_rnn
        self.query, self.attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state)
        )
        self.query = F.dropout(self.query, self.p_attention_dropout, self.training)
        self.attention_rnn_cell_state = F.dropout(
            self.attention_rnn_cell_state, self.p_attention_dropout, self.training
        )
        # B x D_en
        self.context = self.attention(self.query, self.inputs, self.processed_inputs, self.mask)
        # B x (D_en + D_attn_rnn)
        decoder_rnn_input = torch.cat((self.query, self.context), -1)
        # self.decoder_hidden and self.decoder_cell: B x D_decoder_rnn
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        # B x (D_decoder_rnn + D_en)
        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context), dim=1)
        # B x (self.r * self.frame_channels)
        decoder_output = self.linear_projection(decoder_hidden_context)
        # B x (D_decoder_rnn + (self.r * self.frame_channels))
        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)
        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        # select outputs for the reduction rate self.r
        decoder_output = decoder_output[:, : self.r * self.frame_channels]
        return decoder_output, self.attention.attention_weights, stop_token

    def forward(self, inputs, memories, mask):
        r"""Train Decoder with teacher forcing.
        Args:
            inputs: Encoder outputs.
            memories: Feature frames for teacher-forcing.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: (B, T, D_out_enc)
            - memory: (B, T_mel, D_mel)
            - outputs: (B, T_mel, D_mel)
            - alignments: (B, T_in, T_out)
            - stop_tokens: (B, T_out)
        """
        memory = self.get_go_frame(inputs).unsqueeze(0)
        memories = self._reshape_memory(memories)
        memories = torch.cat((memory, memories), dim=0)
        memories = self._update_memory(memories)
        memories = self.prenet(memories)

        self._init_states(inputs, mask=mask)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            decoder_output, attention_weights, stop_token = self.decode(memory)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [attention_weights]

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)
        return outputs, alignments, stop_tokens

    def inference(self, inputs):
        r"""Decoder inference without teacher forcing and use
        Stopnet to stop decoder.
        Args:
            inputs: Encoder outputs.

        Shapes:
            - inputs: (B, T, D_out_enc)
            - outputs: (B, T_mel, D_mel)
            - alignments: (B, T_in, T_out)
            - stop_tokens: (B, T_out)
        """
        opt_s = _ao_opt_for(self, inputs)
        if opt_s is not None:
            processed = self.attention.preprocess_inputs(inputs)
            n = opt_s.run(inputs, processed, inputs.shape[0] // 2)
            return (opt_s.out[:n].view(1, n * self.r, self.frame_channels).transpose(1, 2).contiguous(),
                    opt_s.align[:n].unsqueeze(0),
                    opt_s.stop[:n].view(1, n, 1))

        memory = self.get_go_frame(inputs)
        memory = self._update_memory(memory)

        self._init_states(inputs, mask=None)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments, t = [], [], [], 0
        while True:
            memory = self.prenet(memory)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > self.stop_threshold and t > inputs.shape[0] // 2:
                break
            if len(outputs) == self.max_decoder_steps:
                logger.info("Decoder stopped with `max_decoder_steps` %d", self.max_decoder_steps)
                break

            memory = self._update_memory(decoder_output)
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_truncated(self, inputs):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None, keep_states=False)
        else:
            self._init_states(inputs, mask=None, keep_states=True)

        self.attention.init_states(inputs)
        outputs, stop_tokens, alignments, t = [], [], [], 0
        while True:
            memory = self.prenet(self.memory_truncated)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > 0.7:
                break
            if len(outputs) == self.max_decoder_steps:
                logger.info("Decoder stopped with `max_decoder_steps` %d", self.max_decoder_steps)
                break

            self.memory_truncated = decoder_output
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_step(self, inputs, t, memory=None):
        """
        For debug purposes
        """
        if t == 0:
            memory = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None)

        memory = self.prenet(memory)
        decoder_output, stop_token, alignment = self.decode(memory)
        stop_token = torch.sigmoid(stop_token.data)
        memory = decoder_output
        return decoder_output, stop_token, alignment

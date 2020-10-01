
class BaseDecoding(object):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.quiet = args.quiet
        self.args = args
        self.domain = args.domain
        self.tokenizer = tokenizer

        self.domain_to_max_len = {
            'arggen': 140,
            'opinion': 243,
            'news': 335
        }

        self.pad_idx = 1
        self.mask_idx = 50264
        self.eos_idx = 2
        self.decoder_bos_idx = 0

        self.do_sampling = args.do_sampling
        self.topk = args.sampling_topk
        self.topp = args.sampling_topp
        self.temperature = args.temperature

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "generation_mode": True,
        }

    def generate(self, **kwargs):
        raise NotImplementedError
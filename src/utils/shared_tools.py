from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
from sentence_transformers.models import Pooling

from src.data.medic_tree import MEDICTree


class SharedTools:
    def __init__(self, args):
        self._init_config(args)
        self._init_tokenizer()
        self._init_encoder()
        self._init_tree_sim_tool()

    def _init_config(self, args):
        self.model_name_or_path = args.model_name_or_path
        self.last_layer = args.last_layer
        self.max_length = args.max_length
        self.use_cuda = args.use_cuda
        self.use_tree_similarity = args.use_tree_similarity
        self.dataset_name_or_path = args.dataset_name_or_path
        self.tree_ratio = args.tree_ratio
        self.root_path = args.root_path

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def _init_encoder(self):
        """Initialize the encoder model."""
        if self.last_layer in ["cls", "mean"]:
            word_embedding_model = models.Transformer(self.model_name_or_path, max_seq_length=self.max_length)
            pooling_model = Pooling(
                word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=self.last_layer,
            )
            self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            raise NotImplementedError

        if self.use_cuda:
            self.encoder.to("cuda")

    def _init_tree_sim_tool(self):
        """Initialize tree similarity if enabled."""
        if not self.use_tree_similarity:
            return
        self.tree_sim = MEDICTree(dataset_name_or_path=self.dataset_name_or_path, root_path=self.root_path, tree_ratio=self.tree_ratio)


class Sim_Cos():
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from ..data_manager import DataManager
        self.model = SentenceTransformer(DataManager.load("AttackAssist.SentenceTransformer"), device='cuda')
    def __call__(self, sen1, sen2):
        from sentence_transformers import util
        emb1,emb2 = self.model.encode([sen1,sen2],show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim.cpu().numpy()[0][0]

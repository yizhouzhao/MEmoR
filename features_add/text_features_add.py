import torch
from utils import read_json
from base import BaseFeatureExtractor

PARSED_TEXT_FOLDER = "./parsed_data/memor_text_answers"

class AddTextFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing Add_TextFeatureExtractor...")
        #self.feature_file = config["text"]["feature_file"]

        self.feature_dim = self.get_feature_dim()
        self.data = read_json(config["data_file"])
        self.features = read_json("./parsed_data/gathered_text_ans.json")
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature_dim(self):
        sample_json = read_json(PARSED_TEXT_FOLDER + "/S01E01_000/0.json")
        return len(sample_json["answer_vec"])

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ret = []
        ret_valid = []
        for character in on_characters:
            for ii, speaker in enumerate(speakers):
                if character == speaker:
                    index = "{}+{}".format(clip, ii)
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)
        ret = torch.stack(ret, dim=0)
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8)
        return ret, ret_valid

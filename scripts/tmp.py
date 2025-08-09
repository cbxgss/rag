import json
from tqdm import tqdm

from src.dataset import Item
from src.dataset.hotpotqa import get_hotpotqa

logs = {
    "nq": {
        "direct": [
            "1114_154503-direct",
            "1124_172531-direct",
            "1124_172533-direct",
        ],
        "native": [
            "1124_172319-native",
            "1124_172325-native",
            "1124_172340-native",
        ],
        "ircot": [
            "1124_172759-ircot",
            "1124_172802-ircot",
            "1124_172805-ircot",
        ],
        "krag": [
            "1124_171703-krag",
            "1124_171735-krag",
            "1124_171739-krag",
        ],
    },
    "eli5": {
        "direct": [
            "1111_131930-direct",
        ],
        "native": [
            "1111_131427-native",
        ],
        "ircot": [
            "1124_173742-ircot",
            "1124_173748-ircot",
            "1124_173750-ircot",
        ],
        "hrag": [
            "1111_131107-hrag",
        ],
        "krag": [
            "1124_173850-krag",
            "1124_173940-krag",
            "1124_173945-krag",
        ],
    },
    "asqa": {
        "direct": [
            "1111_134059-direct",
        ],
        "native": [
            "1111_134108-native",
        ],
        "ircot": [
            "1124_175023-ircot",
        ],
        "hrag": [
            "1217_172122-hrag",
        ],
        "krag": [
            "1124_175031-krag",
        ],
    },
    "hotpotqa": {
        "direct": [
            "1108_181439-direct",
            "1110_181316-direct",
            "1110_181326-direct",
        ],
        "native": [
            "1108_181734-native",
            "1110_181441-native",
            "1110_181444-native",
        ],
        "ircot": [
            "1122_164122-ircot",
            "1122_164125-ircot",
            "1122_164127-ircot",
        ],
        "hrag" : [
            "1122_193804-hrag",
            "1122_193811-hrag",
            "1122_193814-hrag",
        ],
        "lazykrag": [
            "1202_133406-lazykrag",
            "1202_134405-lazykrag",
            "1202_134409-lazykrag",
        ],
        "lazykrag-2hop": [
            "1202_145608-lazykrag",
            "1202_150048-lazykrag",
            "1202_150053-lazykrag",
        ],
        "native_top200": [
            "1202_133647-native",
        ],
        "native-corpus": [
            "1213_140530-native",
            "1213_140533-native",
            "1213_140535-native",
        ],
        "ircot-corpus": [
            "1213_123142-ircot",
            "1213_140057-ircot",
            "1213_140103-ircot",
        ],
        "hrag-corpus": [
            "1217_231804-hrag",
            "1217_231806-hrag",
            "1217_231808-hrag",
        ],
        "hrag-wo_ner-corpus": [
            "1219_125601-hrag",
            "1219_125603-hrag",
            "1219_125605-hrag",
        ],
        "lazykrag-corpus": [
            "1213_145200-lazykrag",
            "1213_145203-lazykrag",
            "1213_145205-lazykrag",
        ],
        "oracle-doc": [
            "1217_162440-upper",
            "1217_162755-upper",
            "1217_162757-upper",
        ],
        "oracle-sentence": [
            "1217_161207-upper",
            "1217_161236-upper",
            "1217_161239-upper",
        ],
    },
    "2wikimultihopqa": {
        "direct": [
            "1119_153655-direct",
            "1119_153718-direct",
            "1119_153722-direct",
        ],
        "native": [
            "1121_120023-native",
            "1121_120027-native",
            "1121_120029-native",
        ],
        "ircot": [
            "1122_165628-ircot",
            "1122_165855-ircot",
            "1122_165857-ircot",
        ],
        "hrag": [
            "1203_202500-hrag",
            "1203_202506-hrag",
            "1203_202523-hrag",
        ],
        "lazykrag": [
            "1203_135932-lazykrag",
            "1203_135945-lazykrag",
            "1203_135949-lazykrag",
        ],
        "native-corpus": [
            "1213_153844-native",
            "1213_153847-native",
            "1213_153849-native",
        ],
        "ircot-corpus": [
            "1213_154114-ircot",
            "1213_154117-ircot",
            "1213_154120-ircot",
        ],
        "hrag-corpus": [
            "1217_234757-hrag",
            "1217_234800-hrag",
            "1217_234803-hrag",
        ],
        "hrag-wo_ner-corpus": [
            "1219_123706-hrag",
            "1219_123707-hrag",
            "1219_123709-hrag",
        ],
        "lazykrag-corpus": [
            "1216_113611-lazykrag",
            "1216_113616-lazykrag",
            "1216_113622-lazykrag",
        ],
        "oracle-paragraph": [
            "1212_163713-upper",
        ],
        "oracle-sentence": [
            "1212_194316-upper",
            "1213_161512-upper",
            "1213_161517-upper",
        ],
    },
    "musique": {
        "direct": [
            "1204_112155-direct",
            "1204_112201-direct",
            "1204_112206-direct",
        ],
        "native": [
            "1204_112245-native",
            "1204_112251-native",
            "1204_112256-native",
        ],
        "ircot": [
            "1204_114123-ircot",
            "1204_114125-ircot",
            "1204_121632-ircot",
        ],
        "hrag": [
            "1204_111857-hrag",
            "1204_111900-hrag",
            "1204_111906-hrag",
        ],
        "krag": [
            "1204_114209-krag",
            "1204_114212-krag",
            "1204_121502-krag",
        ],
        "lazykrag": [
            "1204_114221-lazykrag",
            "1204_114224-lazykrag",
            "1204_114227-lazykrag",
        ],
    },
    "bamboogle": {
        "native": [
            "1204_143235-native",
            "1204_143240-native",
            "1204_143245-native",
        ],
        "ircot": [
            "1204_141624-ircot",
            "1204_141631-ircot",
            "1204_141634-ircot",
        ],
        "hrag": [
            "1204_142804-hrag",
            "1204_142807-hrag",
            "1204_142810-hrag",
        ],
    },
    "strategyqa": {
        "native": [
            "1204_144634-native",
            "1204_144636-native",
            "1204_144639-native",
        ],
        "ircot": [
            "1204_144649-ircot",
            "1204_144651-ircot",
            "1204_144654-ircot",
        ]
    },
}


datasets, _ = get_hotpotqa()

def new_file(log_base_path: str, item: Item):
    file_path = f"{log_base_path}/{item.id}.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    data["metadata"] = item.metadata
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

dataset = "hotpotqa"
for method in tqdm(logs[dataset], desc=dataset):
    for path in tqdm(logs[dataset][method], desc=method, leave=False):
        for i in tqdm(range(120), desc=path, leave=False):
            new_file(f"log/rag/{dataset}/{path}/output", datasets[i])

label_hierarchy_image_logic = {
    # Level 0: Specimen Representativeness 外周神经母细胞性肿瘤标本的代表性
    0: {"parent": "Peripheral Neuroblastic Tumor Specimen Representativeness", #外周神经母细胞性肿瘤标本的代表性
        "child": ["Adequate", "Inadequate"]}, # "有", "无"

    # Level 1: (Assuming Adequate Representativeness) Presence of Tumor Nests
    1: {"parent": "Adequate Representativeness", # Corresponds to "有"
        "child": ["Tumor Nests Present", "Tumor Nests Absent"]}, # "有瘤细胞巢", "无瘤细胞巢"

    # Level 2: (Assuming Tumor Nests Present) Schwannian Stroma Percentage
    2: {"parent": "Tumor Nests Present",
        "child": ["Schwannian Stroma >=50% (Stroma-Rich)", "Schwannian Stroma <50% (Stroma-Poor)"]},

    # Level 3a: (Stroma-Rich) Nodularity
    3: {"parent": "Schwannian Stroma >=50% (Stroma-Rich)",
        "child": ["Nodules Present (Ganglioneuroblastoma, Nodular)", "Nodules Absent (Ganglioneuroblastoma, Intermixed)"]},
        # "有结节" -> Ganglioneuroblastoma, Nodular (节细胞神经母细胞瘤，结节型)
        # "无结节" -> Ganglioneuroblastoma, Intermixed (节细胞神经母细胞瘤，混杂型)

    # Level 4a: (Stroma-Rich, Nodular) Differentiation for GNB, Nodular
    4: {"parent": "Nodules Present (Ganglioneuroblastoma, Nodular)",
        "child": [
            "Ganglioneuroblastoma, nodular type (poorly differentiated)", # 节细胞性神经母细胞瘤-结节型GNBn（分化差）
            "Ganglioneuroblastoma, nodular type (differentiated)"     # 节细胞性神经母细胞瘤-结节型GNBn(分化）
        ]},

    # Level 3b: (Stroma-Poor) Neuroblastoma subtypes
    5: {"parent": "Schwannian Stroma <50% (Stroma-Poor)",
        "child": [
            "Neuroblastoma (undifferentiated)", # 神经母细胞瘤NB（未分化）
            "Neuroblastoma (poorly differentiated)", # 神经母细胞瘤NB（分化差）
            "Neuroblastoma (differentiated)"      # 神经母细胞瘤NB（分化）
        ]},

    # Level 2b: (Adequate, No Tumor Nests) Ganglioneuroma types
    6: {"parent": "Tumor Nests Absent", # This is under "Adequate Representativeness"
        "child": [
            "Ganglioneuroma", # 节细胞性神经瘤
            "Ganglioneuroma, maturing", # 节细胞性神经瘤GN（即将成熟型）
            "Ganglioneuroma, mature"    # 节细胞性神经瘤GN（成熟型）
        ]},
    7: {"parent": "Inadequate Representativeness", "child": ["Tumor Nests Present (NOS)", "Tumor Nests Absent (Unclassifiable)"]},
    8: {"parent": "Tumor Nests Present (NOS)", "child": ["Neuroblastoma, NOS", "Ganglioneuroblastoma, NOS"]},
    9: {"parent": "Tumor Nests Absent (Unclassifiable)", "child": ["NTs, Unclassifiable"]}
}

# Mapping your 7 categories to (parent_ID, child_index_in_child_list)
# according to the image logic hierarchy defined above.
# The parent_ID refers to the key in label_hierarchy_image_logic
# whose "parent" string matches the immediate parent in the decision tree.

# 新的 category_mapping_image_logic_cn，包含完整路径
category_mapping_image_logic = {
    # 类别 1: 神经母细胞瘤 (未分化型) Neuroblastoma (undifferentiated)
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质<50% -> 未分化型
    1: [(0, 0), (1, 0), (2, 1), (5, 0)],

    # 类别 2: 神经母细胞瘤 (低分化型) Neuroblastoma (poorly differentiated)
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质<50% -> 低分化型
    2: [(0, 0), (1, 0), (2, 1), (5, 1)],

    # 类别 3: 神经母细胞瘤 (分化型) Neuroblastoma (differentiated)
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质<50% -> 分化型
    3: [(0, 0), (1, 0), (2, 1), (5, 2)],

    # 类别 4: 节细胞神经母细胞瘤，结节型 (低分化型) Ganglioneuroblastoma, nodular type (poorly differentiated)
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质>=50% -> 结节型 -> 低分化型
    4: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],

    # 类别 5: 节细胞神经母细胞瘤，结节型 (分化型) Ganglioneuroblastoma, nodular type (differentiated)
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质>=50% -> 结节型 -> 分化型
    5: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1)],

    # 类别 6: 节细胞神经母细胞瘤，混合型 Ganglioneuroblastoma, mixed type
    # 路径: 代表性 -> 有 -> 有瘤细胞巢 -> 施万基质>=50% -> 混合型 (无结节)
    6: [(0, 0), (1, 0), (2, 0), (3, 1)],

    # 类别 7: 神经节细胞瘤 Ganglioneuroma
    # 路径: 代表性 -> 有 -> 无瘤细胞巢 -> 神经节细胞瘤
    7: [(0, 0), (1, 1), (6, 0)]
}


feature_phrases = {
  "Adequate": "adequate specimen representativeness",
  "Inadequate": "inadequate specimen representativeness", # Not used for these 7 categories as they imply adequacy
  "Tumor Nests Present": "the presence of tumor nests",
  "Tumor Nests Absent": "the absence of tumor nests",
  "Schwannian Stroma >=50% (Stroma-Rich)": "Schwannian stroma comprising 50% or more of the tumor (stroma-rich)",
  "Schwannian Stroma <50% (Stroma-Poor)": "Schwannian stroma comprising less than 50% of the tumor (stroma-poor)",
  "Nodules Present (Ganglioneuroblastoma, Nodular)": "the presence of nodules (consistent with Ganglioneuroblastoma, Nodular type)",
  # "Nodules Absent (Ganglioneuroblastoma, Intermixed)" is a classification itself, so it won't be listed as a preceding feature if it's the final type.
}

def get_path_description_string(path_tuples_for_features, hierarchy_dict, feature_phrases_dict):
    phrased_features = []
    for parent_id, child_idx in path_tuples_for_features:
        raw_feature = hierarchy_dict[parent_id]["child"][child_idx]
        phrased_features.append(
            feature_phrases_dict.get(raw_feature, raw_feature))  # Fallback to raw_feature if not in dict

    if not phrased_features:
        return "basic classification"  # Fallback for direct classifications if any

    if len(phrased_features) == 1:
        return phrased_features[0]

    # Join with commas and 'and' for the last item
    return ", ".join(phrased_features[:-1]) + ", and " + phrased_features[-1]


prompt_templates = {}

full_tumor_names = {
    1: "Neuroblastoma (undifferentiated)",
    2: "Neuroblastoma (poorly differentiated)",
    3: "Neuroblastoma (differentiated)",
    4: "Ganglioneuroblastoma, nodular type (poorly differentiated)",
    5: "Ganglioneuroblastoma, nodular type (differentiated)",
    6: "Ganglioneuroblastoma, mixed type",
    7: "Ganglioneuroma" # Combined category from your descriptions 7 & 8
}



appearance_descriptions = {
    1: "neuroblasts in clusters/nests separated by delicate, incomplete stroma, and characteristically lacking ganglion cell-like differentiation and neurofibrils.", # NB (undifferentiated)
    2: "less than 5% of tumor cells showing differentiation (neuroblastic or ganglion cell-like) within a neurofibrillary background; Schwannian stroma is notably absent or minimal.", # NB (poorly differentiated)
    3: "over 5% of tumor cells showing differentiation (ganglion cell-like, sometimes to mature forms), accompanied by common and abundant neurofibrils; some Schwannian stroma is present.", # NB (differentiated)
    4: "distinct neuroblastoma nodules where the neuroblastic component itself has sparse Schwannian stroma; these nodules are clearly demarcated from surrounding stroma-rich areas, and often encased by a fibrous pseudocapsule.", # GNB, nodular (poorly diff)
    5: "distinct neuroblastoma nodules where the neuroblastic component itself has sparse Schwannian stroma; these nodules are clearly demarcated from surrounding stroma-rich areas, and often encased by a fibrous pseudocapsule. The neuroblastic component shows features of differentiation.", # GNB, nodular (diff) - Added a slight distinction for "differentiated" as implied by name.
    6: "an intermixed pattern (no macroscopic nodules), where well-demarcated nests of neuroblasts (representing ≤50% of tumor volume, showing varying differentiation and containing neurofibrils) are scattered within the stroma.", # GNB, mixed type
    7: "a composition of mature Schwannian stroma containing ganglion cells ranging from maturing to fully mature (mature forms typically with satellite cells), alongside axon-like fascicular processes. Neuroblastic elements are characteristically absent or minimal." # Ganglioneuroma
}

for category_id in range(1, 8):  # Categories 1 through 7
    full_name = full_tumor_names[category_id]
    appearance_desc = appearance_descriptions[category_id]

    path_tuples_full = category_mapping_image_logic[category_id]
    # The features leading to the classification are from all path steps *except* the last one,
    # as the last step's child in the hierarchy is the specific tumor type name itself.
    path_tuples_for_features = path_tuples_full[:-1]

    path_desc_string = get_path_description_string(path_tuples_for_features,
                                                   label_hierarchy_image_logic,
                                                   feature_phrases)

    prompt_templates[
        category_id] = f"A histopathology image of {full_name}.\n Pathologically, this tumor is characterized by {path_desc_string}.\n Microscopically, it shows {appearance_desc}"



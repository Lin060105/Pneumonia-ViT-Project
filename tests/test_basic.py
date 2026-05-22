import torch

from model_utils import (
    CLASS_NAMES,
    create_model,
    decide_screening_status,
    load_model_checkpoint,
    save_model_checkpoint,
)


def test_model_architecture_and_forward_pass():
    model = create_model(pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2)


def test_threshold_decision_uses_pneumonia_probability():
    pneumonia = decide_screening_status([0.2, 0.8], threshold=0.5, uncertainty_margin=0.0)
    normal = decide_screening_status([0.7, 0.3], threshold=0.5, uncertainty_margin=0.0)
    review = decide_screening_status([0.51, 0.49], threshold=0.5, uncertainty_margin=0.02)

    assert pneumonia["decision"] == "PNEUMONIA"
    assert normal["decision"] == "NORMAL"
    assert review["decision"] == "REVIEW"


def test_checkpoint_roundtrip_with_metadata(tmp_path):
    model = create_model(pretrained=False)
    checkpoint_path = tmp_path / "model.pth"
    metadata = {
        "model_name": "vit_base_patch16_224",
        "class_names": list(CLASS_NAMES),
        "threshold": 0.5,
    }

    save_model_checkpoint(checkpoint_path, model, metadata)
    loaded_model, loaded_metadata = load_model_checkpoint(checkpoint_path, device="cpu")

    assert tuple(loaded_metadata["class_names"]) == CLASS_NAMES
    assert loaded_metadata["threshold"] == 0.5
    with torch.no_grad():
        assert loaded_model(torch.randn(1, 3, 224, 224)).shape == (1, 2)

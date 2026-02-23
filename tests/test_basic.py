"""
Pneumonia ViT Model - Unit Tests
用於自動化測試模型架構與資料流是否正常。
"""
import torch
import torch.nn as nn
import timm

def test_model_architecture_and_forward_pass():
    """
    測試 1: 模型是否能成功建立，且輸出層是否正確改為 2 分類。
    測試 2: 輸入一張假的 X 光片，模型是否能成功計算並給出 2 個機率值。
    """
    # 1. 建立模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.eval() # 設定為評估模式
    
    # 2. 模擬一張 X 光片 Tensor (Batch_size=1, Channel=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. 讓假圖片通過模型
    with torch.no_grad():
        output = model(dummy_input)
    
    # 4. 斷言 (Assert)：如果輸出不是 1 筆資料且 2 個類別，程式就會報錯攔截！
    assert output.shape == (1, 2), f"模型輸出維度錯誤！預期 (1, 2)，卻得到 {output.shape}"
    print("✅ 模型架構與前向傳播 (Forward pass) 測試通過！")
# lora-utils

PyTorchモデルに対するLoRA（Low-Rank Adaptation）操作のためのユーティリティライブラリです。このパッケージは、LoRAレイヤーの注入、管理、操作のためのツールを提供します。

## 特徴

- **LoRAレイヤーの注入**: 既存のPyTorchモデルに簡単にLoRAレイヤーを注入
- **事前学習済みLoRAの読み込み**: 事前学習済みLoRA重みの読み込みと適用
- **重みのマージ**: LoRA重みとベースモデル重みのマージ
- **状態辞書の分離**: 学習済みモデルからLoRA重みを抽出
- **LoRAの削除**: モデルからLoRAレイヤーを削除して元のモデルに戻す
- **柔軟な設定**: カスタムrank、alpha、dropoutパラメータのサポート
- **複数のレイヤータイプ**: `nn.Linear`と`nn.Conv2d`の両方のレイヤーに対応

## インストール

### Gitリポジトリからインストール

```bash
pip install git+https://github.com/msqrd6/lora_utils.git
```

## クイックスタート

```python
import torch
import torch.nn as nn
from lora_utils import inject_init_lora_for_model, get_lora_dict_from_model

# シンプルなモデルを作成
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# rank=4でLoRAレイヤーを注入
network_alphas = inject_init_lora_for_model(
    model, 
    rank=4, 
    alpha=1.0, 
    dropout=0.1
)

# モデルを学習...
# ...

# LoRA重みを抽出
lora_state_dict = get_lora_dict_from_model(model)

# LoRA重みを保存
torch.save(lora_state_dict, "lora_weights.pt")
```

## 使用例

### 1. LoRAレイヤーの初期化

学習用にモデルにLoRAレイヤーを注入:

```python
from lora_utils import inject_init_lora_for_model

# すべてのLinearとConv2dレイヤーにLoRAを注入
network_alphas = inject_init_lora_for_model(
    model,
    rank=4,           # LoRAのrank
    alpha=1.0,        # スケーリング係数
    dropout=0.1       # ドロップアウト率
)

# 特定のレイヤーのみにLoRAを注入
network_alphas = inject_init_lora_for_model(
    model,
    rank=8,
    alpha=2.0,
    inject_layer_key=["attention", "mlp"]  # これらのキーワードを含むレイヤーのみに注入
)

# Linearレイヤーのみに注入（Conv2dを除外）
network_alphas = inject_init_lora_for_model(
    model,
    rank=4,
    alpha=1.0,
    linear=True,
    conv2d=False  # Conv2dレイヤーを除外
)

# Conv2dレイヤーのみに注入（Linearを除外）
network_alphas = inject_init_lora_for_model(
    model,
    rank=4,
    alpha=1.0,
    linear=False,  # Linearレイヤーを除外
    conv2d=True
)
```

### 2. 事前学習済みLoRA重みの読み込み

ベースモデルに事前学習済みLoRA重みを適用:

```python
from lora_utils import inject_pretrained_lora_for_model

# LoRA状態辞書を読み込み
lora_state_dict = torch.load("lora_weights.pt")

# 事前学習済みLoRAをモデルに注入
inject_pretrained_lora_for_model(
    base_model,
    lora_state_dict,
    strength=1.0  # LoRAの強度を調整（0.0〜1.0以上）
)
```

### 3. LoRAとベース重みのマージ

LoRA重みとベース重みを結合してマージ済みモデルを作成:

```python
from lora_utils import marge_lora_and_weight

# ベースモデルとLoRA重みを読み込み
base_state_dict = base_model.state_dict()
lora_state_dict = torch.load("lora_weights.pt")

# 重みをマージ
merged_state_dict = marge_lora_and_weight(
    lora_state_dict,
    base_state_dict,
    strength=1.0
)

# マージ済み重みをモデルに読み込み
model.load_state_dict(merged_state_dict)
```

### 4. 学習済みモデルからLoRA重みを抽出

学習済みモデルからLoRA重みを分離:

```python
from lora_utils import get_lora_dict_from_model

# LoRA重みのみを抽出
lora_state_dict = get_lora_dict_from_model(model)

# LoRAとベースモデルの両方の重みを抽出
lora_state_dict, model_state_dict = get_lora_dict_from_model(
    model,
    get_model_dict=True
)
```

### 5. モデルからLoRAを削除

モデルからLoRAレイヤーを削除して元のモデルに戻す:

```python
from lora_utils import remove_lora_from_model

# LoRAレイヤーを削除
original_model = remove_lora_from_model(model)
```

## APIリファレンス

### `inject_init_lora_for_model(model, rank=4, alpha=1.0, dropout=0.0, inject_layer_key=[], linear=True, conv2d=True)`

モデルにLoRAレイヤーを初期化して注入します。

**パラメータ:**
- `model` (nn.Module): 対象のPyTorchモデル
- `rank` (int): LoRAのrank（デフォルト: 4）
- `alpha` (float): スケーリング係数（デフォルト: 1.0）
- `dropout` (float): ドロップアウト率（デフォルト: 0.0）
- `inject_layer_key` (list[str]): 注入するレイヤーをフィルタリングするキーワードのリスト（デフォルト: []）
- `linear` (bool): Linearレイヤーに注入するかどうか（デフォルト: True）
- `conv2d` (bool): Conv2dレイヤーに注入するかどうか（デフォルト: True）

**戻り値:**
- `dict`: ネットワークalphaの辞書

### `inject_pretrained_lora_for_model(base_model, lora_state_dict, strength=1.0)`

事前学習済みLoRA重みをベースモデルに注入します。

**パラメータ:**
- `base_model` (nn.Module): ベースのPyTorchモデル
- `lora_state_dict` (dict): LoRA状態辞書
- `strength` (float): LoRA強度の乗数（デフォルト: 1.0）

### `marge_lora_and_weight(lora_state_dict, base_state_dict, strength=1.0)`

LoRA重みとベースモデル重みをマージします。

**パラメータ:**
- `lora_state_dict` (dict): LoRA状態辞書
- `base_state_dict` (dict): ベースモデル状態辞書
- `strength` (float): LoRA強度の乗数（デフォルト: 1.0）

**戻り値:**
- `dict`: マージ済み状態辞書

### `get_lora_dict_from_model(model, get_model_dict=False)`

モデルからLoRA重みを抽出します。

**パラメータ:**
- `model` (nn.Module): LoRAレイヤーを持つモデル
- `get_model_dict` (bool): ベースモデル重みも返すかどうか（デフォルト: False）

**戻り値:**
- `dict` または `tuple`: LoRA状態辞書、または`get_model_dict=True`の場合は（LoRA状態辞書, モデル状態辞書）

### `remove_lora_from_model(model)`

モデルからLoRAレイヤーを削除して元のモデルに戻します。

**パラメータ:**
- `model` (nn.Module): LoRAレイヤーを持つモデル

**戻り値:**
- `nn.Module`: LoRAレイヤーが削除されたモデル

### `LoRA` クラス

LoRA機能でベースレイヤーをラップするPyTorchモジュールです。

**メソッド:**
- `append_lora_layer(rank, alpha, strength=1.0, dropout=0.0)`: 新しいLoRAレイヤーを追加
- `load_weight(lora_A, lora_B, strength=1.0, alpha=1.0, dropout=0.0)`: 事前学習済みLoRA重みを読み込み
- `forward(x)`: ベースレイヤーとLoRAレイヤーを組み合わせた順伝播

## 必要要件

- Python >= 3.9
- PyTorch >= 2.1.0

## ライセンス

[MIT License](LICENSE)

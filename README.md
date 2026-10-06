# pcd-map-generator

撮影した 3D 点群データ (`.pcd`) から、マウス操作でテスト領域を直感的に切り抜き、**2D 占有格子地図（Nav2 用）** および **自己位置推定用フィールド定義 JSON（`gn10-pointcloud-localization` 用）** を一括自動生成するツールです。

---

## 主な特徴

- 🖱️ **GUI マウスドラッグで一発切り抜き**: 座標の数値を測らなくても、画面上で使いたい部屋・フィールドを四角く囲むだけで切り抜き完了。
- 🤖 **ロボット全高フィルタ**: 天井、照明、梁などの頭上障害物を自動カットし、Nav2 での立ち往生を防止。
- 🧱 **`gn10-pointcloud-localization` 完全互換**: 外壁（立上りフェンス）や机・円柱バケツを自動認識し、C++ パーサー互換の JSON を生成。
- 🗺️ **ROS 2 Navigation (Nav2) 標準対応**: 2D OccupancyGrid (`.pgm` / `.yaml`) を同時出力。

---

## 必要な環境 (依存パッケージ)

```bash
pip install -r requirements.txt
```
- Python 3.8+
- `open3d`
- `opencv-python`
- `numpy`

---

## 使い方

### 1. GUI マウス選択モード (推奨)

```bash
python3 pcd_map_generator.py /path/to/your_map.pcd
```

1. ウィンドウに鳥瞰図プレビュー（1m グリッド線・原点マーカー付き）が表示されます。
2. 使いたいエリアをマウス左ドラッグで四角く囲みます。
3. `Enter` または `Space` キーを押して確定します（`c` で再選択、`ESC` で中止）。

---

### 2. CLI 直接指定モード (ヘッドレス環境 / SSH)

```bash
python3 pcd_map_generator.py /path/to/your_map.pcd \
  --bbox -1.6 4.2 -2.0 1.2 \
  --robot_height_m 1.5 \
  --output_dir output \
  --prefix practice_field \
  --headless
```

---

## 主なオプション引数

| 引数 | 型 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| `pcd_path` | 文字列 | (必須) | 入力 PCD ファイルのパス |
| `--resolution_m` | float | `0.05` | 2D グリッド解像度 [m/pixel] (5cm) |
| `--robot_height_m` | float | `1.5` | ロボットの全高上限 [m] (天井・梁を除去) |
| `--ground_margin_m`| float | `0.03` | 床面ノイズ除外マージン [m] |
| `--output_dir` | 文字列 | `output` | 出力先ディレクトリ |
| `--prefix` | 文字列 | `practice_field`| 出力ファイル名の接頭辞 |
| `--bbox` | float 4つ | なし | 直接範囲指定 `[min_x max_x min_y max_y]` |
| `--headless` | フラグ | `False` | GUI ウィンドウを開かないモード |
| `--no_outer_walls` | フラグ | `False` | 選択範囲の外周壁自動生成をオフにする場合 |

---

## 出力ファイル群

実行後、指定した `--output_dir` 配下に以下のファイルがワンセットで出力されます：

1. **`{prefix}_cropped.pcd`**: 切り抜かれた 3D 実測点群 (`map_source_type: "pcd"` で使用可能)
2. **`{prefix}_map.json`**: 外壁・机・円柱が定義された JSON (`map_source_type: "json"` で使用可能)
3. **`{prefix}_map.pgm` / `.yaml`**: ROS 2 Nav2 標準の 2D Occupancy Grid 地図
4. **`{prefix}_preview.png`**: 検出されたオブジェクト枠（番号・名前付き）の確認用画像

---

## ディレクトリ構成

```text
pcd-map-generator/
├── pcd_map_generator/        # コアモジュールパッケージ
│   ├── __init__.py           # パッケージエクスポート
│   ├── coordinates.py        # 座標系・相互変換 (CoordinateTransformMeta, BoundingBox2D)
│   ├── pointcloud.py         # 点群I/O・床面Z推定・クロップ処理
│   ├── selector.py           # 鳥瞰図プレビュー生成・ROI領域選択
│   ├── extractor.py          # 幾何オブジェクト抽出 (外壁・机・バケツの認識)
│   └── exporter.py           # Nav2 PGM/YAML, JSON, プレビューPNG の出力
├── pcd_map_generator.py      # CLI エントリポイント
├── requirements.txt          # 依存パッケージ定義
└── README.md
```

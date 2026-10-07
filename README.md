# pcd-map-generator

撮影した 3D 点群データ (`.pcd`) から、マウス操作でテスト領域を直感的に切り抜き、**2D 占有格子地図（Nav2 用）** および **自己位置推定用フィールド定義 JSON（`gn10-pointcloud-localization` 用）** を一括自動生成するツールです。

---

## 主な特徴

- 🖱️ **GUI マウスドラッグで一発切り抜き**: 座標の数値を測らなくても、画面上で使いたい部屋・フィールドを四角く囲むだけで切り抜き完了。
- 🧱 **`gn10-pointcloud-localization` 完全互換**: 外壁（立上りフェンス & 土台）を自動生成。ノイズによる誤認識を防ぐため、デフォルトで外壁のみを安全に出力。
- 🎯 **障害物の GUI 手動追加**: 机やバケツを追加したい場合も、画面上でドラッグして高さ推定値を確認しながら簡単に追加可能。
- 🤖 **ロボット全高フィルタ**: 天井、照明、梁などの頭上障害物を自動カットし、Nav2 での立ち往生を防止。
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
4. 確定後、コンソールで「内部の障害物（机・バケツ）を手動追加するか」を確認されます。追加する場合は画面上で障害物をドラッグして登録できます（不要ならそのまま Enter で外壁のみ出力）。

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
| `--add_obstacles` | フラグ | `False` | 内部の障害物（机・バケツ）を手動追加する GUI を起動 |

---

## 出力ファイル群

実行後、指定した `--output_dir` 配下に以下のファイルがワンセットで出力されます：

1. **`{prefix}_cropped.pcd`**: 切り抜かれた 3D 実測点群 (`map_source_type: "pcd"` で使用可能)
2. **`{prefix}_map.json`**: 外壁・机・円柱が定義された JSON (`map_source_type: "json"` で使用可能)
3. **`{prefix}_map.pgm` / `.yaml`**: ROS 2 Nav2 標準の 2D Occupancy Grid 地図
4. **`{prefix}_preview.png`**: 検出されたオブジェクト枠（番号・名前付き）の確認用画像

---

## 各画面・地図における「色」の意味

### 1. GUI 画面（切り抜き・障害物追加）

| 色 | 画面上の表示 | 意味・役割 |
| :--- | :--- | :--- |
| **白** | 背景全体 | **走行可能な床面（フリースペース）** |
| **緑色** | 四隅の外枠線 | **外壁フェンス（立上り壁）の境界線** |
| **薄いグレー** | ポツポツとした淡い点 | **実測点群データ（障害物配置のガイド目印・下書き）** |
| **青色** | 四角い枠線 | **手動追加した直方体障害物 (`BOX`: 机・台座など)** |
| **赤色** | 丸い枠線 | **手動追加した円柱障害物 (`CYLINDER`: バケツ・ポールなど)** |
| **赤い十字** | `(0, 0)` マーク | **ワールド座標系の原点マーカー** |

### 2. Nav2 2D 占有格子地図 (`.pgm` 画像)

ROS 2 Navigation 標準仕様に準拠しています：

| 色 | ピクセル値 | Nav2 での判定 | 説明 |
| :--- | :--- | :--- | :--- |
| **白** | `254` | **Free（走行可能床面）** | ロボットが障害物に衝突せず自由に走行できるクリーンな床 |
| **黒** | `0` | **Occupied（進入禁止・壁）** | 外壁フェンス、机、バケツなど衝突する障害物 |

### 3. 3D 可視化 / RViz (`gn10` 自己位置推定)

| 色 | 形状タイプ | 意味・役割 |
| :--- | :--- | :--- |
| **鮮やかな緑** | `BOX` | **立上り外壁 / 机 (`BOX`)**: LiDAR 点群と照合される本物の壁面 (Z: 0.024m〜0.150m) |
| **半透明の水色** | `VISUAL_BOX` | **土台フェンス (`VISUAL_BOX`)**: 床面土台 (Z: 0.000m〜0.024m, 地面除去で消えるため照合外・表示用) |
| **赤色** | `CYLINDER` | **円柱障害物 (`CYLINDER`)**: バケツやポールなど |

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

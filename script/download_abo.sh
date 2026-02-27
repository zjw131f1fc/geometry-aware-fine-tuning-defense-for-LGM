#!/bin/bash
# 下载 Amazon Berkeley Objects (ABO) 3D 模型数据集到 datas/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_DIR/datas"

mkdir -p "$TARGET_DIR"

URL="https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
TAR_FILE="$TARGET_DIR/abo-3dmodels.tar"

echo "=== ABO 3D Models 下载脚本 ==="
echo "目标目录: $TARGET_DIR"
echo "文件大小: ~154 GB"
echo ""

# 下载（支持断点续传）
if [ -f "$TAR_FILE" ]; then
    echo "检测到已有文件，尝试断点续传..."
    wget -c "$URL" -O "$TAR_FILE"
else
    wget "$URL" -O "$TAR_FILE"
fi

echo ""
echo "下载完成，开始解压..."
tar xf "$TAR_FILE" -C "$TARGET_DIR"

echo ""
echo "解压完成。数据位于: $TARGET_DIR/3dmodels/"
echo "可删除 tar 包节省空间: rm $TAR_FILE"

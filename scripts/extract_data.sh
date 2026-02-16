#!/bin/bash
# 解压OmniObject3D blender_renders数据

# 设置路径
SOURCE_DIR="./datas/omniobject3d___OmniObject3D-New/raw/blender_renders_24_views/img"
TARGET_DIR="./datas/processed"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 统计压缩包数量
total_files=$(find "$SOURCE_DIR" -name "*.tar.gz" | wc -l)
echo "找到 $total_files 个压缩包"

# 解压所有tar.gz文件
count=0
for tarfile in "$SOURCE_DIR"/*.tar.gz; do
    if [ -f "$tarfile" ]; then
        count=$((count + 1))
        filename=$(basename "$tarfile" .tar.gz)
        echo "[$count/$total_files] 正在解压: $filename"

        # 解压到目标目录
        tar -xzf "$tarfile" -C "$TARGET_DIR"

        # 显示进度
        if [ $((count % 10)) -eq 0 ]; then
            echo "已完成: $count/$total_files"
        fi
    fi
done

echo "解压完成！"
echo "数据位置: $TARGET_DIR"

# 统计解压后的对象数量
object_count=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "总共解压了 $object_count 个对象"

# 显示示例结构
echo -e "\n示例数据结构:"
first_obj=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -n "$first_obj" ]; then
    echo "$first_obj/"
    ls -lh "$first_obj" | head -10
fi

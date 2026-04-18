xxx代表数据目录
render_cond需要配置正确的blender路径
python build_metadata.py --root xxx &&
python render_cond.py --root xxx --ultra_fast_mode &&
python encode_block.py --root xxx --device cuda
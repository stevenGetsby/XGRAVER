import importlib

__attributes = {
    'BasicTrainer': 'basic',
    'FlowMatchingTrainer': 'flow_matching.flow_matching',
    'FlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'ImageConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'SparseFlowMultiTokenTrainer': 'flow_matching.feats_matching',
    'SparseFlowMultiTokenCFGTrainer': 'flow_matching.feats_matching',
    'ImageConditionedSparseFlowMultiTokenCFGTrainer': 'flow_matching.feats_matching',
    'ImageConditionedDirectMaskTrainer': 'flow_matching.direct_mask',
    'MaskRefineTrainer': 'flow_matching.mask_refine',
    'ImageConditionedMaskRefineTrainer': 'flow_matching.mask_refine',
    'UDF64FlowTrainer': 'flow_matching.udf64_flow',
    'ImageConditionedUDF64FlowTrainer': 'flow_matching.udf64_flow',
    'Patch2ToPatch4FlowTrainer': 'flow_matching.patch_mask_flow',
    'ImageConditionedPatch2ToPatch4FlowTrainer': 'flow_matching.patch_mask_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import BasicTrainer

    
    from .flow_matching.flow_matching import (
        FlowMatchingTrainer,
        FlowMatchingCFGTrainer,
        TextConditionedFlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.feats_matching import (
        SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer,
        TextConditionedSparseFlowMatchingCFGTrainer,
        ImageConditionedSparseFlowMatchingCFGTrainer,
    )

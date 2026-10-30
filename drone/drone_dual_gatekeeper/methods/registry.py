"""All methods share the same optimizer, tube, simulator and estimator.
Only learning/exploration decision logic changes.
"""
from dataclasses import dataclass
@dataclass(frozen=True)
class Method:
    name: str
    label: str
    learning: bool
    strategy: str
METHODS={
 'robust':Method('robust','Robust Tube MPC',False,'backup'),
 'passive':Method('passive','Passive Learning',True,'backup'),
 'weighted':Method('weighted','Weighted Information Gain',True,'weighted'),
 'proposed':Method('proposed','Proposed Dual-Gatekeeper',True,'gatekeeper')}

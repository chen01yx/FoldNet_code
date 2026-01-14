from .tshirt import TShirt, TShirtSP, TShirtCfg
from .trousers import Trousers, TrousersCfg
from .vest import Vest, VestSP, VestCfg
from .vest_close import VestClose, VestCloseSP, VestCloseCfg
from .shirt import Shirt, ShirtCfg
from .shirt_close import ShirtClose, ShirtCloseCfg
from .hooded import Hooded, HoodedCfg
from .hooded_close import HoodedClose, HoodedCloseCfg


garment_dict = dict(
    tshirt=TShirt, tshirt_sp=TShirtSP, trousers=Trousers, 
    vest=Vest, vest_sp=VestSP, vest_close=VestClose, vest_close_sp=VestCloseSP,
    shirt=Shirt, shirt_close=ShirtClose, hooded=Hooded, hooded_close=HoodedClose,
)

__all__ = [
    "TShirt", "TShirtSP", "TShirtCfg", 
    "Trousers", "TrousersCfg", 
    "Vest", "VestSP", "VestCfg", 
    "VestClose", "VestCloseSP", "VestCloseCfg", 
    "Shirt", "ShirtCfg", 
    "ShirtClose", "ShirtCloseCfg", 
    "Hooded", "HoodedCfg", 
    "HoodedClose", "HoodedCloseCfg", 
    
    "garment_dict", 
]
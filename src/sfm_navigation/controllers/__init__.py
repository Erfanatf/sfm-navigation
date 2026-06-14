from .sfm_controller import SFMController
from .dwa.basic_dwa import BasicDWA
from .dwa.dw4do import DW4DO
from .dwa.dwa_vo import DWA_VO
from .dwa.dwa_rvo import DWA_RVO
from .dwa.dwa_orca import DWA_ORCA
from .mpc.mppi import MPPIController
from .mpc.dcbf_mppi import DCBFMPPIController
from .mpc.risk_aware_mppi import RiskAwareMPPIController
from .mpc.standard_mpc import StandardMPCController
from .mpc.nmpc import NMPCController
from .mpc.dcbf_nmpc import DCBFNMPCController
from .mpc.dcbf_mpcc_mppi import DCBFMPCCMPPIController

def create_controller(name, config, **kwargs):
    mapping = {
        'SFM': SFMController,
        'DWA_BASIC': BasicDWA,
        'DWA_DW4DO': DW4DO,
        'DWA_VO': DWA_VO,
        'DWA_RVO': DWA_RVO,
        'DWA_ORCA': DWA_ORCA,
        'MPPI': MPPIController,
        'DCBF_MPPI':DCBFMPPIController,
        "DCBF_MPCC_MPPI": DCBFMPCCMPPIController,
        "RISK_AWARE_MPPI": RiskAwareMPPIController,
        "STANDARD_MPC": StandardMPCController,
        "NMPC": NMPCController,
        "DCBF_NMPC": DCBFNMPCController,
    }
    if name not in mapping:
        raise ValueError(f"Unknown controller: {name}")
    if name == 'SFM':
        return mapping[name](config, kwargs.get('robot_params', {}))
    else:
        return mapping[name](config)
ALGORITHM_POOL = {
    "SA": {
        "class_name": "OriginalSA",
        "single_based": True,
        "tuning_space": {
            "temp_init": ("float", 50.0, 500.0),
            "step_size": ("float", 0.01, 0.5),
        },
    },
    "GA": {
        "class_name": "BaseGA",
        "pop_size_min": 20,
        "pop_size_step": 2,
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "pc": ("float", 0.5, 0.99),
            "pm": ("float", 0.001, 0.3),
        },
    },
    "PSO": {
        "class_name": "OriginalPSO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "c1": ("float", 0.5, 4.0),
            "c2": ("float", 0.5, 4.0),
            "w": ("float", 0.1, 0.9),
        },
    },
    "DE": {
        "class_name": "OriginalDE",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "wf": ("float", 0.1, 1.0),
            "cr": ("float", 0.1, 0.99),
        },
    },
    "IWO": {
        "class_name": "OriginalIWO",
        "pop_size_min": 20,
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "exponent": ("int", 2, 4),
            "sigma_start": ("float", 0.5, 4.0),
            "sigma_end": ("float", 0.001, 0.49),
        },
    },
    "TLO": {
        "class_name": "OriginalTLO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "BSO": {
        "class_name": "OriginalBSO",
        "defaults": {"m_clusters": 4},
        "pop_size_min": 20,
        "pop_size_multiple_of": "m_clusters",
        "tuning_space": {
            "pop_size": ("int", 20, 100),
            "m_clusters": ("int", 2, 4),
            "p1": ("float", 0.05, 0.5),
            "p2": ("float", 0.5, 0.95),
            "p3": ("float", 0.1, 0.6),
            "p4": ("float", 0.2, 0.8),
            "slope": ("int", 10, 50),
        },
    },
    "WDO": {
        "class_name": "OriginalWDO",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "RT": ("int", 1, 4),
            "g_c": ("float", 0.1, 0.9),
            "alp": ("float", 0.1, 0.9),
            "c_e": ("float", 0.1, 0.9),
            "max_v": ("float", 0.1, 0.9),
        },
    },
    "GWO": {
        "class_name": "OriginalGWO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "WOA": {
        "class_name": "OriginalWOA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "HHO": {
        "class_name": "OriginalHHO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "AEO": {
        "class_name": "OriginalAEO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "SSA": {
        "class_name": "OriginalSSA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "ST": ("float", 0.4, 0.95),
            "PD": ("float", 0.1, 0.9),
            "SD": ("float", 0.05, 0.5),
        },
    },
    "GBO": {
        "class_name": "OriginalGBO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "pr": ("float", 0.2, 0.9),
            "beta_min": ("float", 0.05, 1.5),
            "beta_max": ("float", 0.6, 4.0),
        },
    },
    "AOA": {
        "class_name": "OriginalAOA",
        "tuning_space": {
            "pop_size": ("int", 10, 100),
            "alpha": ("int", 2, 10),
            "miu": ("float", 0.2, 1.5),
            "moa_min": ("float", 0.05, 0.4),
            "moa_max": ("float", 0.45, 0.95),
        },
    },
    "VCS": {
        "class_name": "OriginalVCS",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "lamda": ("float", 0.1, 0.9),
            "sigma": ("float", 0.5, 4.0),
        },
    },
    "GCO": {
        "class_name": "OriginalGCO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "cr": ("float", 0.1, 0.99),
            "wf": ("float", 0.5, 2.0),
        },
    },
    "HGS": {
        "class_name": "OriginalHGS",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "PUP": ("float", 0.01, 0.3),
            "LH": ("int", 1000, 20000),
        },
    },
    "CDO": {
        "class_name": "OriginalCDO",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "ESO": {
        "class_name": "OriginalESOA",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "CMA-ES": {
        "class_name": "CMA_ES",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
        },
    },
    "ES": {
        "class_name": "OriginalES",
        "tuning_space": {
            "pop_size": ("int", 5, 100),
            "lamda": ("float", 0.5, 0.95),
        },
    },
}